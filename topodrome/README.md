# Topodrome

Models use registered HF-style `repo_id` / `revision` / `subfolder` addresses;
old model aliases are not accepted. See [model addressing](docs/MODEL_ADDRESSES.md).

Topodrome presents one queue for batch experiments and interactive Jupyter
sessions across Lab1, outbound-only Jupyter sites, and replaceable AutoDL GPU
instances. Each physical site keeps its own Ray cluster and model cache; the
central FastAPI scheduler stores metadata and fenced leases in PostgreSQL.
Accepted leases renew with replay-safe token rotation: the PostgreSQL attempt
and lease digests move atomically, the site adopts the fresh token only after a
correlated ACK, and an ACK-lost retry reconstructs the same grant while stale
tokens remain rejected.

The architectural source of truth is
[`../TOPODROME_ARCHITECTURE.md`](../TOPODROME_ARCHITECTURE.md).

For a task-oriented walkthrough of batch jobs, multi-GPU model sharding, and
IDE-compatible Jupyter sessions, see [`TUTORIAL.md`](TUTORIAL.md).

## Development

Topodrome is managed exclusively with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra scheduler --extra agent --extra jupyter
uv run pytest tests/unit tests/component
uv run ruff check src tests
```

The default environment deliberately omits Torch and Ray. Install the execution
stack only on a GPU site:

```bash
uv sync --extra agent --extra runtime --extra jupyter
```

The PostgreSQL recovery acceptance test uses a real, unprivileged PostgreSQL 16
process from the uv-managed `postgres` dependency group; it needs neither a
system PostgreSQL installation nor Docker. It applies the production Alembic
migration and then restarts the scheduler's database pools while checking queue,
lease-expiry, retry, and fencing-epoch recovery:

```bash
uv run --group postgres pytest -m postgres tests/integration/test_postgres_recovery.py
# or: ./scripts/test-postgres-recovery.sh
```

The configured CUDA 12.8 PyTorch index supports RTX 5090 workers. Scheduler-only
hosts do not download it.

## Job arrays

Independent shards and parameter sweeps can share one submission:

```bash
topodrome submit job.yaml --array 0-99 --max-parallel 4 --idempotency-key sweep-v1
```

Use `run(context, array_index, **kwargs)` and per-element resource requirements.
Arrays provide atomic submission, bounded concurrency, independent retries,
aggregate notifications and inline workbench expansion. See [Job arrays](docs/JOB_ARRAYS.md)
for controls and timing breakdowns. Job priority is per job; aged queue entries
receive starvation protection without preempting running work. Dependencies are
intentionally deferred; agents can submit follow-ups after completion notifications.

## Services

`topodrome top` → **7 System** shows active central storage usage, estimated
cleanup eligibility within 24 hours, filesystem free space and scheduler health.
Storage samples are cached for five minutes and the view never triggers cleanup.
Estimates respect retention references; remote caches/models and external
archives are excluded. See the workbench section in `TUTORIAL.md` for scope and
limitations. The diagnostics API (`GET /api/v1/system`) requires operator access.

Copy the example configuration files and keep all credentials outside the
repository. Start the scheduler and one agent with:

Set `TOPODROME_FENCE_SIGNING_KEY` on the scheduler and the same stable value as
`TOPODROME_SITE_FENCE_VERIFICATION_KEY` on every site. The value must be at
least 32 bytes and must survive scheduler and agent restarts; neither service
generates an ephemeral fallback.

The example is a production configuration: set
`insecure_development_mode: false`, publish `public_url` (and any managed
Jupyter origin) through HTTPS, and inject `TOPODROME_API_TOKEN`. The built-in
scheduler and managed-Jupyter servers speak plaintext HTTP and therefore
always refuse non-loopback bind addresses; terminate TLS at a reverse proxy
and forward only to their loopback listeners. The only tokenless mode is the
migration-compatible local-development default, which additionally requires
loopback scheduler and managed-Jupyter public origins.

Before starting or upgrading the scheduler, provide the same database DSN used
by `database_dsn` to Alembic through the protected service environment and run
the production migrations exactly once. Do not use the ORM `create_schema`
helper in production. Readiness remains HTTP 503, and dispatch remains paused,
until the database records the exact migration head expected by this build.

```bash
TOPODROME_DATABASE_DSN='postgresql+asyncpg://topodrome@127.0.0.1/topodrome' \
  uv run alembic upgrade head
uv run topodrome-scheduler --config configs/scheduler.yaml
uv run topodrome-agent --config configs/site.lab1.yaml
```

The `0019_remove_db_legacies` upgrade removes the unused SQL log-index table
and kernel URL column; logs and on-demand kernel connections are unaffected.
It refuses nonempty legacy indexes, stored URLs, or unscoped job idempotency
keys, requiring an explicit archival decision rather than discarding data.
Back up PostgreSQL and apply this revision with the matching scheduler stopped
at an idle maintenance boundary. Site agents do not need a restart. A downgrade
restores the empty table and nullable column. Historical migrations and provider
audit records are retained.

### Local multi-user access

For a scheduler shared by Unix accounts on the same server, enable
`user_access`, point `credential_store_path` at an owner-only file, and retain
the scheduler's master API token as the operator credential. The operator can
then grant or rotate one random credential for an existing local account:

```bash
topodrome user-grant alice
topodrome user-list
topodrome user-revoke alice
```

The grant response shows the bearer token exactly once. Give it directly to
that user; do not put it in shell history or a project. The user installs it
through a hidden prompt into their own mode-0600 client file:

```bash
/ssd/xiangkun/Topodrome/topodrome/.venv/bin/topodrome login \
  --url http://127.0.0.1:8042
```

Whitelisted users can upload source bundles, submit jobs, follow logs, and
create interactive kernels. Job and kernel identifiers are owner-scoped:
another ordinary user receives the same not-found response as for an unknown
identifier, while the operator can inspect all work. Idempotency keys are also
scoped by owner. Site/GPU/model inventory is a shared, sanitized view, but each
ordinary dashboard snapshot includes only that user's job and kernel rows.
Policy reload, AutoDL mutation/audit, timed GPU reservations, and whitelist
management remain operator-only. Revocation is effective on the next request,
also revokes relay access and stops only that user's managed IDE child; it never
cancels a job or kernel that is already running.

Ray keeps its logs, session metadata, and runtime state below the configured
site `scratch_root`. Linux also limits AF_UNIX socket names to 107 bytes, while
Ray appends a timestamped session path. If the scratch path is too long, the
agent automatically passes Ray a short private alias below
`ray_short_temp_root` (default `/tmp`) that points back to `scratch_root/ray`.
The target, private per-user directory, and exact symlink ownership are checked
before startup; an unexpected file, owner, writable directory, or symlink
target fails closed. Set the option to `null` only to disable aliasing and make
an overlong scratch path a startup error.

Warm actor records are scoped to one Ray cluster generation. Reconnecting to
the same cluster preserves and validates exact detached actors. Starting a
brand-new cluster drops only the prior actors' local allocation rows by exact
actor ID while retaining GPU fence epochs; no process is signalled, and any
unverified orphan remains ordinary foreign NVML pressure.

Warm-actor retention for the shared Lab1/Jupyter deployment is controlled by
the scheduler's reloadable `scheduling` policy. The default
`institutional_warm_actor_idle_timeout_seconds: 1200` releases a complete actor
after 20 continuous idle minutes. The idle timestamp is published by the site,
resets on every reuse, and survives a scheduler restart. While runnable work is
queued, `institutional_actor_floor_gpu_count: 1` protects the last reusable
institutional actor allocation; without queued work, the pool may naturally
reach zero. Multi-GPU replicas are always released atomically.

Set the legacy site-local `warm_actor_idle_timeout_seconds` to `null` and
`idle_actor_rebalance_enabled` to `false` when this central policy is enabled,
so a local reaper cannot bypass the global floor or contention threshold. A
policy reload changes the central timeout and floor without restarting an
agent: dispatch drains, already-running work finishes, and then the new policy
becomes active. Actor cleanup never stops the site agent or Ray head, affects
an active batch or interactive lease, or signals a foreign process.

An operator's timed GPU reservation is the deliberate exception to the warm
floor. It excludes that physical UUID from new placement immediately and
unloads an idle actor there, even if this leaves no institutional warm actor.
If work is still using the GPU, the reservation remains pending until that work
finishes; Topodrome never preempts it. A later job still tries every clean
Lab1/Jupyter gang before AutoDL; safe foreign-sharing capacity
remains the fallback if paid startup is unavailable or fails. Expiry or
cancellation makes the UUID schedulable again but does not synthesize a
replacement actor.

The institutional model catalog exposes the following exact, preseeded repositories
on both Lab1 and the remote Jupyter site:

- `meta-llama/Meta-Llama-3-8B-Instruct` (15 GiB);
- `google/gemma-3-12b-it` (24 GiB);
- `allenai/OLMo-2-1124-7B-Instruct` (14 GiB); and
- `Qwen/Qwen3-4B-Instruct-2507` (8 GiB).

Every default snapshot is pinned to an exact Hugging Face commit. Lab1 verifies the Hub
metadata beside `/ssd/models`; the read-only Jupyter checkpoints under
`/mnt/model/ckpts` are bound to owner-controlled external revision manifests.

An older, read-only preseeded checkpoint may not contain Hugging Face's
revision-scoped local-dir metadata. In that case set the site's
`revision_manifest` to an absolute, operator-owned copy of the exact revision
tree manifest. Topodrome accepts only a regular file owned by root or the agent
user and not writable by group/other, then hashes every required config,
tokenizer, index, and weight file against it before advertising the model as
cached. This avoids downloading an otherwise identical model merely to rebuild
3 KiB of metadata without weakening exact-revision verification.

### Remote global storage without FUSE

Lab1's configured `global_root` is the only storage authority. Remote agents
configure a verified local HTTP mirror, never an unrelated native authority:

```yaml
global_root: /fast/local/topodrome-global
storage:
  small_file_max_bytes: 1048576
  min_free_space_gb: 10
  request_timeout_seconds: 30
  verification_timeout_seconds: 120
  transfer_timeout_seconds: 21600
```

All metadata, tiny files, paged directory listings, resumable bulk reads and
prefix writes use the enrolled-agent HTTP API. No external transfer executable,
filesystem mount, or new public storage service is required.
Tiny files use persistent connections and one verified request; bulk streams
have bounded memory and exact Range/If-Match identity checks. Checksummed
upload extents adapt from 128 KiB up to 8 MiB; ACKs follow fsync. Restart
validates the owned stored prefix before resuming. Independent Lab1 SHA-256
verification and atomic publication still precede durable completion.

HTTPS is required for non-loopback origins. If the site reaches the authority
through SSH, configure `ssh_local_forward` for control and a second
`bulk_ssh_local_forward` plus `storage.bulk_url` for bulk, with separate ports
and state directories. Both routes target the same authority and disable
ControlMaster reuse. Use only a dedicated pinned forwarding identity restricted
to the scheduler endpoint, never a personal login key.
See the complete AutoDL example. The agent reconnects owned forwards while
preserving a healthy control route if bulk fails.

Unsupported site configuration fields fail explicitly. Preserve an existing
opaque journal identity as `storage.authority_id` while upgrading an idle site
with pending receipts; this is not an executable protocol fallback.
No content or database migration is needed. Upgrade the scheduler first, then
idle agents, including pending finalization/staging/outboxes in the idle check.
Sources, asset pins and attempt paths remain immutable; model caches and large
temporary tensors stay outside experiment storage.
See [HTTP-only transport](docs/HTTP_TRANSPORT.md) for recovery and tests.

### Terminal-only Jupyter site recovery

The scheduler can recover an outbound agent on a Jupyter server whose Terminal
API is its only management surface. This path is disabled by default. Enable
`jupyter_bootstrap` in `configs/scheduler.yaml`, select the exact `site_id`, and
configure only a static, credential-free remote wrapper command. The wrapper
must read its own protected site configuration, start the agent detached from
the terminal, and print the configured ready marker after startup succeeds.

Install the same uv-locked Topodrome release on the remote Jupyter host. Copy
`configs/site.jupyter.example.yaml` to an owner-only site directory and keep
the agent's site-specific environment in a separate owner-only file:

```bash
install -d -m 0700 ~/.config/topodrome
install -m 0600 configs/site.jupyter.yaml ~/.config/topodrome/site.yaml
install -m 0600 configs/models.yaml ~/.config/topodrome/models.yaml
install -m 0600 /path/to/site.env ~/.config/topodrome/site.env
topodrome-jupyter-agent-bootstrap \
  --config ~/.config/topodrome/site.yaml \
  --site-env-file ~/.config/topodrome/site.env
```

The packaged wrapper holds a private lifetime lock and records an exact boot
ID, PID, process-group ID, `/proc` start time, command digest, configuration
digest, and random generation. Concurrent and crash-recovery invocations reuse
only that exact live process. Stale or forged records are removed without
signalling their PID, and a live agent with changed protected configuration
fails closed. The public terminal command contains no credentials. The child
inherits only site/runtime variables; scheduler, provider, test-Jupyter, and
managed-Jupyter credentials are stripped. `TOPODROME_AGENT_READY` means the
local application was constructed and remained alive for the stabilization
interval. The scheduler separately requires the exact outbound connection and
a fresh authenticated heartbeat before it stops recovery or dispatches work.

Upgrade a terminal-only site through an immutable, credential-filtered source
stage; never overwrite files underneath a running release in place. Stop
scheduler admission first and prove that the site has no active work. Then use
the currently active release's wrapper and unchanged site configuration to stop
only its exact owned generation:

```bash
PYTHONPATH=/home/jovyan/.local/share/topodrome/release/src \
  /home/jovyan/.local/share/topodrome/release/.venv/bin/python \
  -m topodrome.deployment.jupyter_agent \
  --config /home/jovyan/.config/topodrome/site.yaml \
  --site-env-file /home/jovyan/.config/topodrome/site.env \
  --state-dir /home/jovyan/.local/state/topodrome/jupyter-agent \
  --stop-owned --shutdown-timeout-seconds 150
```

`TOPODROME_AGENT_STOPPED` confirms that SIGTERM was sent through a Linux pidfd
to the exact boot/PID/process-group/start-time/command identity in the private
ownership record and that process exited. `TOPODROME_AGENT_ABSENT` means there
was no live matching generation and no process was signalled. A changed or
ambiguous live identity fails closed. The stop path never falls back to a
numeric PID, never escalates to SIGKILL, and never runs `pkill` or global
`ray stop`; the agent's normal SIGTERM handler drains work and stops only its
separately proven Ray process group. The scheduler then recovers the site with
its static bootstrap command and still requires a fresh authenticated
heartbeat.

Only after that exact stop succeeds, take an owner-only deployment lock,
preserve the complete active release and site configuration as rollback, and
atomically place the staged source at the stable `release` path used by the
scheduler bootstrap command. Build a fresh `uv --offline --frozen` environment
at that final path and verify it with the identical extras before changing the
site configuration or reopening scheduler admission. Never move a virtual
environment from a staging path: generated console-script shebangs embed that
path. An existing environment may be reused only in place after
`uv sync --check --frozen` proves it matches the exact lock and requested
extras. Keep the untouched rollback until the new heartbeat and deployed source
identity are verified.

The Jupyter deployment explicitly recognizes a direct Python invocation of
`gpu_idle_load.py` as its cooperative, self-yielding idle worker. It is not
treated as a blocking foreign workload. After an assignment is fenced, the
agent starts an owned CUDA sentinel, waits for the idle worker to observe the
new CUDA process and exit on its own, and rechecks the whole GPU gang before
loading the model. Topodrome never signals or kills `gpu_idle_load.py`. If it
does not yield within the configured timeout, the assignment fails safely and
the worker is left untouched. Similar script names and unrelated GPU processes
remain foreign and are governed by the site's explicit capacity-sharing policy.

Shared Jupyter launchers can also publish advisory physical-GPU reservations by
setting `gpu_reservation_lock_dir` to an absolute, owner-controlled directory.
This integration is opt-in. When the setting is absent, external lock files are
ignored and only Topodrome's own atomic leases are hard reservations.
For GPU index `N`, Topodrome reads only `gpu_N.lock`. The small JSON document
must contain `owner`, matching integer `physical_gpu`, positive
`supervisor_pid`, positive `world_size`, and exactly one of `acquired_at`,
`created_at`, or `timestamp`; an optional positive `supervisor_start_ticks`
provides an exact Linux process identity. Only a regular, single-link,
non-symlink file in a trusted directory that is not group/other-writable is
accepted, and its same-UID supervisor must still be a live non-zombie process.
Invalid or stale entries are ignored and are never removed or signalled.
Without the optional start ticks, the reader also compares procfs creation
time with the lock and acquisition timestamps to reject ordinary PID reuse,
but a deliberately rewritten stale record can remain ambiguous.

Put the management endpoint and token only in the scheduler's mode-0600 secret
file (whose parent is mode 0700), using these production-specific names:

```dotenv
TOPODROME_JUPYTER_BOOTSTRAP_URL=https://jupyter.example/base
TOPODROME_JUPYTER_BOOTSTRAP_TOKEN=replace-me
```

The URL must not contain user information, a query string, or a fragment. A
non-loopback endpoint must use HTTPS; plaintext HTTP is accepted only for a
loopback endpoint. The token is sent in HTTP/WebSocket authorization headers
and is never placed in a command, URL, database row, or log. The scheduler runs
one bootstrap at a time, backs off to the configured maximum after failures,
and stops retrying only when the configured site has both a live outbound
connection and a fresh heartbeat for that exact connection generation. On a
failed or cancelled attempt its temporary terminal is deleted; the successful
detached-agent terminal survives scheduler restarts.

Submit the runnable manifest, then inspect the common queue. Its `source_dir: .`
is resolved relative to the manifest, uploaded content-addressably, and written
into the immutable job specification before the job is queued:

```bash
uv run topodrome submit examples/model_summary.yaml
uv run topodrome sites
uv run topodrome gpus
uv run topodrome models
uv run topodrome top --once
```

For interactive management, run `topodrome top`. Under **6 AutoDL**, press `n`
or choose **Register AutoDL**, paste the instance's SSH command and password,
then review and confirm. The password is masked. Accepted registrations appear
as starting before their site agent sends its first heartbeat; registration is
not yet confirmation that the GPU is ready.

Under **2 GPUs**, press **Space** or click the disclosure arrow to expand the
read-only foreign-process list. **Enter** opens the GPU summary and management
actions, including the same process list. It shows PID, process/script name,
user and VRAM where reported; recognized cooperative idle workers are labeled
separately. Samples refresh with accepted heartbeats and are marked stale or
unavailable when appropriate. Older agents may not report users or scripts.
Up/down still select rows; left/right still switch tabs. No process-killing
actions are exposed.

Job arrays use the same controls: **Space** or the disclosure arrow expands or
collapses members inline; **Enter** or clicking the row opens its summary.
Array summaries show counts and parallelism; individual member details retain
their own Summary, Attempts and Logs tabs. **Esc** returns to the same list
selection. Space on a child row collapses its parent group in either view.

Assets use the same interaction: **Space** or the disclosure arrow on an asset
shows older versions inline; **Enter** opens the selected version's details.
Each asset expands independently. History loads metadata only, with explicit
loading/error and result-limit notices. The `h` shortcut applies only to Jobs.

An authenticated client discovers the active source-admission snapshot before
packaging. The production policy permits 1,024 files, 500 characters per
normalized path, 256 MiB per file, and 1 GiB decoded in total. The compile-time
safety ceilings are the same byte limits and 4,096 files/characters. JSON/base64
is intentionally for project code and modest immutable inputs, not model
weights or multi-gigabyte assets. Older servers without discovery retain the
client's conservative 16 MiB/file and 64 MiB/total fallback.

### Reuse committed output as a global asset

Every output from a succeeded job is immediately reusable without copying it.
Reference one exact file or a committed directory prefix in another job:

```yaml
assets:
  lens: job://job_e8b06bd3636647baa92640c3a92e0668/Llama-3-8B-Instruct-tuned-lens/
```

The entry point reads the resolved local path from
`context.committed_inputs["lens"]`. Topodrome authorizes and pins the exact
attempt, member paths, sizes, and SHA-256 values before the job enters the
queue. A remote site downloads missing members only after assignment and before
model initialization; verified members are reused by later jobs. Each site
advertises the full immutable asset cache key, which includes both the content
digest and its pinned file/directory view. The locality bonus therefore applies
only when that exact view is reusable, not merely when another asset has the
same payload digest.

Give frequently reused output an immutable private name, or add `--shared` to
make it visible to all whitelisted users:

```bash
topodrome asset-publish tuned-lens v1 \
  --job job_e8b06bd3636647baa92640c3a92e0668 \
  --path Llama-3-8B-Instruct-tuned-lens/
topodrome assets
topodrome asset tuned-lens v1
```

Then use `asset://private/tuned-lens@v1`, or
`asset://shared/tuned-lens@v1` for a shared publication. Publication writes a
small immutable manifest only; it does not duplicate a multi-GiB payload.
Reusing the same name and version for different content is rejected. Asset
payloads are artifact outputs and are not subject to source-bundle size limits.

Retrieve job outputs or assets directly, without a GPU job or temporary asset:

```bash
topodrome download 'job://JOB_ID/results/' ./results
topodrome download 'asset://private/tuned-lens@v1' ./lens
```

Both SDKs provide `download(reference, destination)` (await for async).
Copies are editable, size/SHA-256 verified, and never overwrite a destination.
Retry the same URI/destination to resume a private partial transfer. Active
downloads temporarily protect their members from GC; they do not make outputs
permanent. See the [tutorial](TUTORIAL.md#download-job-outputs-or-assets-without-a-gpu-job).

The operator can reserve one exact physical GPU for a bounded maintenance
window by UUID or by that site's current index:

```bash
uv run topodrome gpu-reserve lab1 --gpu-index 3 \
  --duration-seconds 1800 --reason "driver maintenance" \
  --idempotency-key maintenance-2026-09-12
uv run topodrome gpu-reservations --active-only
uv run topodrome gpu-reservation reservation_0123456789abcdef
uv run topodrome gpu-reservation-cancel reservation_0123456789abcdef
```

Reservation creation, audit listing/detail, and cancellation require the
operator credential. Every record retains the authenticated requester, reason,
resolved UUID/index, timestamps, and expiry. Creation immediately removes the
GPU from new placement. A running batch or kernel is never preempted: the
reservation remains `pending`, then evicts an exact idle warm replica after the
claim ends (even if it is the last institutional warm actor) and becomes
`ready` only after a fresh heartbeat proves clearance. Cancellation is
idempotent, and expiry automatically restores placement eligibility without a
service restart. `topodrome top` shows the sanitized `reserved` effect without
exposing audit reasons or requester identities to ordinary users.

Add `--wait` to stream each attempt's merged stdout/stderr to CLI stdout while
lifecycle notices and the final structured job summary stay on stderr:

```bash
uv run topodrome submit examples/model_summary.yaml --wait
uv run topodrome submit examples/live_log_smoke.yaml --wait
```

`live_log_smoke.yaml` deliberately emits two records two seconds apart, so it
is also a quick end-to-end check that output arrives before the final job
summary rather than only after artifact collection.

The follower tracks retries and log replacement by byte range without printing
acknowledged bytes twice. `Ctrl-C` detaches only; it does not cancel the job.
Once the job is terminal, `--log-grace` (10 seconds by default) bounds how long
the CLI waits for a disconnected site's verified final flush. If that deadline
expires, use `topodrome logs <job-id>` for output that arrived later. Set, for
example, `--log-grace 60 --poll-interval 1` for a slower remote site.

The Python API has the same source behavior:

```python
import topodrome

job = topodrome.submit(
    source_dir="examples",
    entrypoint="model_summary.py",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    resources={"gpu": 1, "min_vram_gb": 15},
    idempotency_key="model-summary-v1",
)
```

### Timed operator GPU reservations

Reserve an exact physical UUID (preferred) or its current site-local index for
a bounded interval:

```bash
topodrome gpu-reserve lab1 \
  --gpu-index 7 \
  --duration-seconds 1800 \
  --reason 'exclusive debugger session' \
  --idempotency-key lab1-gpu7-debug-20260912
topodrome gpu-reservations --active-only
topodrome gpu-reservation <reservation-id>
topodrome gpu-reservation-cancel <reservation-id>
```

The scheduler resolves an index to the heartbeat's physical UUID before
committing the reservation. Active reservations are durable across scheduler
restart and appear as `reserved` (or `leased+reserved`) in `topodrome top`.
They immediately block new offers. Existing jobs and kernels are never
interrupted; once their exact lease ends, an idle warm replica touching the
reserved GPU is evicted as a whole. This operator intent overrides the warm
floor and may leave the institutional pool empty. It does not move or prewarm
an actor elsewhere. Cancellation and natural expiry are idempotent and do not
disturb a running job.

### Explicit multi-GPU execution

For a GPU job that does not need a preloaded model, use `model: null` and an
explicit positive per-GPU `resources.min_vram_gb`. `context.model/tokenizer`
are `None`; the full context API remains available. `context.device` and
`context.devices` contain real assigned worker-local PyTorch devices.
`execution: single` runs once with the entire GPU gang; `execution: distributed`
runs one private rank per GPU. See [model-free jobs](docs/MODEL_FREE_JOBS.md)
for runnable examples and rank/lifecycle rules.

Topodrome still schedules one physical GPU as one slot. A job that needs
several GPUs requests several slots as one atomic gang; it never partially
starts. The model's `parallelism` selects what happens after that gang is
leased:

- `single` (the default) is one reusable actor on one GPU;
- `distributed` with `ray_worker_group` starts one private entrypoint per GPU
  for collective-aware training or experiment code;
- `model_sharded` with `transformers_device_map` starts one reusable actor and
  invokes the user entrypoint once per request while Transformers spreads one
  model replica across the ordered visible GPU list.

For preloaded models, multi-GPU intent is never inferred from `resources.gpu`: the resource count
must exactly equal `parallelism.degree`. Sharded jobs also set a GPU-only
`max_memory_gb_per_gpu`; Topodrome rejects a resulting device map that omits a
selected GPU or offloads any module to CPU or disk. Warm reuse requires every
member heartbeat to report the same replica ID, ordered GPU membership, model
identity, and runtime fingerprint. Any mismatch causes all-or-none eviction
and a fresh gang load.

The runnable two-GPU manifest uses Llama 3 8B as a deliberately forced
oversized-model exercise with a 10 GiB cap per GPU:

```bash
uv run topodrome submit examples/model_summary_sharded.yaml
uv run topodrome kernel --model meta-llama/Meta-Llama-3-8B-Instruct \
  --gpus 2 --min-vram-gb 12 --max-memory-per-gpu 10
uv run topodrome kernel-status <session-id>
uv run topodrome kernel-stop <session-id>
```

Llama-70B is enabled for Lab1 only as of
2026-09-16; see `examples/model_summary_llama70b.yaml`. Generic multi-GPU support remains;
`examples/model_summary_llama8b_sharded.yaml` demonstrates explicit two-GPU
sharding with the supported Llama-8B model.

The kernel command supports model-sharded inference only: `--gpus` greater
than one constructs an explicit `transformers_device_map` request and requires
`--max-memory-per-gpu`. It does not infer or launch distributed training.
`kernel-status` reads the interactive session without rotating its connection
descriptor, while `kernel-stop` requests the normal fenced kernel shutdown and
prints the resulting session state as JSON.

`source_dir` is uploaded before submission and the caller's idempotency key is
applied to the resulting job request. Advanced callers may upload once with
`Client.upload_source()` and provide its `source_ref` instead. Submitting without
either is rejected locally, rather than queuing an attempt with no executable
source.

Optional per-job Ray dependencies must also be reproducible. Every
`runtime.pip` item uses pip's exact hash-checked form, for example
`package==1.2.3 --hash=sha256:<64-lowercase-hex>`, including all transitive
requirements needed by that environment. `runtime.working_dir_ref` accepts only
the content-addressed `sha256:<digest>` returned by source upload; direct HTTP,
S3, and GS working-directory URLs are rejected.

External tests are opt-in. Tests never rent capacity through the retired AutoDL API.

Production sets `scheduling.prefer_autodl_over_foreign_processes: true`:
scheduling prefers clean institutional GPUs, compatible AutoDL capacity,
lightly shared institutional GPUs, then heavily
contended institutional GPUs. Light sharing requires every gang member to have
at least two complete requested VRAM footprints free and utilization at or
below `scheduling.light_contention_utilization_percent`. Within a band, an
all-member fast-GPU match and predicted readiness (exact selected-GPU warmth,
configured model startup time, and queue delay) determine placement. Thus a
portable burst fills clean Lab1/Jupyter capacity before paid fallback.
Only already registered, ready AutoDL capacity participates in placement.
If no eligible site is available, jobs stay queued. This priority flag remains
live-reloadable without draining.

A GPU reporting a blocking foreign CUDA process is scheduler-ineligible by
default, even when its latest free-VRAM measurement appears sufficient. The
operator may opt in an exact site ID through
`scheduling.foreign_capacity_sharing_site_ids` only when that site's agent is
also configured to admit safely alongside foreign allocations. The shipped
shared deployment opts in both `lab1` and `remote-jupyter-1`; each site rechecks
live physical free VRAM plus its configured safety margin immediately before
loading. Cooperative self-yielding Jupyter idle workers are classified
separately. This scheduler-side filter avoids repeatedly offering work that a
stricter site agent must reject; final physical admission always remains with
the site agent.

Idle manually registered AutoDL instances shut down after five continuous idle
minutes through the durable SSH shutdown protocol. No dummy GPU load is generated.
Operators can request earlier graceful or forced release as described below.

The scheduler can reload this policy without restarting. Edit only the
`scheduling` mapping (including `source_limits`) and/or per-site
`estimated_startup_seconds` values in the configured model catalog, then run:

```bash
topodrome policy-reload
```

Changing only `scheduling.prefer_autodl_over_foreign_processes` and/or model
startup-cost estimates activates on the
next dispatch tick without waiting for running jobs. Existing jobs, leases,
actors, and provider operations remain untouched. Combining this toggle with
any other policy change still requires the drain described below.

This authenticated command calls `POST /api/v1/policy/reload`. Topodrome first
validates both files; malformed input or changes to any other scheduler/model
field retain the prior generation. A valid changed policy becomes a pending
generation and reports `status: draining`. New batch and kernel dispatch pauses
while work already dispatched under the active generation terminates. Queued
jobs remain queued; lease/kernel/provider reconciliation continues, and no
running work is canceled, preempted, or moved. In-flight and durably unresolved
provider startups are blockers too, so a reload cannot forget paid capacity and
fall back onto a noisy GPU. Topodrome then activates the pending generation
atomically and resumes dispatch; submitting the same pending content is
idempotent, while a different candidate conflicts until the drain completes.
`topodrome top --once` or `GET /api/v1/status` reports the active generation,
pending generation, timestamps, and current blocker count. Active leases,
queued/running jobs, and provider operations are never rewritten by a reload.
The owner-only active/pending state survives scheduler restart, so edited YAML
cannot be silently adopted after a crash midway through a drain. Runtime code
and credentials are not hot-patchable.

### Manual AutoDL capacity

Rent and start capacity in AutoDL yourself, then register it:

```bash
topodrome autodl register --ssh 'ssh -p PORT root@HOST.seetacloud.com'
# Enter the password at the hidden prompt.
topodrome autodl
```

Stop accepting new jobs and shut down after current work finishes:

```bash
topodrome autodl release --site autodl-manual-EXACT_ID
```

To preempt current jobs and retry them on other eligible sites:

```bash
topodrome autodl release --site autodl-manual-EXACT_ID --force
```

Release requests persist across scheduler restarts. Repeating a request is
idempotent; `--force` may upgrade a pending graceful request. Forced release uses
the existing checkpoint/preemption protocol, retaining the job ID and starting a
new attempt. Work since the last verified checkpoint may be lost; a job without
one starts again. Jobs wait in the queue if no eligible alternative is available.
Explicit site/region affinity remains binding. Interactive kernels cannot migrate:
graceful release waits for them, while forced release stops them.

Release waits for the normal durability barrier and owned-runtime shutdown, then
submits `/usr/bin/shutdown` over SSH. It does not delete an AutoDL rental or shared
storage. The console remains authoritative for billing; an ambiguous SSH shutdown
is surfaced for operator attention. Stopped capacity disappears from active views,
while audit records remain. `autodl unregister --site SITE` is different: it removes
an already-disconnected registration without shutting down its host.

Idle shutdown remains automatic after five minutes, with required transfers and
active work blocking shutdown. There is no automatic renting, power-on, replacement,
pool limit, provider token, or `autodl on/off` command. Rent and register another
instance when more capacity is needed. GPU type/count comes from inspected inventory.

The Python clients expose `release_autodl(site_id, force=False)` (await for the
async client). The operator-only endpoint is
`POST /api/v1/providers/autodl/registrations/{site_id}/release` with `{"force": false}`.

### AutoDL production bootstrap

Manual registration installs an instance-specific bootstrap under
`/root/autodl-fs/topodrome/manual/<site-id>/`. Build that artifact from the
exact checkout and deployment configuration before registering manual capacity:

```bash
uv run topodrome-build-autodl-bootstrap \
  --site-config configs/site.autodl.example.yaml \
  --model-config configs/models.example.yaml \
  --output-dir /path/to/autodl-bootstrap-staging
```

The builder obtains `uv` from `PATH` unless `--uv-binary` is supplied. It rejects
embedded credentials, non-AutoDL runtime paths, a placeholder scheduler URL, a
missing HTTP storage mirror, model paths outside `/root/autodl-fs` and
`/root/autodl-tmp`, or a Hugging Face origin that does not use
`https://hf-mirror.com`. It also scans every bundled text file, including
`uv.lock`, for credential-shaped material. It emits a SHA-256-addressed release,
checksum, atomic `current` pointer, and `bootstrap.sh`. Copy those generated files
as a set to the Beijing-B regional directory
`/root/autodl-fs/topodrome/bootstrap/`; they contain no secrets.

For a faster cold start after releasing and re-applying a worker, optionally bind
a Lab1 runtime seed into the release:

```bash
uv run topodrome-build-autodl-bootstrap \
  --site-config configs/site.autodl.example.yaml \
  --model-config configs/models.example.yaml \
  --runtime-seed /path/to/runtime-site-packages.tar.gz \
  --runtime-seed-platform x86_64-manylinux_2_31 \
  --output-dir /path/to/autodl-bootstrap-staging
```

The seed format is a gzip-compressed tar containing exactly one relative
`site-packages/` tree. The builder validates every member and prints the
credential-free logical seed destination. That destination is content addressed by
the seed SHA-256, the current `uv.lock` SHA-256, Python major/minor, Linux
architecture, and libc major/minor. `--runtime-seed-platform` names the worker's
exact uv wheel target; Beijing-B currently uses `x86_64-manylinux_2_31`. Omit it
only when the build host and worker have the same libc ABI. A manylinux seed
also supports a newer glibc on the same architecture, with the same Python
major/minor and frozen uv dependency verification; older glibc and different
architectures remain rejected. Stage the verified archive once under Lab1's
configured global root at the manifest's `runtime_seed.logical_path`; the seed
is not included in the release. AutoDL keeps a verified regional cache for reuse.

The builder stages only the seed's installed-distribution metadata and asks uv
to check the selected agent/runtime extras against the frozen lock and target
wheel platform. An incompatible seed is rejected before publication when the
release is offline-only. With the explicit `--allow-online-fallback` opt-in, the
manifest records that the seed needs target-specific repair and the worker runs
one frozen online uv sync after extraction. For example, a Lab1 glibc 2.35 seed
may contain a `manylinux_2_34` wheel that AutoDL glibc 2.31 must replace with a
compatible hash already authorized by `uv.lock`.

On a cold worker the bootstrap first checks the content-addressed regional
cache under `/root/autodl-fs/topodrome/runtime-seeds`. On a miss it downloads
from Lab1 over resumable HTTP and persists the verified archive there for future
instances. It verifies size and SHA-256 before every extraction, creates the
target venv with the bundled `uv`, safely expands only `site-packages/` into the
venv's actual purelib directory, and runs `uv sync --frozen --offline`. uv
console scripts created in the temporary venv are atomically replaced with
relocation-safe Topodrome-owned launchers before the venv is published. The
working copy and staging trees remain ephemeral. A lock, platform, archive, or
member-integrity mismatch always fails closed. `--allow-online-fallback` may be
added only when package-index fallback is acceptable; it applies to an
unavailable seed or missing locked distributions, never to an integrity
mismatch. Without a configured seed, the existing frozen online `uv sync`
behavior is retained.

Provision `/root/autodl-fs/topodrome/secrets/site.env` separately, owner-only
mode 0600 in a mode-0700 directory. It contains enrollment, fence verification,
and Hugging Face credentials.
The stdlib seed helper reads only enrollment in-process; uv never inherits it.

The site YAML expects a dedicated forwarding identity at
`/root/autodl-fs/topodrome/secrets/control_ed25519` and a pinned host-key file
at `/root/autodl-fs/topodrome/secrets/lab1_known_hosts`. Allow local forwarding
only to Lab1's `127.0.0.1:8042`. The agent owns separate control and bulk SSH
processes on local ports 18042 and 18043; the scheduler remains loopback-bound.

For an OpenSSH version supporting `restrict`, the Lab1 control-key entry should
have the following capability shape (substitute the generated public key):

```text
restrict,port-forwarding,permitopen="127.0.0.1:8042" ssh-ed25519 <site-control-public-key>
```

The generated startup command needs no legacy Python environment on regional
storage. On every newly created instance it verifies and expands the release
under `/root/autodl-tmp/topodrome`, uses the bundled `uv` to create or validate a
locked ephemeral `.venv`, places uv/Hugging Face/Ray caches on ephemeral
disk, and runs the site agent in the foreground. The agent mirrors required
global objects over HTTP without FUSE, establishes its
authenticated outbound scheduler connection, and owns the site-local Ray
lifecycle. A retained power cycle can reuse ephemeral files; a
released/recreated instance rebuilds them. Only the small immutable bootstrap
releases, protected configuration, forwarding identity, agent spool/state, and
model cache remain under `/root/autodl-fs`; neither a venv, global mirror, nor
package cache is persisted.
Release verification, extraction, HTTP seed transfer, and `uv` reconciliation
run in an allowlisted environment. `site.env` is sourced only after those steps,
immediately before the foreground agent starts.

Inspect retained capacity and the sanitized operation journal with:

```bash
uv run topodrome autodl
```

The five idle signals are reported by the site agent and ORed with scheduler
job, kernel, and artifact state. Missing or stale telemetry is treated as
activity. Power-off requires `drain_ready` and `durability_ready` acknowledgements;
the agent synchronizes regional filesystem metadata and stops Ray before the
provider call. The idle start timestamp is persisted, so a scheduler restart
cannot accidentally shorten or discard the configured continuous idle window.
The separate clean-alternative stability observation must be rebuilt after a
scheduler restart before early drain is allowed.

GPU admission margins are reported in each site heartbeat and enforced by both
the scheduler and the site's final physical admission check. The Jupyter
example reserves 5 GiB, so the configured 15 GiB Llama model needs at least 20
GiB of physical free VRAM there.

## Interactive kernel connections

Create an interactive session with `topodrome kernel` or `POST /api/v1/kernels`,
inspect it with `topodrome kernel-status <session-id>`, and stop it with
`topodrome kernel-stop <session-id>`. Wait for its state to become `running`,
then explicitly request a connection descriptor from
`GET /api/v1/kernels/{session_id}/connect`. The descriptor contains separate
`base_url`, `ws_url`, `websocket_url`, `token`, `kernel_id`, and `expires_at`
fields. Requesting another descriptor rotates the authorization generation and
revokes every older URL and established channel. Credentials and responses are
never stored in the kernel row, request journal, agent event spool, or logs.

Opening the returned `url` in a browser gives a small native cell console. Its
short-lived query credential is exchanged for a scoped HttpOnly cookie and a 303
redirect to a query-free URL. Programmatic clients use `websocket_url` directly,
or authenticate REST and WebSocket gateway requests with
`Authorization: token <token>`.

The scoped `base_url` implements the Jupyter Gateway kernel subset used by a
central Jupyter Server: gateway metadata, list and individual kernelspec reads,
kernel list/attach/get/delete/interrupt, and native multiplexed channels. It
supports both classic JSON/binary frames and
`v1.kernel.websocket.jupyter.org`. A central Jupyter Server can be pointed at
one pre-authorized Topodrome session with values from the connection descriptor:

```bash
export JUPYTER_GATEWAY_URL="$base_url"
export JUPYTER_GATEWAY_WS_URL="$ws_url"
export JUPYTER_GATEWAY_AUTH_TOKEN="$token"
export JUPYTER_GATEWAY_AUTH_SCHEME=token
uv run --extra jupyter jupyter lab \
  --ServerApp.kernel_manager_class=\
topodrome.jupyter.server_bridge.AuthenticatedGatewayMappingKernelManager
```

The Topodrome kernel-manager class is required because Jupyter Server 2.x does
not otherwise forward its configured gateway Authorization header to the
native WebSocket. This manual form remains bounded by the descriptor's short
TTL and is intended for diagnostics; use the managed server below for a stable
IDE session.

### IDE-compatible managed Jupyter Server

For PyCharm and clients that require a complete stock Jupyter Server rather than
the scoped kernel gateway, enable the scheduler's opt-in `managed_jupyter`
section. Install the `jupyter` extra, use the bundled
`topodrome-managed-jupyter` executable, and put a stable, random value of at
least 32 bytes in the scheduler's protected secret file:

```dotenv
TOPODROME_MANAGED_JUPYTER_SERVER_TOKEN=replace-with-a-stable-random-secret
```

The API starts one stock server for an already-running Topodrome kernel:

```text
POST   /api/v1/kernels/{session_id}/notebook-server
GET    /api/v1/kernels/{session_id}/notebook-server
DELETE /api/v1/kernels/{session_id}/notebook-server?stop_kernel=true
```

The same operation is available from the SDK as
`handle.notebook_server()` and from the CLI:

```bash
uv run topodrome notebook-server <session-id>
```

The response contains a query-free `url`, a separate `token`, and a
`token_url`. It has `Cache-Control: no-store`; none
of those credentials are written to PostgreSQL, process arguments, runtime
state, or logs. The managed entry point retains the stock contents, sessions,
kernelspecs, kernels, and native-channel APIs, but disables local terminals and
redacts the IDE token from Jupyter's server-discovery metadata. Configure
`public_url_template` as a TLS-protected route
that exposes the managed port to the IDE. For an ordinary user, the stock
server roots its Contents API in that owner's mode-0700 subdirectory of the
configured notebooks directory; the operator retains the legacy notebooks
root. It supplies the normal contents, sessions, kernels, and native channel
APIs expected by PyCharm.
Topodrome does not bundle JupyterLab browser assets; install JupyterLab
separately if a browser Lab UI is also required.

When development mode returns a loopback URL, the CLI also prints a
credential-free workstation command such as
`ssh -N -L 8891:127.0.0.1:8891 lab1` and the matching local URL. Start that
tunnel before pasting the local URL and separately returned token into the IDE.
Production deployments should expose the configured TLS-protected route
instead of relying on a workstation tunnel.

The IDE token and public port remain stable throughout the live kernel session,
except that granting/rotating or revoking and later re-granting the owner's
whitelist credential deliberately changes the IDE token.
The child server uses a different HMAC-derived, session-and-process-generation
bridge capability against the scheduler. That capability exists only in the
child environment, is checked against the currently supervised PID on every
gateway request/frame, and is revoked before stop or process replacement.
Ordinary short-lived `/connect` token rotation never restarts or disconnects
the managed server. A transient scheduler/gateway outage likewise does not
rotate a healthy child.

Revoking the owner's whitelist entry invalidates subsequent relay requests and
terminates only that exact managed Jupyter child. It deliberately leaves the
underlying Topodrome kernel and any batch work running, so re-granting access
can resume control without corrupting work.

Deleting a stock Jupyter session (for example, when an IDE closes a notebook)
only detaches that notebook from the managed kernel. It does not stop the
Topodrome kernel, release its GPU lease, or terminate the managed server. Use
the explicit Topodrome kernel or notebook-server stop operation for lifecycle
changes.

A graceful scheduler restart deliberately leaves active, detached Jupyter
Server children running and persists only their non-secret PID/start-time,
port, generation, and activity metadata in a mode-0700 runtime directory. The
replacement scheduler recovers the same PID and port. Explicit managed-server
deletion, direct kernel stop, terminal kernel state, and configured idle expiry
revoke the bridge and terminate the exact PID; a service manager can request
controller-wide termination during a permanent deployment shutdown.

The reference user-systemd unit is
`configs/systemd/topodrome-scheduler.service`. Its `KillMode=process` is part of
the recovery contract: `control-group` would kill the detached stock servers
before the replacement scheduler can adopt their exact PIDs. Before a permanent
service shutdown, explicitly stop the managed notebook servers (normally with
their kernels) rather than using cgroup-wide signaling.

Gateway attach accepts and discards bounded, non-sensitive `KERNEL_*` hints
from Jupyter Server; they are never forwarded to a site. Service-mode cells are
limited to literal `describe()`, `generate(...)`, `tokenize(...)`, operation
JSON, or `%topodrome` commands, and are serialized on the shared actor. Raw mode
runs stateful Python against its exclusive private actor until explicit stop or
idle expiry. Shell work is FIFO across clients, control interrupts bypass it,
shell replies stay private to the requesting socket, and IOPub is broadcast to
all authorized clients.

The gateway deliberately returns HTTP 501 for kernel restart because live
model/Python state cannot be migrated safely; create a replacement kernel
instead. The scoped gateway itself does not reimplement notebook contents,
sessions, user identity, or JupyterLab static assets. Manual deployments or the
managed path above use stock Jupyter Server for those APIs, while Topodrome
supplies its authenticated remote kernel gateway. Hermetic tests exercise a
real Jupyter Server 2.21 contents and sessions lifecycle, authenticated native
WebSocket execution, Jupyter framing, service and raw execution, interrupt,
multi-client routing, credential rotation/revocation, stable managed-server
identity beyond the ordinary connect TTL, restart-safe no-replay, and cleanup.
