# Topodrome tutorial

Operator code/configuration releases can be queued with `topodrome upgrade queue
/absolute/plan.yaml --pause targets|all|none`. An independent updater prepares the
release, waits for affected computation and required I/O to finish, applies and
verifies it, then resumes dispatch. Submissions stay open and running jobs are
never preempted. `upgrade status` shows blockers; `upgrade cancel ID` is allowed
before live mutation. See [idle-gated upgrades](docs/IDLE_UPGRADES.md) for protected
plans, scope, the separate runner and crash/rollback safety.

`top` shows only nonterminal interactive kernels (including idle/start/stop
transitions). Freed rows expand the jobs table. Running/assigning/canceling jobs
come first, then queued and retry-wait jobs, then completed history if space
remains. These display priorities do not change scheduling or delete history.

Remote-storage administration: remote agents configure `storage` and use
HTTP for tiny files, resumable large files, metadata, directories and independent
authority verification. Lab1 uses its native path. No external transfer tool,
configuration or FUSE mount is required. Small transfers have persistent bounded
connections; large streams and checksummed prefix uploads have isolated pools.
When routed through SSH, use separate owned control and bulk forwards so bulk
loss/congestion does not monopolize the control connection.
SSH bulk routes may opt into measured `storage.bulk_reconnect` renewal. It
compares a fresh direct connection before switching new requests, while open
streams drain and running computation remains untouched. Reconnection does
not guarantee a different network route; see the transport guide below.
No integrity/authentication failure silently switches protocols.

`storage.small_file_max_bytes` defaults to 1 MiB (maximum 4 MiB; zero uses HTTP
streaming throughout). Completion still requires authority-observed durable
bytes, not merely receipt or 100% progress. Inputs verify exact content pins;
unfinished user output is not uploaded while code still changes it.
Transport configuration requires idle-gated service/agent deployment, not a
scheduling policy reload. Never restart with active computation, staging or
pending finalization/outbox delivery. Models and scratch remain separate.
See [HTTP transport and validation](docs/HTTP_TRANSPORT.md).

This walkthrough uses the deployed Topodrome control plane on Lab1. Run the
commands on Lab1. The installed `topodrome` command and the project-local
`uv run topodrome` command are equivalent; examples below use the installed
command for brevity.

The current institutional suppliers are `lab1` and `sqz-128` (two A800 80-GB
GPUs). SQZ-128 uses the existing four-model cache and an isolated agent/runtime
under `/cephfs/tianhaolin/sxk`; all its managed storage and scratch stay under
that directory. The superseded `remote-jupyter-1`, `sqz`, and `sqz2` registrations
are retired, not renamed or deleted: their historical jobs and files are
preserved, their enrollment credentials are revoked, and they cannot rejoin or
receive assignments. Ordinary offline suppliers still appear in Sites with
their GPUs hidden.

### Access from a whitelisted server account

An operator grants or rotates access for an existing Unix account and delivers
the returned token directly to that user. The token is displayed only once:

```bash
topodrome user-grant alice
topodrome user-list
```

While logged in as that account, install the token in the user's owner-only
client configuration:

```bash
/ssd/xiangkun/Topodrome/topodrome/.venv/bin/topodrome login \
  --url http://127.0.0.1:8042
```

If `topodrome` is not already on that account's `PATH`, the shared absolute
command above can be used directly or linked from the user's own
`~/.local/bin/topodrome`.

The user can then submit jobs and create kernels normally. Job idempotency keys
are private to that username, so two users may independently choose the same
key. Direct job status, attempts, logs, cancellation, kernel connection, and
kernel stop operations are limited to the owner; probing another user's ID
returns the same `404` as an unknown ID. Site, GPU, model, and sanitized
inventory views remain shared; the dashboard includes only the caller's jobs
and kernels. Only the operator can manage users, reservations, provider
capacity, or scheduling-policy reloads. Revoke API access with
`topodrome user-revoke alice`. Revocation also closes only that user's managed
IDE server and rejects existing relay credentials, but deliberately leaves
their running kernel and batch jobs intact.

## 1. Inspect the cluster

Start with the live, single-page dashboard:

```bash
topodrome top
```

Press `Ctrl-C` to leave it. For a scriptable snapshot, use:

```bash
topodrome top --once
topodrome sites
topodrome gpus
topodrome models
topodrome autodl
```

To discover the registered training checkpoints for one repository, use
`topodrome models --repo EleutherAI/pythia-160m`. Results include the exact commit,
registered `refs` such as `step20000`, `is_default`, and per-site availability.
Use a returned label in `model.revision`; no Hugging Face lookup is needed.
Both Python clients offer `models(repo_id=...)`. Deduplicate repeated site/replica
rows by repository, revision and subfolder before generating a sweep. See
[model addresses](docs/MODEL_ADDRESSES.md#discover-checkpoint-labels) for details.

### Interactive workbench

On a terminal, `topodrome top` opens the Textual workbench (`topodrome[ui]`).
Use `top --plain` for the previous text monitor; `top --once` is unchanged.
Without a terminal, ordinary `top` prints one snapshot and exits.

Use `←`/`→` to switch tabs directly from the list; they wrap at either end.
`1` Jobs, `2` GPUs, `3` Sites, `4` Assets, `5` Reservations, `6` AutoDL, and `7` System
also switch directly to a view. `?` opens shortcut help; `Esc` closes it. Number
shortcuts are inactive while typing in inputs or confirming an action; arrows
retain normal text editing and dialog controls there.
Inside job details, the visible navigation is `1` Summary, `2` Attempts, and `3`
Logs. Other resource details show a curated summary only, with no raw API dumps.
`←`/`→` cycles these detail tabs without leaving the selected resource.
These shortcuts focus the selected content for scrolling; `Esc` returns
to the main views, where `1–7` select Jobs through System again.

**System** is a read-only operator view of the active central store: allocated
file bytes (hardlinks counted once), category breakdown, filesystem free space,
and estimated cleanup eligibility now and within one day. The one-day total
includes already eligible data; a separate row shows newly eligible bytes.
These are estimates using current references and the actual retention rules,
not guaranteed deletions. Published assets, live work and recovery/input pins
remain protected; automatic cleanup can be disabled or limited per pass.
Remote caches, external models, archived stores outside the active root and
external database storage are **not** included. Host filesystem free space is
shared with other applications. Sparse files use allocated, not logical, bytes.
The background storage sample is cached for five minutes; service uptime,
database/schema health, connected agents, policy state, RSS and host load refresh
with the view. Failed samples retain their previous values marked stale;
unknown estimates display `—`, not zero. `Enter` explains a metric. Merely opening
or refreshing the tab never starts cleanup. An older scheduler without the
diagnostics endpoint displays an unavailable notice until safely upgraded.

The Jobs tab defaults to active work, including queued jobs and retries. It
preserves server queue ordering. `h` toggles terminal history within the bounded
status snapshot; `s` cycles queue/name/site sorting. `/` focuses the filter,
`Enter` finishes filtering, `Esc` clears it, and `r` refreshes. `↑`/`↓` select
rows; `Enter` opens their details and `Esc` returns. Tab moves focus.
Selection uses complete IDs even though the table displays only four characters.
The layout keeps your terminal's default background with no colored panels or
zebra stripes. Restrained foreground accents mark the active tab and status:
green for ready/success, yellow for waiting/warnings, red for failure/offline,
and cyan for running work. Names and logs stay plain. `--no-color` disables
these accents. GPU ownership is green for free, cyan for leased/warm, and yellow
for reserved/foreign (including mixed ownership). GPU utilization is green below
30%, yellow from 30% to below 80%, and red from 80% to 100%; red here indicates
high activity, not a fault. These GPU colors also apply to the plain monitor
when color is enabled. One full-width list is shown at a time; the filter is
hidden until `/`. `d` switches to full-page details and back: full identity,
metadata, attempts, and a bounded live log tail (64 KiB). Refresh does not force
the log scroll to the bottom. Use `Shift+←`/`Shift+→` for horizontal scrolling of wide content.
`q` or Ctrl-C exits.

GPUs, Sites, Assets, Reservations, and AutoDL have separate tabs. Offline/stopped
sites remain listed, but their GPU inventory is hidden. Jupyter-server management
is not shown. Jobs on any site remain visible. Lists show up to 100 assets and
1,000 active reservations; this is not a complete historical browser. A failed refresh retains
the last snapshot with a stale indicator, rather than presenting it as current.

`Enter` opens details with the relevant actions displayed directly above the
summary: Reserve GPU, Extend/Cancel reservation, Release instance, or Remove offline
registration. There is no separate action-menu step. `↑`/`↓` chooses an action;
`Enter` opens its form, never immediately executes it. `Tab` moves to the details
for scrolling; `a` focuses the action list again, and `Esc` returns to the main
list. From the main list, `a` also opens this details-and-actions view.
Clicking a GPU row opens the same view. A reserved GPU shows its remaining time,
expiry (with timezone), reason, and **Extend reservation / Cancel reservation**
instead of Reserve. Extend adds the entered duration (`30m`, `1h`, etc.) to the
existing expiry; review shows the new expiry before confirmation. The same
actions are available in Reservations. No temporary unreserve/re-reserve occurs,
and the existing force mode is preserved. Expired reservations cannot be extended.
After reserving, fresh data returns focus to Extend/Cancel; `↓`, then Enter opens
Cancel. After extending, the same action stays selected; after canceling from
GPU details, Reserve GPU is selected again. Opening a reserved GPU while its
metadata loads also focuses these actions once ready. AutoDL actions and toolbar
buttons use the same focus handling. If you navigate elsewhere while a request
is pending, its completion leaves your focus alone. Empty lists and removed
resources keep a visible keyboard target, never a hidden input or disabled button.
`a` or Shift+Tab from the summary returns to the action list explicitly.
Temporary read failures preserve the selected action only for the same exact
reservation; replacement reservations require a new selection and review.
Reservation details are operator-only; unavailable or stale reads disable these
actions, rather than guessing which reservation to change. A changed expiry
rejects a stale extension; refresh and review again. Replaying an identical SDK/API
request never adds time twice. The action cursor remains visible even with
`--no-color`; backgrounds use terminal defaults.
This is a **human management console**: jobs, attempts, logs and assets remain
read-only, without mutation actions. AI agents continue to submit jobs and
manage assets through the existing CLI/SDK, not through the workbench.

Select a GPU to reserve it, a reservation to cancel it, or an AutoDL site to
release it. Quick actions are also clickable in the main list's compact bottom toolbar.
The command palette is temporarily removed; `Ctrl+P` does nothing. Use the
bottom toolbar instead: Register AutoDL on the AutoDL page, Models on GPUs/Sites,
and Reload policy on Sites. Storage cleanup remains a CLI operation.
Registration accepts the SSH command and a masked password;
credentials are never displayed in results. Offline/stopped AutoDL rows also
offer removal of the registration; removal is explicitly **not a shutdown**.
All operator permissions and in-use checks remain enforced by the server.

Open details remain attached to the selected full ID/UUID through background
reordering and filtering. If that resource disappears from the latest snapshot,
its cached details remain visible with a warning and its actions become
unavailable; they never switch to a different resource. `Esc` returns to the updated list.
Dialogs also capture the full target ID/UUID
before confirmation, so background reordering cannot retarget an action.
Reservations accept seconds or whole-number durations such as `30m`, `12h`,
or `1d`, with an editable reason defaulting to `Manual reservation`. Force is off by default;
forced reservations/releases may preempt jobs for retry and stop interactive
kernels. Graceful release drains work and required I/O before shutdown.
Before confirming a forced reservation, the review fetches the current snapshot
and matches live attempts to the exact GPU UUID. It lists matching job names,
full IDs, states, model and reported checkpoint availability, with a whole-job
warning for multi-GPU leases. Confirm is unavailable while checking. Missing
details or failed reads show an explicit warning, never an assertion that the
GPU is idle. This bounded, permission-scoped preview is advisory; assignment can
change before application. Reviews older than 15 seconds refresh before accepting
confirmation. Use PgUp/PgDn to read longer reviews.

Forms use two steps: edit, then review the exact target and consequences before
confirming. Reservation reasons default to `Manual reservation` if left blank.
`↑`/`↓` moves between controls; `Tab`/`Shift+Tab` also works. Enter in a text field advances. Space
toggles the force option, displayed unambiguously as `[ ]` or `[x]`, even without
color. Tab to Review and press Enter. On review, use Tab or the left/right arrows
to choose Edit or Confirm, then Enter to activate that button. **Confirm is selected
by default**, so a second Enter applies the reviewed action. Fields still receive
initial focus for editing; force remains off by default. During a forced-reservation
impact check, Confirm is disabled and Enter does nothing. Once ready, Confirm gains
focus unless you have navigated elsewhere; the check never applies the action itself.
No Ctrl+Enter or Alt-key shortcut is used for form actions. Text fields retain
normal left/right cursor movement; left/right button navigation only applies when a button has focus.
`Esc` returns from review to editing, or closes an editing form without changes.

The workbench never executes shell commands or accesses the
database directly. Existing server ownership/operator checks remain authoritative;
ordinary users cannot gain administrative rights through a visible button.

Actions run in the background and are never retried automatically. An uncertain
network result tells you to inspect the target before retrying. Normal quit waits
while an action is pending; abruptly killing the terminal can still
leave a server mutation in flight. UI-only changes take effect when the workbench
is reopened; API additions such as reservation extension also require a scheduler
upgrade during an idle window, but no site-agent restart. Install the UI extra
with `uv sync --extra ui` in a development checkout,
or install the CLI with `uv tool install '/path/to/topodrome[ui]'`.

The scheduler first prefers clean compatible institutional GPUs (Lab1 or SQZ-128),
then registered AutoDL capacity, then GPUs with foreign processes where sharing is explicitly enabled (light before heavy). `scheduling.prefer_autodl_over_foreign_processes: true` enables this
ordering. Topodrome does not rent capacity when no site is eligible; those jobs remain queued. In the GPU table, `leased` and `warm` are
Topodrome-owned; `foreign` means unrelated CUDA work is present; `reserved`
means an explicit external reservation.

If an agent disconnects or its connection changes before an assignment is sent,
the scheduler requeues that proven-unsent offer immediately instead of waiting
for its timeout. The job keeps its ID, staged inputs and failure budget. A send
failure with unknown receipt still waits for acceptance or expiry to prevent
duplicate execution; running work is never revoked by this recovery.

An operator can validate and stage a changed scheduling policy without
restarting the service:

```bash
topodrome policy-reload
```

### Operator: live site preference

In `topodrome top`, go to **Sites** (`3` or the arrow keys). Use **↑↓** to select
a site, then **Space** to pick it up: a `↕` marker appears, and ordinary **↑↓**
moves that site. **Space** drops it so you can select another. **Enter** saves
the order; **Esc** cancels. No modifier keys, new tab, editor screen, or extra
review are required. **Shift+↑↓** remains an optional direct shortcut when your
terminal forwards it.

For mouse controls, click a site, then **Reorder sites** in its details. This
returns to the same table with inline **↑ Up / ↓ Down** and **Save / Cancel**.
Moving/selecting rows alone never saves. Clicking a row during reordering drops
the picked-up site and selects the clicked row; press Space to move that one.

Starting a reorder clears the filter so all registered sites are visible.
Existing explicit preferences come first; an unranked site is an automatic
fallback. Reordering ranks the complete captured list. Offline sites stay in
the list and are skipped until eligible. Newly registered/unlisted sites remain
automatic fallbacks. **Auto**, then **Enter** or **Save**, clears the entire
order and restores measured-cost placement. Leaving the Sites tab discards
unsaved edits; background refresh does not overwrite a draft. A concurrent
policy edit disables Save: cancel, refresh, and reorder again. The Sites table
follows the saved order and its **Pref** column shows ranks (`#1` first).
These are operator-only actions; they take effect live and persist across
scheduler restarts. Reopen an existing workbench to load new UI controls.

Equivalent CLI commands:

```bash
topodrome site-preference                    # inspect the active order
topodrome site-preference sqz-128             # SQZ first; other sites are fallbacks
topodrome site-preference sqz-128 lab1        # explicit first and second choices
topodrome site-preference --automatic        # restore automatic placement
```

Preference ranks **eligible** sites within the same contention/cost band. It
never bypasses reservations, foreign-process exclusions, model/GPU/host-memory
requirements, or job affinity. A preferred site can replace an incompatible
idle actor, but never a busy one. If it has no eligible capacity, fallback is
immediate rather than waiting for it. Clean institutional GPUs still precede
paid capacity. Job priority, array limits, and starvation protection are unchanged.
Running jobs, leases and input transfers already in progress are not moved.
Concurrent edits use a policy generation check; a conflict requires refreshing
and explicitly retrying, not an automatic mutation retry.

The API is `PUT /api/v1/policy/site-preference` with `preferred_site_ids` and
the current `expected_generation` from `/api/v1/status` → `policy`.
An empty list restores automatic placement. Unknown or retired sites are
rejected; an offline registered site can remain preferred for when it returns.
If `scheduling.preferred_site_ids` is omitted from YAML, ordinary policy reloads
preserve the live order. An explicit YAML value (including `[]`) replaces it.

Changing only `scheduling.prefer_autodl_over_foreign_processes`, `preferred_site_ids`, model
startup-cost estimates, and/or removing sites from
`scheduling.foreign_capacity_sharing_site_ids` activates at the
next dispatch boundary without waiting for running jobs or changing their
leases. Combining it with other changes still requires draining.

A changed policy normally returns `status: draining` with both active and
pending generations. Topodrome pauses new job and kernel dispatch, lets work
already dispatched under the active generation terminate, then atomically
activates the pending generation and resumes dispatch. Queued jobs stay queued
throughout; they are not canceled or rewritten. `topodrome top --once` shows
the active/pending generations and blocker count. Invalid changes leave the
active policy untouched.

The same drain-safe reload applies to `scheduling.source_limits`. Current
clients discover the active values before reading a source tree. This
deployment allows 1,024 files, 500 characters per normalized path, 256 MiB per
file, and 1 GiB decoded in total; running jobs and already accepted
content-addressed sources are unaffected by a later limit change.

## 2. Submit the one-GPU Llama example

Idle model actors expire after one minute on Lab1 and twenty minutes on
SQZ-128. Active jobs are never interrupted. The queue-active last-actor
protection still applies. Operators configure overrides in scheduler YAML:

```yaml
scheduling:
  institutional_warm_actor_idle_timeout_seconds: 1200
  institutional_warm_actor_idle_timeout_seconds_by_site:
    lab1: 60
```

`topodrome policy-reload` applies per-site timeout changes live without draining
running jobs. Omitted sites inherit the default timeout.

### Operator: suppliers that reclaim idle GPUs

Only an explicitly opted-in supplier runs cooperative
idle compute, including when the global queue is empty. This is not a fake job
or a model actor: a child per idle GPU performs bounded FP16 GEMMs (4096-square,
approximately 96 MiB of tensors plus its CUDA context). Real jobs retain
priority and the helper is stopped before physical admission. A live, proven
idle warm actor may remain resident while the helper runs.
SQZ-128 is explicitly opted in on both A800 GPUs, including when the queue is
empty. Other suppliers remain disabled unless explicitly enabled. Helpers
inherit the site's approved temporary/cache paths, never its enrollment tokens.

```yaml
# Site YAML; disabled everywhere unless explicitly enabled.
idle_compute:
  enabled: true
  matrix_size: 4096
  minimum_free_vram_gb: 2
  poll_interval_seconds: 1
```

Helpers self-exit when an unknown CUDA process appears, and never signal other
users' processes. Active leases, non-idle actors, draining, external locks and
Topodrome reservations prevent idle load. Reservation changes reach helpers on
the next heartbeat (normally within five seconds). Scheduler permission expires
after at most 15 seconds; the child also requires a parent renewal every three
seconds and stops on parent pipe closure. Provider idle-reclamation rules remain
provider-specific; this feature cannot guarantee a lease is retained.

Suppliers sharing a filesystem must still use separate spool, lease database,
finalization journal, input cache, scratch, tunnel records and Ray state. Share
immutable model files and dependency packages, not mutable site authority.

### Operator: manual AutoDL capacity and release

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

Automatic refresh also unregisters disconnected `starting_failed` and `error`
instances, including registrations that never sent a heartbeat. The grace period
is `autodl.manual_offline_unregister_seconds` (default 600 seconds), measured
from the latest heartbeat, registration, or failure as applicable. Reconnected
agents, active leases, and nonterminal kernels prevent removal. Audit history
is retained; unregistering does not confirm power-off or stopped billing.

Idle shutdown remains automatic after five minutes, with required transfers and
active work blocking shutdown. There is no automatic renting, power-on, replacement,
pool limit, provider token, or `autodl on/off` command. Rent and register another
instance when more capacity is needed. GPU type/count comes from inspected inventory.

The Python clients expose `release_autodl(site_id, force=False)` (await for the
async client). The operator-only endpoint is
`POST /api/v1/providers/autodl/registrations/{site_id}/release` with `{"force": false}`.

### Run the example

The example requests one GPU with at least 15 GiB free, loads the configured
Llama model, and commits `model-summary.json` as a required artifact:

```bash
cd /ssd/xiangkun/Topodrome/topodrome
topodrome submit examples/model_summary.yaml --wait
```

With `--wait`, attempt output is streamed as it is produced. Workload stdout
and stderr retain their byte order and are written to the command's stdout;
Topodrome lifecycle notices and the final job JSON are written to stderr. This
keeps `topodrome submit ... --wait > workload.log` machine-friendly. Retries
are followed by attempt number without replaying already printed bytes.

Pressing `Ctrl-C` only detaches the follower—the submitted job keeps running.
`topodrome logs JOB_ID` also returns currently available live output, including
remote jobs that have not uploaded final logs yet. It switches to verified
final logs after completion; it is a snapshot, not a continuous follower.

After terminal job state, the follower drains the independently verified final
attempt log for up to 10 seconds. A permanently lost site cannot block the CLI
forever: expiry prints a warning, returns the terminal job, and any output that
arrives later remains available through `topodrome logs <job-id>`. Adjust the
bounded drain and refresh cadence when needed:

```bash
topodrome submit examples/model_summary.yaml --wait \
  --log-grace 60 --poll-interval 1
```

Without `--wait`, submission returns immediately. Copy the `job_id` from its
JSON output and inspect or cancel it with:

```bash
topodrome status <job-id>
topodrome logs <job-id>
topodrome cancel <job-id>
```

Cancel is safe to repeat. A running attempt gets up to 30 seconds to exit
cooperatively, after which Topodrome force-stops only that attempt's owned
actor or gang. A terminal job remains final, and cancellation never retries it.
Input transfers are cancellable without a GPU lease. Failed/canceled jobs can
reach their terminal state before the remote log upload completes; retain the
job ID and use `topodrome logs JOB_ID` again if final output is still arriving.

## 3. Write and submit a job

Need GPUs but no preloaded model? Use explicit `model: null` and declare positive
per-GPU peak `resources.min_vram_gb`. The full context remains available, with
`context.model/tokenizer = None` and actual assigned `torch.device` objects in
`context.device`/`context.devices`. Read [model-free GPU jobs](docs/MODEL_FREE_JOBS.md)
for single-process gangs, native distributed workers and runnable examples.

Give jobs optional human-readable metadata in the YAML:

```yaml
name: llama-lens-seed42
description: Train a tuned lens on Llama-8B; seed 42, baseline configuration.
```

`name` is at most 160 characters; `description` is at most 4,000 and may contain
multiple lines. Names need not be unique. `top` displays names alongside job IDs;
`status JOB_ID` includes both under `spec`. Existing unnamed jobs remain valid.
CLI overrides are `submit job.yaml --name "My run" --description "Baseline"`;
both Python clients accept the same keyword arguments in `submit(...)`.
Metadata is fixed at submission and does not change scheduling, actor reuse,
or retry behavior. Keep using the returned `job_id` for commands and assets.

### Job arrays

Arrays can sweep different registered models or revisions using per-element
`model` overrides, with independently pinned inputs, retries and outputs. Omitted
model inherits; explicit `model: null` means a model-free GPU job. Compatible idle
workers can switch checkpoint weights without a new process; bounded read-ahead
uses existing local files without extra disk copies. See
[checkpoint sweep examples and worker behavior](docs/JOB_ARRAYS.md#checkpoint-sweeps-and-different-models).

Use arrays for independent shards, seeds and parameter sweeps:

```bash
topodrome submit job.yaml --array 0-99 --max-parallel 4 --idempotency-key eval-v1
topodrome array ARRAY_ID
topodrome array-set ARRAY_ID --max-parallel 2
topodrome array-set ARRAY_ID --hold
topodrome array-set ARRAY_ID --resume
```

The entry point accepts `run(context, array_index, **kwargs)`. GPU resources apply
per element, not per array. Arrays have independent element retries, checkpoints,
logs and committed outputs; concurrency limits include starting allocations but
exclude detached GPU-free finalizers. Hold/lower-limit does not interrupt admitted
work. Read [Job arrays](docs/JOB_ARRAYS.md) for parameterized manifests, both SDKs,
idempotency, notifications, cancellation, and workbench navigation.

Job dependencies are intentionally not implemented. Use completion notifications
to let an agent submit authorized follow-up work; referenced outputs must already
be durably committed when the follow-up is submitted.

### Priority and estimated runtime

Provide a scheduling estimate as a separate top-level field, not text inside
`description`:

```yaml
name: quick-lens-eval
description: Evaluate the trained lens on held-out examples.
priority: 0
estimated_runtime_seconds: 600
```

The estimate is optional, finite, positive seconds of expected work per attempt
(including checkpoint/output I/O, excluding queue wait and model startup).
Use a measured similar run when available; omit it if unknown. CLI override:
`topodrome submit job.yaml --estimated-runtime-seconds 600`. Both Python clients
accept `estimated_runtime_seconds=600` in `submit(...)`.

Before aging promotion, eligible queued jobs and retries whose backoff elapsed are considered by higher
`priority` first, then shorter estimated runtime, then earlier submission time
and job ID. Unknown estimates follow estimated jobs at the same priority and
remain FIFO among themselves. Existing jobs without the field remain valid.
The `top` queue follows this ordering as well.

This is non-preemptive shortest-estimated-job-first ordering, not a time limit,
remaining-time prediction, or placement override. Running work is untouched;
jobs that cannot fit do not block other eligible work. Retry attempts retain
the submitted estimate; it is not reduced by checkpoint progress.

After `scheduling.starvation_seconds` of queued waiting (default 1,800 seconds),
aged jobs are considered FIFO ahead of newer jobs, including higher-priority
arrivals. This is a ranking guarantee, not a start-time SLA: affinity, foreign
processes, reservations, array holds/limits and missing capacity still apply.
An aged multi-GPU job with ready inputs also protects a prospective GPU group
from younger backfill while current Topodrome work finishes. It acquires no
partial lease and never preempts work. Other GPUs remain usable. Array holds
and retry backoff still prevent dispatch; retry waiting age restarts after its
backoff. Priority is per job, not a separate per-user scheduling rank.

A batch entry point exports `run(context, **kwargs)`. The context supplies the
already-loaded model, selected device, IDs, and an isolated workspace. Required
artifacts must be registered before success:

```python
import json


def run(context, *, prompt: str) -> dict[str, str]:
    result = {"prompt": prompt, "device": str(context.device)}
    output = context.workspace / "result.json"
    output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    context.register_artifact(output, logical_path="result.json")
    return result
```

Save that as `my_job/run.py` and add `my_job/job.yaml`:

The cross-site deployment model repository IDs are
`meta-llama/Meta-Llama-3-8B-Instruct`, `google/gemma-3-12b-it`,
`allenai/OLMo-2-1124-7B-Instruct`, and `Qwen/Qwen3-4B-Instruct-2507`.
Each supports an ordinary one-GPU request. Llama-70B is additionally enabled
for Lab1 only as of 2026-09-16. Use
`examples/model_summary_llama70b.yaml`; Jupyter and AutoDL remain unsupported.

Use `model: {repo_id: owner/repository, revision: registered-checkpoint}` for an
explicit registered checkpoint; optionally select a registered `subfolder`.
Omitting revision pins the registered default at submission. Unknown repositories
or snapshots are rejected; old aliases and `model.alias` are not accepted.
See [model addressing](docs/MODEL_ADDRESSES.md) for checkpoint labels and URLs.

```yaml
source_dir: .
entrypoint: run.py
model: meta-llama/Meta-Llama-3-8B-Instruct
resources:
  gpu: 1
  min_vram_gb: 15
kwargs:
  prompt: Explain atomic GPU leases.
required_outputs:
  - result.json
retry:
  max_attempts: 2
```

Submit it from any directory; `source_dir` is resolved relative to the manifest:

```bash
topodrome submit my_job/job.yaml --wait
```

Topodrome uploads the source content-addressably before queuing the job. Do not
put API tokens, passwords, or credentials in job environment variables or
keyword arguments; credential-bearing submissions are rejected.

Small immutable inputs can live in `source_dir`. Any artifact from a succeeded
producer job can instead be consumed directly as a global asset:

```yaml
assets:
  lens: job://job_e8b06bd3636647baa92640c3a92e0668/Llama-3-8B-Instruct-tuned-lens/
```

The entry point receives its local path as
`context.committed_inputs["lens"]`. The scheduler accepts the reference only
after the producer succeeds, authorizes access to that producer, and persists
an exact content pin. Files are materialized only on the assigned site; a
several-GiB member is therefore downloaded by a remote site only when a job
actually needs it. Placement recognizes only the exact immutable cache key for
that pinned view. Equal bytes exposed as a different file/directory view do not
create a false cache-locality preference.

For repeated use, publish a metadata-only immutable name:

```bash
topodrome asset-publish tuned-lens v1 \
  --job job_e8b06bd3636647baa92640c3a92e0668 \
  --path Llama-3-8B-Instruct-tuned-lens/
topodrome assets
```

Private names are referenced as `asset://private/tuned-lens@v1`. Pass
`--shared` during publication and use `asset://shared/tuned-lens@v1` when every
whitelisted user should be able to consume it. Publication copies no payload,
and the same name/version can never be rebound to different content.

Do not automatically publish every job or array element. Keep ordinary outputs
as transient artifacts; publish selected reusable inputs or experiment records
that should persist. Include the necessary schema/configuration, model and data
identity, seeds, provenance and usage notes. Prefer a curated aggregate over
hundreds of incidental child-job assets. Publication currently selects a tree
from one producer job, not a native multi-job collection.

### Tag and delete assets without a GPU

`topodrome assets` shows only the most recently published surviving version of
each asset name, separately for private and shared namespaces. Publication time,
not version-label ordering, defines latest (ties use reference ordering).
Use `topodrome assets --all-versions` for history. In `top`'s Assets page,
select an asset and press **Space** or click its disclosure arrow to expand
older versions beneath it. Each asset expands independently; **Space** on an
older version collapses its parent, and **Enter** opens that exact version's
details. History metadata is fetched on expansion (up to 1,000 versions across
assets, with an explicit limit notice); payload files are not downloaded.
`h` remains a Jobs-only history shortcut. Both Python clients accept
`assets(all_versions=True)`; the API accepts `all_versions=true`.
Latest selection precedes tag filters and the result limit, so an older tagged
version is not presented as latest. Use history to search older tagged versions.
No stored versions are removed, and exact references/downloads remain unchanged.

```bash
topodrome asset-tags tuned-lens v1 --add project:retrieval --add kind:lens
topodrome assets --tag project:retrieval --tag kind:lens
topodrome asset-tags tuned-lens v1 --remove kind:lens
topodrome asset-delete tuned-lens v1 --yes
```

Tags are mutable metadata outside immutable content manifests. Adding/removing
tags does not change content hashes, versions, pins, permissions or retention.
Repeated `--tag` filters match all tags, before the list limit is applied. Tags
are trimmed, lowercased, deduplicated and sorted: at most 32, each 1–64 ASCII
letters/digits or `_ . : / -`. Workbench asset rows and details display tags;
its existing text filter searches them too.

Deletion removes exactly one catalog reference, never a producer job. Without
`--yes`, the CLI asks for confirmation. Repeating a completed deletion is safe.
Deleted references disappear from listings and cannot serve new downloads or
submissions; accepted jobs retain their pinned files. Other asset references and
unexpired artifacts also protect payloads. Normal collection may later reclaim
unreferenced bytes, not necessarily immediately. A durable tombstone records
who deleted the version and when; it is not a backup. Deleted name/version pairs
cannot be reused—publish a new version instead. There is no undelete command.

All commands default to the caller's private namespace. Add `--shared` for
shared references; only their publisher or an operator may tag/delete them.
Both SDKs provide `update_asset_tags(name, version, add=(...), remove=(...),
visibility="private")`, `assets(tags=(...))`, and `delete_asset(name, version,
visibility="private")`. Await async calls. HTTP uses `PATCH /api/v1/assets/
{name}/{version}/tags` with `{"add": [...], "remove": [...]}`, repeated `tag`
query parameters on `GET /api/v1/assets`, and `DELETE /api/v1/assets/{name}/{version}`;
mutation routes accept the `visibility` query parameter.

### Download job outputs or assets without a GPU job

Download a committed job file/tree directly, or retrieve a named asset using
the same command:

```bash
topodrome download 'job://JOB_ID/results/' ./results
topodrome download 'job://JOB_ID/results/metrics.json' ./metrics.json
topodrome download 'asset://private/tuned-lens@v1' ./edited-lens
# Edit ./edited-lens locally, then publish a new version:
topodrome asset-upload tuned-lens v2 ./edited-lens
```

For a directory asset, the destination becomes a directory containing its
members; for a file asset, the destination is the exact output filename.
The destination must not already exist. Use `asset://shared/name@version` to
read a shared asset; private assets and job outputs require the appropriate
owner access. `JOB_ID` is the full ID, and the path must select registered,
durably committed outputs of a succeeded job. Both uploaded and job-published
assets work. No temporary asset registration is needed for a download.

The sync/async SDK provides `client.download(reference, destination)`, returning
a local `Path` (await the async version). Downloads stream multi-GiB bodies,
verify every member's size/SHA-256, and expose an editable copy only after
verification. Interrupted downloads keep a private
`.topodrome-download-<destination-key>` directory beside the destination;
rerun the same command to resume missing bytes, not restart the whole transfer.
Concurrent downloads into that staging tree are rejected. Corrupted data is
never accepted as complete. If content/server identity changes, choose a new
destination; do not overwrite or delete unrelated files to retry.

While downloading, a short-lived server reference protects the selected bytes
from GC, including across scheduler restarts. It renews during active transfer,
expires after 15 minutes without renewal, and has a 24-hour absolute lifetime.
Clients release it on completion or error. After a long interruption, the
ordinary artifact retention may expire; download before that deadline.
Downloading does not make server outputs permanent, create jobs, load models,
or acquire GPU leases. To retain results permanently on the server, publish an
asset explicitly. The older `asset-download NAME VERSION DESTINATION [--shared]`
and `download_asset` SDK method remain available, without unified byte-range
resume; prefer `download` for new workflows.

Upload an existing local file or directory directly:

```bash
topodrome asset-upload tuned-lens v2 ./edited-lens
topodrome asset tuned-lens v2
```

This creates `asset://private/tuned-lens@v2` without creating a job, loading a
model, or leasing a GPU. For edits, work on a separate local copy and upload
under a new version; never edit committed storage or a resolved input cache
in place. Existing versions and already-submitted jobs retain their old pins.
Use `--shared` only to intentionally share with all whitelisted users.

The synchronous and asynchronous SDKs both provide
`client.upload_asset(path, name="tuned-lens", version="v2")`. Use the returned
`.ref` in the same `assets` manifest field as any job-produced asset.

Uploads stream binary file bodies in bounded chunks, not base64 source JSON.
Several-GiB files are supported independently of source limits. Safety bounds
are 10,000 files, 256 GiB per file, 1 TiB total, and a 16-MiB metadata header;
the server keeps at least 1 GiB free and accepts two uploads concurrently.
Upload processing is bounded to six hours. Symlinks/special files, duplicate
or escaping paths, empty trees, and checksum mismatches are rejected. Empty
files are supported; empty directories and filesystem permissions are not
preserved. Only fully verified and fsynced content receives a catalog entry.

Keep the local files unchanged during hashing/upload. An interrupted upload
does not publish a partial version. Retry the same name/version with the same
content after an uncertain response: publication is idempotent, but network
transfer restarts from the beginning (byte-range resume is not implemented).
Different content under an existing version conflicts. If bytes already exist
as committed job outputs, prefer metadata-only `asset-publish` to avoid upload.

Prefer packages already present in the managed runtime. Extra packages use
`runtime.pip`, where every direct and transitive requirement needs an exact
version and real SHA-256 hash derived from project lock data. Do not invent
hashes or run an unpinned installer inside the entry point.

Submission does not automatically download committed artifacts. To retrieve
them without another job or asset registration, use
`topodrome download 'job://JOB_ID/committed-path' ./local-copy`.
If an automated caller needs a small
immediate result, print a non-sensitive summary for `topodrome logs` and also
commit the complete result as an artifact or reusable asset.

### Inspecting failed or canceled jobs

`topodrome logs JOB_ID` remains available after failure and cancellation. Python
exceptions include their traceback, and output is flushed even without a final
newline. If a worker is killed externally, Topodrome preserves output already
captured and records the failure reported by the runtime; SIGKILL itself cannot
generate a Python traceback. A job whose site disappears keeps its bounded
streamed log copy until a verified final log replaces it, rather than losing
those diagnostics after 24 hours.

Failure/cancellation reporting does not wait for a multi-GiB checkpoint upload:
checkpoint recovery is bounded to five seconds and stops waiting when canceled.
Only a verified checkpoint reference is eligible for automatic retry. An
unverified local checkpoint may survive for operator recovery but is not a
guaranteed resumable input.

### Optional Codex CLI notifications

On the machine and user account hosting the intended Codex CLI session:

```bash
topodrome submit job.yaml --notify-codex <session-uuid>
# Or attach to an already submitted job (does not submit again):
topodrome codex-notify <full-job-id> --thread <session-uuid>
topodrome codex-bridge --status
```

For one notification covering a fixed batch of 2–100 jobs, submit without
individual `--notify-codex` flags, then register the full set:

```bash
topodrome codex-notify JOB1 JOB2 JOB3 --thread <session-uuid>
```

The group sends once: when all members succeed, or on the first poll observing
a terminal failure or cancellation. Failures take precedence when a poll sees
both failure and cancellation. Retry-wait is not failure; unknown/unreachable
members cannot establish all-success, but do not hide an observed failure.
The message lists every member's observed state; remaining jobs continue and
are never canceled by the notification bridge. No later group messages follow
an early failure/cancellation alert. Re-registering the same set and destination
is idempotent, regardless of order. Separate individual subscriptions remain
independent and can still send their own messages. Use the returned `group_...`
ID with `codex-bridge --retry-uncertain` if manual delivery recovery is needed.

Requires a Codex CLI supporting `codex queue --thread`. Use an exact session
UUID, not a session name. Delivery uses only `codex queue --thread --message`:
it does not steer or interrupt an active turn. Opting in queues a message into that session and may
trigger Codex activity/token usage; it is not merely a desktop popup or
new authorization to fix failed jobs automatically. The message is a completion
handoff: resume the original task, take next steps already authorized by the
user, or report the result if none remain. Ask before actions outside that
scope. Messages explain whether a single job finished, all subscribed jobs
succeeded, or a batch alerted on failure/cancellation while others may continue.
Job metadata and logs remain data, not instructions.

A detached, per-user bridge polls every ten seconds and reports only final
`succeeded`, `failed`, or `canceled` states, not intermediate retry failures.
The association is local metadata, not part of the server job spec. Private
SQLite state under `$XDG_STATE_HOME/topodrome/codex-bridge` (default
`~/.local/state/topodrome/codex-bridge`) stores destinations and delivery state,
not credentials. Existing API ownership checks still apply. Keep that machine
awake and the same Topodrome profile configured; a remote scheduler cannot
directly notify a session on a sleeping laptop.

After a reboot, run `topodrome codex-bridge` in a persistent terminal or your
own process supervisor. Submission/attachment automatically starts the bridge;
only one bridge per state directory runs. No scheduler restart is required.
Delivered messages are not replayed. An interrupted/ambiguous delivery becomes
`uncertain`; inspect the target session before explicitly using
`topodrome codex-bridge --retry-uncertain <full-job-id>` because this can duplicate
a message. Ordinary status/network polling failures remain pending and retry.
To stop a subscription without canceling jobs, run
`topodrome codex-unsubscribe <job-id-or-group-id>` as the subscribing user on
the bridge machine. Find group IDs with `topodrome codex-bridge --status`.
This repeat-safe command affects only that subscription, not overlapping
groups or individual subscriptions. In-flight/already queued messages cannot
be recalled; the command warns when delivery may already have occurred.
The durable `unsubscribed` record prevents replay and re-registering that same
subscription is rejected rather than silently reactivating it.
If notification setup fails after submission, retain the returned job ID and
attach with `codex-notify`; never resubmit merely to repair notifications.

## 4. Submit an atomic two-GPU model

This project includes a deliberately forced two-GPU Llama exercise:

```bash
topodrome submit examples/model_summary_sharded.yaml --wait
```

The manifest requests two slots and explicitly selects
`transformers_device_map` sharding with a 10 GiB cap per GPU. Both GPUs are
leased, revalidated, started, and released as one atomic gang. A two-GPU job
never starts with only one GPU.

For distributed experiment code instead of one sharded model, use this exact
parallelism shape, with degree equal to `resources.gpu`:

```yaml
model:
  repo_id: meta-llama/Meta-Llama-3-8B-Instruct
  parallelism:
    mode: distributed
    degree: 2
    engine: ray_worker_group
resources:
  gpu: 2
```

The entry point runs on every rank with `RANK`, `WORLD_SIZE`, `MASTER_ADDR`,
and `MASTER_PORT` set. Only rank zero should register shared outputs. Multi-GPU
intent is always explicit; Topodrome never guesses it from model size. For a
sharded model, use `context.model` in place and never call `.to(...)` or
`.cuda()` on it because its device map is already active.

## 5. Submit with Python

Use the synchronous client when orchestration is part of another Python tool:

```python
from topodrome import Client


with Client() as client:
    job = client.submit(
        source_dir="my_job",
        entrypoint="run.py",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        resources={"gpu": 1, "min_vram_gb": 15},
        kwargs={"prompt": "Hello from the SDK"},
        required_outputs=["result.json"],
        idempotency_key="my-job-v1",
    )
    result = job.wait(timeout=900)
    print(result.state, job.job_id)
    print(job.logs())
```

Reuse the same idempotency key when retrying an uncertain client request. It
returns the same logical job instead of creating duplicate work.

## 6. Start an interactive model kernel

Request a one-GPU interactive Llama session:

```bash
topodrome kernel \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --gpus 1 \
  --min-vram-gb 15
```

Copy the returned `session_id`, then wait until it is running:

```bash
topodrome kernel-status <session-id>
```

For a forced two-GPU sharded session:

```bash
topodrome kernel \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --gpus 2 \
  --min-vram-gb 12 \
  --max-memory-per-gpu 10
```

## 7. Connect PyCharm or another Jupyter IDE

Once the kernel is running, start its managed stock Jupyter Server:

```bash
topodrome notebook-server <session-id>
```

The JSON response contains a URL and a separate token. If it also contains
`ide_connection`, run the printed `ssh_tunnel` command on the IDE workstation,
then configure the IDE with its `local_url` and the separately returned token.
If the returned public URL is directly reachable over the configured TLS route,
use that URL instead. Do not paste the token into logs or source files.
Each ordinary user's managed server exposes only that user's private notebook
subdirectory through the Contents API; the operator keeps the legacy global
notebook root.

Closing a notebook in the IDE only detaches that notebook; it does not destroy
the Topodrome kernel. Stop the managed server and kernel explicitly when done:

```bash
topodrome notebook-server <session-id> --stop
```

To stop only the managed server while retaining the kernel:

```bash
topodrome notebook-server <session-id> --stop --keep-kernel
```

To stop a kernel that has no managed server, use:

```bash
topodrome kernel-stop <session-id>
```

## 8. Optional site affinity

Ordinary jobs should remain portable and let the scheduler choose. For a
diagnostic that must run at one site, add both fields to its manifest:

```yaml
portability: site-affine
affinity_site_id: sqz-128
```

Valid deployed IDs are shown by `topodrome sites`. Avoid affinity in normal
workloads because it disables institutional fallback for that job.

## 9. Operator: reserve a GPU temporarily

This section uses the operator credential. Prefer the stable UUID shown by
`topodrome gpus`; a current site-local index is also accepted:

```bash
topodrome gpu-reserve lab1 \
  --gpu-index 7 \
  --duration-seconds 1800 \
  --reason 'exclusive debugging window' \
  --idempotency-key lab1-gpu7-debug-20260912
topodrome gpu-reservations --active-only
```

The default reservation blocks new placement on that physical GPU immediately. It
never interrupts an active job or kernel; while one is running the reservation
is `pending`, and Topodrome unloads any resulting idle warm replica only after
the lease ends. A replica spanning several GPUs is unloaded atomically. The
reservation overrides the last-actor floor that applies while work is queued,
so institutional warm capacity may reach zero. Topodrome does not relocate or
prewarm that actor. A new job tries every other clean
institutional gang before AutoDL; otherwise it stays queued under the current
no-foreign-contention policy.

To explicitly preempt active Topodrome work instead of waiting, add `--force`:

```bash
topodrome gpu-reserve lab1 --gpu-index 7 --duration-seconds 1800 \
  --reason 'urgent exclusive debugging' --force
```

This can lose uncheckpointed progress. The entire owned attempt or multi-GPU
gang gets its configured checkpoint grace (30 seconds by default), followed
by an exact-owned force stop if necessary. Batch jobs retry on an eligible GPU
using their last committed checkpoint, without consuming the normal failure
budget. Original affinity still applies; a site-affine job may have to queue.
Interactive sessions stop and are not migrated. Foreign processes are never
killed. The reservation remains pending until physical ownership is cleared.
Canceling/expiring a reservation prevents new stop commands but cannot undo an
already delivered preemption. Existing ordinary reservations are not converted;
cancel one and create an explicit forced reservation if escalation is intended.
The SDK and API expose the same opt-in `force=True` / `"force": true` field.

Cancel early or inspect the durable record with:

```bash
topodrome gpu-reservation <reservation-id>
topodrome gpu-reservation-cancel <reservation-id>
```

To add 30 minutes to an existing expiry without releasing the GPU:

```bash
topodrome gpu-reservation-extend <reservation-id> --seconds 1800
```

Only active reservations can be extended, at most one year into the future.
This preserves the exact reservation ID, GPU UUID, reason, state and force mode.
The CLI reads the current expiry and sends a compare-and-set extension. A
concurrent change causes a conflict instead of silently overwriting it. Each
fresh CLI invocation adds time; after an uncertain network response, inspect
the reservation before issuing another command. The SDK uses
`GPUReservationExtensionRequest(expected_expires_at=..., additional_seconds=...)`;
replaying that identical request is safe and does not add the duration again.

Natural expiry and cancellation make the GPU schedulable again and are safe to
repeat; neither action disrupts work already running.
# Removing disappeared manual AutoDL capacity (operator only)

Run `topodrome autodl unregister --site autodl-manual-<id>` after externally
stopping an instance. Wait for stale heartbeat and active lease recovery first.
This hides the registration and revokes enrollment, but does not shut down the
machine, stop billing, or delete job history, logs, checkpoints, or cached files.
Connected instances and active work are refused. Use `topodrome sites` to find
the site ID; `topodrome autodl register` can register the instance again later.
Previously running instances are also unregistered automatically after ten minutes
offline, once active leases/kernels have recovered. Operators can configure
`autodl.manual_offline_unregister_seconds` (default 600) in scheduler configuration.
This does not confirm that provider billing has stopped.

## Workload progress

Call `context.report_progress(current=120, total=1000, stage="training",
message="loss=0.42")` from an experiment. It emits an advisory structured log
record at most every two seconds; unknown totals may be omitted. Counts must be
finite/nonnegative, total positive, and current no greater than total. Stage and
message are printable strings limited to 64 and 256 characters. Only rank zero
should report distributed progress. Output failures return False, not a job failure.
Invalid arguments raise ValueError. Reports appear in live logs and `top`, including
an approximate ETA after sufficient samples; `~` means no update for 30 seconds.
The progress column puts numeric completion and ETA before stage/message text.
Verbose descriptions are clipped to the available terminal cells, and narrow
windows hide lower-priority job columns to keep progress visible. Widening the
window reveals more detail; `topodrome logs JOB_ID` retains the full reports.
Retries have independent progress, and reporting 100% does not commit outputs or
replace checkpointing. `top` abbreviates job IDs to four characters after `job_`;
these may collide, so use the full submission ID in all commands and asset references.

Updated site agents automatically report output finalization after the experiment
returns. For example, `upload 50% 1.0GiB/2.0GiB` shows aggregate artifact bytes,
followed by `verify`, `manifest`, `checkpoint`/`ckpt-upload`, and `logs` phases.
Upload percentage is not job completion: authoritative verification and the final
manifest are still required. Local/native storage reports verification rather than
network upload; byte-level upload statistics are available for HTTP mirror sites.
The final log stays unchanged while it is uploaded, so its `logs` phase has no
byte counter. No experiment changes are needed. Older, already-running agents
retain their behavior until their next safe upgrade.

Updated agents free batch GPUs after computation, before final output transfers.
`top` displays `finalizing` with zero GPUs while the released actor/GPU can serve
another job. The job API remains `running`, and `--wait`/completion notifications
still wait for verified outputs and logs—not merely completed computation.
Cancellation during finalization never interrupts a successor using that GPU.
Idle shutdown and graceful release still wait for all finalizers. An agent/site
restart during acknowledged finalization resumes the uploader from a private,
durable receipt, without rerunning computation. Network outages retry uploads
with backoff; expired upload authority pauses I/O until it is renewed. Canceled
or terminal work is never revived. A lost disk/site, recovery-grace expiry
(default 24 hours), or failure before acknowledged compute completion can still
require a computation retry, so resumable jobs remain recommended.

Inputs are now staged and verified before a GPU gang is leased. `top` shows
`downloading` with zero GPUs while the API job remains queued, then automatic
`loading`, `computing`, and finalization progress. Input errors show `input-error`
and retry with backoff without occupying GPUs. Job priority/resource checks still
apply; a job with ready inputs may run while another is downloading.

Before GPU allocation, a portable job's staging site may change if another
eligible site is expected to finish substantially sooner. For example, Lab1
becoming free can stop an unnecessary SQZ input transfer. This does not create
an attempt/retry or move a running job. Completed cache members and resumable
partials remain reusable; explicit site/region affinity still applies.

The scheduler learns cold model preparation and download throughput from real
site measurements, persisted across restarts. Warm reuse is not counted as a
cold start, and resumed bytes are not counted as newly transferred bytes.
Fresh transfer progress takes precedence over historical rates. Exact-workload
compute/finalization history also contributes when available; otherwise the
runtime hint and conservative defaults apply. These are estimates, not promised
ETAs, and history begins accumulating after this upgrade (old logs are not mined).
Defaults under scheduler `scheduling.input_placement` are:

```yaml
input_placement:
  enabled: true
  min_saving_seconds: 30
  min_saving_fraction: 0.2
  cooldown_seconds: 60
  history_max_age_seconds: 604800
  default_download_mib_per_second: 1
  local_read_mib_per_second: 512
  preparation_overhead_seconds: 5
```

Both savings thresholds must pass; a 60-second cooldown prevents repeated
bouncing. No eligible GPU remaining at the old site permits retargeting after
the cooldown without an artificial savings estimate. A full target staging
queue leaves the current transfer alone. These settings use the existing
validated, drain-before-swap policy reload; no active job is preempted.

Verified remote asset replicas may be removed after seven days without use or
when the configured cache budget (default 100 GiB) is exceeded. Active inputs
are pinned. This does not delete global assets, models, checkpoints, partial
downloads, or uncommitted outputs; a later job redownloads an evicted input.
Operators configure `asset_cache_max_age_seconds`, `asset_cache_max_bytes`, and
`asset_cache_gc_interval_seconds` in the site YAML and
`finalization_recovery_grace_seconds` in the scheduler YAML. Roll out configuration
and schema revision `0017_adaptive_inputs` during an idle, guarded service upgrade.

Operators can bound I/O using site settings `finalization_concurrency: 2` and
`finalization_backlog_limit: 4`. The latter pauses new admissions at the threshold;
already-running computations may finish and add their outputs to the backlog.

## Authoritative-storage retention

In `blobs-v1` deployments, payloads are immutable whole-file SHA-256 blobs;
manifests map logical output names to those bytes. Identical files share storage,
not permissions. Use the job/asset APIs, never edit internal blob paths. A fresh
archive/reset removes old jobs and asset names from active APIs without migrating
their references; offline recovery requires the operator. See
`docs/STORAGE_RETENTION.md` for layout, publication/GC safeguards and recovery.

Job artifacts are transient; named asset versions are persistent. The default
artifact lifetime is seven days after the job and attempt finish. Before then,
use `topodrome asset-publish NAME VERSION --job JOB_ID --path OUTPUT` to retain
a result. Publication pins the existing bytes without copying them. All named
versions remain protected until explicitly removed, including private assets.
Job YAML may override the temporary lifetime with `artifact_retention_seconds`
(3,600–31,536,000 seconds). Jobs accepted before this feature use the configured
server default. A bare `job://` reference is not permanent storage.

Queued/running consumers protect their resolved input files. Successful-job
recovery checkpoints become eligible after a one-hour grace. For failed or
canceled jobs, the latest checkpoint plus recorded recovery references remain
for seven days; older unreferenced snapshots may expire after the grace.
Checkpoints are retry state: publish final trained weights and learning-curve
files as assets when they should persist. Live jobs, unfinished finalization,
live leases, and active restore references are never collected by this policy.

Operator configuration (scheduler YAML; restart during an idle upgrade):

```yaml
storage_retention:
  enabled: true
  interval_seconds: 600
  artifact_seconds: 604800
  failed_checkpoint_seconds: 604800
  terminal_grace_seconds: 3600
  max_files_per_pass: 1000
  max_bytes_per_pass: 68719476736
```

The code default is disabled until an operator enables it. Inspect candidates
with `topodrome storage-gc`; explicitly run one bounded pass with
`topodrome storage-gc --apply`. Both commands require operator credentials.
The byte budget permits one oversized file so cleanup can make progress.
Deletion is permanent, with an exact intent/completion journal under
`GLOBAL_ROOT/.topodrome-retention`; journals do not contain payload backups.

Collection is serialized with input pinning and artifact publication using a
process lock and PostgreSQL advisory lock. It rechecks file identity before
unlinking, refuses symlinks, and aborts if the asset catalog is unreadable or
ambiguous. Expired inputs are rejected at submission/publication. Historical
job/artifact metadata remains available even after an unpinned payload expires.

This first-stage collector preserves logs, metadata, model files, source
bundles, unknown files, and direct asset uploads. It does not prune checkpoints
while their job is live. Remote replica eviction remains the existing separate
site-cache policy; remote pending uploads are not deleted by central retention.
Old-data migration, log compression/expiry, and orphan-upload/source GC are not
implemented. The blob collector additionally reclaims indexed unreferenced
payloads, including deleted uploaded assets, after the grace period.

### Bounded live checkpoints

The site writer keeps the newest two complete snapshots by default. Operators
may increase `checkpoint_keep_snapshots` in the site YAML (2–1,000); apply during
an idle agent upgrade. Snapshots selected for upload/verification are pinned
durably and may exceed this count. Failed saves never trigger rotation; pinned
files, other attempts' restore inputs, and unrelated files remain untouched.
Rotation uses a short non-blocking lock and adds no network calls to saving.
Checkpoint serialization, fsync and hashing remain synchronous as before.

`context.checkpoints` exposes only the newest two in-memory records to avoid
retaining every full optimizer/model payload. This is not a training-history
archive. Write metrics and desired final weights as artifacts and publish them
as assets. Remote replicas and pending outputs are still governed separately;
checkpoint rotation is not a claim that all remote storage is now bounded.
