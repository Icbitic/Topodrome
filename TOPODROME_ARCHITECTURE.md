# Topodrome Architecture

Status: implemented production candidate  
Last updated: 2026-09-28

This document is the architectural source of truth for Topodrome. It is written
for both maintainers and coding agents. Implementation choices may be refined,
but changes to the boundaries and invariants in this document should be made
explicitly rather than emerging accidentally during implementation.

## 1. Purpose

Storage amendment: deployments using `storage_layout: blobs-v1` store completed
artifact and asset payloads as immutable whole-file SHA-256 blobs. Schema-2
manifests preserve logical paths and authorization. Pending output claims precede
transfer; reference-aware, serialized collection protects accepted consumers and
published assets. Checkpoints retain their separate retry-state format. The
legacy attempt paths described below apply only to `attempt-v1`. A fresh-start
cutover archives old files and metadata, without migration, after a verified
database backup and an idle gate. See `topodrome/docs/STORAGE_RETENTION.md` for
the concrete layout, lock ordering, retention and rollback boundaries.

Topodrome is a general system for submitting batch experiments and interactive
Jupyter sessions to heterogeneous GPU resources spread across multiple sites.
The initial sites are:

- Lab1, with eight shared GPUs and persistent local storage;
- AutoDL, with GPU instances that may be stopped, released, and recreated;
- remote JupyterLab environments that may be reachable only through their
  Terminal API.

Topodrome provides one user-facing queue and one model-oriented interface while
keeping execution, model weights, and high-volume data local to each physical
site.

## 2. Architectural decisions

The following decisions are fixed for the first implementation:

1. Topodrome is implemented in Python.
2. One central scheduler coordinates all sites.
3. Scheduler state is stored in PostgreSQL on a persistent machine or volume
   with abundant disk space.
4. Every physical site runs an independent Ray cluster. Ray does not span the
   slow wide-area links between sites.
5. Every site runs a Topodrome agent outside Ray. The site agent is the local
   authority for GPU availability, leases, storage, and Ray lifecycle.
6. Remote sites initiate outbound control connections to the scheduler. Ray
   ports are not exposed to the public network.
7. Model weights are managed independently by each site's model service. They
   are not distributed through Lab1's global filesystem.
8. Lab1 provides the authoritative global experiment filesystem. Remote sites
   access it through the enrolled-agent HTTP API and a verified site-local
   mirror. Every file size, metadata query and directory listing uses HTTP;
   no external transfer executable or filesystem mount is required.
9. Every successfully committed job output is immediately addressable as an
   immutable global asset. A user may additionally publish a private or shared
   name and version; publication stores only a small manifest and never copies
   the payload. Users may also directly stream local assets to authoritative
   storage without a producer job or GPU. Remote sites materialize exact pinned
   members on first use.
10. Large temporary tensors and activation dumps remain on site-local scratch
    storage unless explicitly exported.
11. AutoDL capacity is rented/started externally and registered by SSH. Topodrome
    shuts it down after its idle period or an explicit operator release. The
    retired provider API, automatic rental, power-on, and replacement are removed.
12. Batch experiments and interactive Jupyter kernels are both first-class
    Topodrome workloads.
13. Institutional actor maintenance is queue-aware and centrally coordinated:
    while work is queued it preserves a model-agnostic one-GPU warm floor,
    permits only idle whole-replica replacement, and begins contention shedding
    only at five Topodrome-occupied GPUs. With no queued work, 20-minute idle
    expiry may reach zero actors. An explicit timed GPU reservation overrides
    the warm floor. Ordinary reservations never interrupt active work; an
    explicit operator forced reservation may preempt it as described in 5.1.
14. Whitelisted local users receive individual API credentials. Jobs and
    kernel sessions have durable owners; the operator may administer every
    resource, while ordinary users can inspect or mutate only their own direct
    job and kernel resources.
15. The public input abstraction is `assets`; there is no dataset-specific
    compatibility interface. Asset bodies may be several GiB, are never stored
    in PostgreSQL, and are transferred lazily through the configured storage
    adapter.

## 3. Major components

Operator upgrade amendment (2026-10-02): code/configuration releases use a
separate, single-writer updater and an immutable private release snapshot.
Preparation does not pause dispatch. After preparation, an optional target/all
admission pause drains only the requested scope; natural-idle mode leaves
dispatch open until a final atomic idle claim. Every live change requires fresh
generation-bound site inactivity plus no active leases/attempts/kernels, staging,
pending required output/log delivery or conflicting provider operations. The
claim shares the offer/kernel-admission lock, persists its fence before mutation,
and never preempts work. Submissions remain accepted. The fsynced owner-private
`.topodrome-upgrades` journal survives scheduler replacement; interrupted apply
is fenced for explicit recovery rather than blindly replayed. Verified release
or declared/verified rollback and fresh post-upgrade process identities precede
resuming dispatch. See `topodrome/docs/IDLE_UPGRADES.md`. Ordinary scheduling-policy
hot reload remains distinct and unchanged.

### 3.1 Central scheduler

The central scheduler is a long-running Python service with an HTTP API. FastAPI
is the default implementation choice.

It is responsible for:

- accepting batch jobs and interactive-kernel requests;
- maintaining the global queue;
- recording site, GPU, cache, model, job, attempt, and artifact state;
- selecting a physical site and GPU for each workload;
- issuing leases and fencing tokens;
- retrying failed or preempted attempts;
- registering external AutoDL instances and coordinating durable SSH shutdown;
- supervising opt-in stock Jupyter Server processes for IDE clients;
- exposing the Python SDK, CLI, and Jupyter integration;
- presenting status and logs without requiring users to contact a Ray cluster.

The scheduler may run on Lab1, but its PostgreSQL database is configured through
a DSN and may run on any durable host with sufficient disk. PostgreSQL stores
metadata, state transitions, leases, and verified log descriptors. Large tensors, model
weights, assets, and artifact bodies must not be stored as database blobs.

If PostgreSQL is unavailable, the scheduler stops issuing new leases. Running
site agents may finish current work and buffer status updates locally until the
database returns.

Production startup and every dispatch pass also require the database's recorded
Alembic revision to equal the single migration head expected by that build.
Being merely migration-compatible is insufficient: a missing, older, newer, or
branched revision keeps readiness unhealthy and dispatch paused until the
operator applies the exact release migration.

Schema `0019_remove_db_legacies` removes the unused SQL log-index table and
persisted kernel-URL column. Advisory live bytes remain filesystem-backed;
verified final-log descriptors remain on attempts, and kernel connect URLs are
minted on request. Job replay uses owner-scoped idempotency keys only. The
cleanup migration refuses unexpected log-index rows, stored kernel URLs, or
unscoped job keys instead of discarding historical data. Deploy it with the
matching scheduler build at an idle maintenance boundary; migration history and
provider audit records are retained.

### 3.2 Site agent

Institutional suppliers explicitly removed by an operator retain a `retired`
site tombstone rather than deleting their database row. Retirement requires
offline, quiescent capacity with no active attempts, leases, physical claims,
kernels or input staging, and revocation of the old enrollment/bootstrap
configuration. Retired sites and their GPU/model inventory are excluded from
active views and placement; a later heartbeat cannot revive a retired identity.
Historical attempts, outputs, logs and reference metadata remain intact. A
replacement physical server receives a fresh site identity and private mutable
state, while it may reuse the same immutable model files and locked dependencies.

Each site runs one asynchronous Python agent outside Ray. It maintains an
outbound authenticated connection to the scheduler and is responsible for:

- reporting a heartbeat and per-GPU NVML measurements;
- detecting Topodrome-managed and foreign GPU processes;
- reporting warm, cached, and obtainable model capabilities;
- verifying the HTTP storage authority and local cache;
- making the final local admission decision for a lease;
- reserving exact physical GPU UUIDs;
- starting or reconnecting to the site-local Ray cluster;
- creating, locating, and stopping model actors;
- preparing job workspaces and runtime environments;
- committing results and checkpoints;
- initiating graceful preemption under foreign-user pressure;
- reporting attempt state back to the scheduler.

The central scheduler makes a placement decision from the latest inventory, but
the site agent must recheck the physical GPU before accepting it. A stale
assignment is rejected and immediately returned to the central queue.

Read-only process diagnostics use the existing authenticated GPU heartbeat.
The agent reports PID, best-effort Unix user, process name, an optional Python
script basename/module and VRAM usage; it never exports command arguments or
inline code. The scheduler keeps only a bounded, process-local sample after
the corresponding inventory transaction commits. Status joins it by exact
site connection generation, GPU UUID and sample timestamp, and reports
unavailable/stale samples explicitly. Restart discards this advisory cache;
the next accepted heartbeat repopulates it. No new database history is created,
and these diagnostics never grant admission or authority to signal a process.

For a terminal-only JupyterLab site, the Terminal API is used to bootstrap or
recover the site agent. Once running, the agent communicates through its own
outbound connection and does not require inbound access from Lab1.
The static terminal command invokes Topodrome's packaged bootstrap wrapper. It
contains no credential; the wrapper reads owner-only site configuration and
site environment files, owns one detached process through a private lifetime
lock plus exact boot/PID/start-time/process-group/command/configuration identity,
and never signals a process whose full identity it cannot prove. Its local ready
marker is not scheduler readiness: dispatch still requires the exact outbound
connection generation and a fresh authenticated heartbeat.

### 3.3 Site-local Ray cluster

There is one Ray cluster per physical site. A site with several GPUs still has
one cluster, not one cluster per GPU.

Ray is the local execution plane. It provides:

- named model actors;
- local task and actor queues;
- CPU and GPU worker processes;
- temporary object transfer within a site;
- process supervision and bounded retries.

Ray's GPU resources alone are not considered evidence that a physical GPU is
free. Foreign processes are invisible to Ray's logical accounting. The site
agent therefore owns physical admission and creates actors only after acquiring
a Topodrome GPU lease.

Each physical GPU is represented by a Topodrome-specific logical slot so that
two Topodrome actors cannot claim it simultaneously. The actor is launched with
an explicit `CUDA_VISIBLE_DEVICES` assignment corresponding to the leased GPU.

Warm-allocation recovery is bound to the Ray cluster generation. If the site
lifecycle reconnects to the existing cluster, the agent reattaches only to the
exact persisted, clean actor identity. If the lifecycle proves that it started
a brand-new cluster, actors from the prior cluster cannot be reattached, so the
agent removes their local allocation rows by exact actor ID before admission
while retaining the monotonic GPU fence epochs. This bookkeeping cleanup sends
no signal; any unverified orphan process remains foreign NVML pressure.

A slot remains exactly one physical GPU; multi-GPU work does not change slot
granularity. The scheduler composes an ordered set of same-host slots into one
gang lease. PostgreSQL fencing claims, the site agent's durable local claims,
and the Ray reservation must each be all-or-none. A model-sharded replica uses
one Ray actor that requests every selected UUID resource and receives the
ordered devices in `CUDA_VISIBLE_DEVICES`. A distributed-training worker group
instead starts one private rank per selected slot only after its placement
group has reserved every member.

### 3.4 Model runtime

Checkpoint-sweep amendment (2026-10-18): a compatible idle Transformers worker may
retain its process and ordered GPU set across exact model revisions. Physical
admission first atomically claims the complete source replica for the new lease,
checking exact source actor/model/PIDs and reclaimable VRAM. Persisted residency
remains on the old identity in busy state until preparation succeeds. The worker
checks functional config, runtime profile and complete tensor layout before any
in-place write; local safetensors copies retain parameter storage and reset small
model-dependent buffers/caches. Configuration/layout/hook differences reconstruct
the model in that same compatible worker. Unsupported/private/distributed/adapter
or changed-resource workers retain ordinary whole-replica disposal. A partial
switch, changed ownership or unconfirmed cleanup never exposes the target model;
claims remain fenced until exact disposal. No active previous job/kernel is a
switch candidate. Exact-model warm reuse remains distinct from a revision switch
for startup measurement. Restart mismatches fail closed through exact actor
reconciliation rather than advertising the previous checkpoint against new bytes.

A model actor owns the live framework model, tokenizer, CPU memory, GPU memory,
and experiment hooks. An actor is keyed by at least:

- registered repository ID, exact snapshot revision, and optional subfolder;
- loader backend and backend version;
- dtype or quantization mode;
- adapter identity, when applicable;
- physical site and the exact ordered GPU UUID set;
- execution mode, sharding engine, degree, and per-device memory policy.

The actor is named and detached within its site-local Ray cluster. It therefore
appears as a reusable singleton to jobs at that site, while remaining an
implementation detail of the site service.

An ordinary actor owns one slot. An oversized-model replica may own a gang of
slots while still running the user entrypoint exactly once. Its backend must
verify that the loaded model's device map occupies every leased GPU and does not
offload to unleased GPUs, CPU RAM, or disk. It advertises one stable replica
identity and the same ordered member set on every member heartbeat. Reuse,
dirty marking, preemption, eviction, restart recovery, and lease release apply
to the complete replica; an individual shard is never reusable or releasable.
This is distinct from distributed training, whose entrypoint runs once per rank
and whose worker group is private and disposable.

The site advertises model states as:

- `warm`: an actor already holds the model in GPU memory;
- `cached`: the exact weights exist on persistent or local site storage;
- `obtainable`: the site knows how to acquire the weights but does not currently
  cache them.

Lab1 schedules against this capability information. It does not know or depend
on the site's model filesystem path.

Mechanistic-interpretability experiments execute inside the actor that owns the
model. The basic experiment contract is:

```python
def run(context, **kwargs):
    model = context.model
    tokenizer = context.tokenizer
    device = context.device
    # Experiment code, including hooks or activation modification.
```

Model-mutating calls are serialized by default. Hooks, temporary weight edits,
and activation patches must be removed in a `finally` block or by runtime-level
cleanup after every attempt. Adapter training receives exclusive ownership of
its actor for the duration of training, and base weights remain immutable.

Source imports are attempt-scoped as well: entrypoints, sibling helpers, and
source packages must not inherit another attempt's `sys.modules` state. The
source search path remains available through execution, async completion, and
cleanup, then the runtime restores shadowed installed modules and removes
attempt-owned import entries. Model/framework dependencies remain cached.
Local actors serialize these process-global changes; separate Ray worker
processes retain parallel execution. Experiment-created threads must finish
before the attempt returns; this is import hygiene, not a hostile-code sandbox.

### 3.5 Client interfaces

Topodrome exposes the same scheduler through:

- a typed Python SDK;
- a command-line interface;
- an HTTP API;
- a scoped Jupyter kernel gateway;
- an opt-in managed stock Jupyter Server surface for IDE compatibility.

Provider names, GPU indexes, model paths, storage paths, and credentials are
never required in ordinary experiment code.

## 4. Job specification

A batch submission contains a stable job description. A representative Python
interface is:

```python
topodrome.submit(
    entrypoint="experiments/label_heads.py",
    model={
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "parallelism": {
            "mode": "model_sharded",
            "degree": 2,
            "engine": "transformers_device_map",
            "strategy": "balanced",
            "max_memory_gb_per_gpu": 38,
        },
    },
    assets={
        "lens": "asset://shared/llama-tuned-lens@v1",
        "calibration": "job://job_a1b2c3/calibration/",
    },
    resources={"gpu": 2, "min_vram_gb": 38},
    kwargs={"limit": 128},
)
```

The persisted job record includes:

- a stable `job_id`;
- experiment source or source revision;
- entrypoint and arguments;
- model capability requirements;
- GPU count and minimum memory;
- immutable global asset inputs, pinned to exact content before dispatch;
- runtime environment requirements;
- checkpoint and retry policy;
- workload kind: batch, interactive, or adapter training;
- portability classification.

Custom runtime dependencies are immutable inputs. Each pip requirement is an
exact version pin with one or more allowed SHA-256 distribution hashes, and a
runtime working directory is referenced only by its scheduler-owned
`sha256:<digest>` source identity. The site resolves that identity before Ray
receives either a verified local directory or Ray's content-addressed package
URI; mutable HTTP or object-store working-directory URLs are rejected.

GPU batch jobs may explicitly use `model: null`. They retain the full experiment
context but do not resolve/acquire/load a catalog model. Positive per-GPU peak
`resources.min_vram_gb` is mandatory. `execution: single` (default) invokes the
entrypoint once with the complete atomically admitted same-site GPU set;
`execution: distributed` requires at least two GPUs and invokes one private
worker per GPU. Context exposes actual worker-local PyTorch `device`/`devices`,
`rank`, `world_size`, `local_rank` and `is_main_process`; each native rank sees
only its own `cuda:0`. Model/tokenizer are `None`, not a fake model alias.
Physical UUIDs establish CUDA visibility before worker imports. No transparent
VRAM pooling or framework-specific launcher is implied. These workers are
always private/disposable and are stopped before releasing their GPU ownership,
including detached finalization; they never contribute empty warm actors to the
retention floor. Agents advertise model-free support, and the scheduler fails
closed for older agents. Existing model identities and wire shapes are preserved.

For preloaded-model jobs, multi-GPU intent is never inferred from `resources.gpu`. The model requirement
must explicitly select `distributed`/`ray_worker_group` or
`model_sharded`/`transformers_device_map`, and its degree must equal the GPU
resource count. The model-sharded per-GPU maximum may not exceed the physical
per-GPU free-VRAM requirement.

Every execution receives a distinct `attempt_id`. Retrying a job does not
change its `job_id`.

Jobs may include optional display-only `name` (160 characters) and `description`
(4,000 characters). These persist in the existing job-spec JSON; old rows default
to no metadata. Names are non-unique, and commands/assets continue to use job IDs.
Metadata appears in owner-authorized status, never affects runtime fingerprints,
and is excluded from assignment wire payloads so agents need not restart.
Changing metadata with an already-used idempotency key conflicts like any other
submission change; older metadata-free records are normalized before comparison.

## 5. Scheduling

The scheduler first filters sites and GPUs by hard requirements:

- recent heartbeat;
- compatible model revision and runtime backend;
- sufficient physical free VRAM;
- no conflicting Topodrome lease;
- acceptable foreign-process pressure under explicit per-site operator policy;
- required GPU count;
- required storage and artifact accessibility;
- supported workload kind.

As of 2026-09-17, production sets
`scheduling.foreign_capacity_sharing_site_ids: []`: ordinary foreign CUDA
processes exclude a GPU from new placement regardless of free memory or low
utilization. Only positively recognized cooperative idle workers are exempt,
subject to the existing self-yielding handoff and physical memory checks.
Clean institutional capacity and compatible AutoDL remain eligible; otherwise
jobs stay queued. Already running work is not interrupted by this policy change.
The following sharing bands describe the optional opt-in policy, not a fallback
enabled in production.

Production retains `scheduling.prefer_autodl_over_foreign_processes`. Its
strict contention/cost bands are:

1. institutional gangs with no blocking foreign CUDA process;
2. compatible already-registered AutoDL capacity;
3. lightly contended institutional gangs, where every member has at least two
   complete requested VRAM footprints free and device utilization is at or
   below the configured light-contention threshold;
4. physically admissible institutional gangs under heavier foreign pressure.

The order is intentionally banded rather than a single warmth score. It fills
clean Lab1 and Jupyter capacity before sharing and prefers registered paid isolation to
any foreign process, including one at low utilization. False retains the
previous light-sharing-before-AutoDL order for older policy snapshots.
Changing only this priority toggle activates at the next serialized dispatch
boundary without draining running work. Physical free-VRAM admission
and safety margin remain mandatory in every band.

Blocking foreign CUDA processes make an individual GPU scheduler-ineligible by
default. An operator may opt in an exact site ID for foreign-capacity sharing
only when that site's independent admission policy also permits it. This
allow-list is empty by default and affects candidate filtering only: the site
agent remains the final physical authority and rechecks the complete gang. A
cooperative, self-yielding process is not reported as blocking foreign
pressure. Keeping the scheduler's fail-closed filter aligned with the stricter
site policy prevents rejection loops on one institutional site from masking an
otherwise eligible institutional or AutoDL placement.

Capacity sharing makes a GPU eligible; it does not make a crowded GPU equally
desirable. Both the utilization threshold and the two-footprint test must hold
for every selected gang member to enter the light-sharing band. Consequently,
a crowded warm Lab1 actor cannot pin a portable burst while clean or lightly
contended Jupyter capacity is eligible, but memory-safe foreign-sharing
capacity remains a last-resort institutional fallback. Instantaneous NVML
utilization is a conservative pressure signal, not a prediction of another
process's future load.

Actor replacement is driven by real queued demand, never by speculative
prewarming. Already-rented AutoDL capacity permits the same queue-driven idle
whole-replica replacement as institutional sites when actor policy is enabled.
This stays in the AutoDL placement band and never replaces active leases or
interactive kernels. After normal compatible/free clean institutional placement, the
scheduler may choose a clean institutional gang occupied by incompatible idle
actors before AutoDL. With paid preference enabled, replacements that share
with foreign processes follow AutoDL, light before heavy. The job's atomic
lease reserves every target GPU first. The
site then fences every complete source replica from `idle` to `stopping` in one
local transaction, proves exact backend identity, disposes only those owned
replicas, freshly re-probes the entire target gang, and creates the requested
model. Thus one idle Llama actor can become a Gemma actor, and two independent
one-GPU actors on one site can make room for one leased two-GPU actor, without
ever mutating a live model process in place. A changed, busy, incomplete, or
unprovable source rejects the complete offer; active batch work and interactive
kernels are never replacement candidates.

Targetless warm-actor release is a reloadable scheduler policy. Physical GPU
memberships are counted rather than actor objects: a two-GPU replica counts as
two. The occupied count is the union of resident actor GPUs and active lease
claims on Lab and Jupyter sites; AutoDL is excluded. Early shedding of an idle
warm replica that shares a GPU with blocking foreign CUDA work starts only
when the pre-action occupied count is at least the configured threshold (five
in production). The scheduler requires stable fresh heartbeats, chooses at
most one complete replica per cooldown window, and sends an exact,
generation-bound eviction intent. It never signals or terminates a foreign
process.

While runnable demand is queued, every ordinary targetless institutional
release also preserves a model-agnostic warm floor: after the action, at least
the configured number of resident actor GPU memberships (one in production)
remains across the institutional fabric. That actor need not retain its current
model, because a queued job may use the atomic replacement path above. The
floor is deliberately computed from reusable resident actors rather than
temporary active leases. If the invariant cannot be proved from fresh state,
no ordinary maintenance actor is released. With no queued demand the floor is
inactive, so ordinary idle expiry may naturally reduce residency to zero. An
explicit operator GPU reservation is the exception even while demand is
queued: it may reduce warm residency below the floor after the GPU becomes
idle.

The same central policy releases idle institutional actors after
`institutional_warm_actor_idle_timeout_seconds` (1,200 seconds in production),
while preserving the queue-active cross-site floor. The field is part of the
versioned scheduling snapshot, so changing it follows the drain-before-swap
hot-reload protocol and does not require a service restart. The legacy independent
site-local `warm_actor_idle_timeout_seconds` remains available for standalone
deployments, but is disabled on centrally managed production Lab1 and Jupyter
sites because it cannot enforce a cross-site floor. The durable idle interval
begins when the last exact workload lease releases the complete actor or
replica to `idle`, and every successful reuse resets it. Reconnecting an
existing Ray cluster does not reset that interval.

Per-site overrides use
`institutional_warm_actor_idle_timeout_seconds_by_site`. As of 2026-09-17,
`lab1` and `remote-jupyter-1` use 300 seconds; SQZ retains the 1,200-second
default. Omitted sites inherit the default; an explicit null disables timed
expiry for that site. Overrides are live-reloadable at the serialized dispatch
boundary: only complete idle replicas are candidates, with the same queue-active
floor, cooldown, and final site-local lease revalidation. Active jobs and kernels
are never evicted by a timeout update.

When the interval expires, the site agent uses a targetless, whole-replica
eviction fence. Under the same local offer serialization, it excludes every
replica referenced by an active process-local attempt or interactive session,
then atomically rechecks that every ordered member is still `idle`, past the
cutoff when one applies, and free of an active durable claim before changing
the complete group to `stopping` in one transaction. If admission wins that transaction, the
actor becomes `busy` and is not evicted; if expiration wins, a stale warm offer
is rejected and normal scheduling may reload the model later. The agent then
proves and disposes only the exact backend actor identity once for the whole
group. This path needs no alternative target. It never stops the site-local Ray head or agent, signals a
foreign or cooperative process, or preempts active work. An ambiguous disposal
retains the `stopping` fence; a proved-still-live failed disposal may restore
the group to `idle` and start a new full idle interval.

Warm-actor maintenance is serialized with liveness reconciliation, assignment-slot
replacement, and targetless eviction. After an agent crash, a complete
`stopping` group remains a disposal intent: an agent that rejoins the same Ray
cluster exact-reconnects and finishes that eviction, while ambiguous ownership
remains fenced. A complete `idle` group preserves its durable idle timestamp
across exact reconnect and is reaped promptly if already expired. Starting a
new, proved-empty Ray cluster continues to discard only the prior cluster's
allocation rows while retaining monotonic GPU fence epochs.

Within the first non-empty band, a gang whose every GPU matches the configured
fast-GPU patterns ranks first. An empty pattern list safely treats every
eligible institutional GPU as fast. The scheduler then minimizes estimated
ready time. A compatible warm actor on the exact selected UUIDs has zero model
startup; a cold actor uses the optional estimate on that site/model catalog
location, with conservative cached/obtainable defaults when it is not
configured. Reported site queue depth is converted to seconds with the
scheduler's bounded per-job queue-delay estimate and added to the cold start.
This comparison lets a short queue on fast local storage beat a long remote
cold start while spilling to the remote site when the estimated queue delay
crosses that startup difference. A warm capability on some other GPU proves a
local cache but never erases the selected GPU's cold-start cost. Exact warm
multi-GPU replicas participate in the same all-member contention
classification, so a heavy warm replica cannot hide a clean cold gang.

A local admission rejection proves that the persisted sample used for that
offer was stale. The scheduler durably quarantines only the exact rejected GPU
UUIDs from subsequent placement until that site's next authenticated heartbeat
receipt. This bounded quarantine is neither a lease nor an external
reservation; the new physical sample clears it automatically. Other GPUs and
sites remain dispatchable immediately, preventing a requeued portable job from
looping on one stale GPU.

When only foreign-sharing institutional bands are available and compatible
on-demand AutoDL is enabled, paid preference gives the provider an opportunity
before sharing (legacy policy does so only for heavy sharing).
The scheduler rechecks provider state after each configured wait window and
continues deferring the sharing fallback while a compatible startup is durably in
progress. Provider lifecycle deadlines bound failed starts. If provisioning is
unavailable or fails and no compatible operation remains pending, the queued
job immediately becomes eligible for the memory-safe sharing fallback on the
next dispatch tick. The scheduler does not create synthetic work merely to
keep paid capacity busy; running AutoDL capacity follows the configured
idle shutdown ceiling, with earlier drain only after a clean institutional gang
remains stable for the configured observation window.

While scanning past blocked work to find a later clean/light placement, the
scheduler retains the first provider-compatible demand in global queue order,
including its exact GPU shape and region. A later heavy-sharing job cannot
replace that demand merely because it has a different fallback reason. Durable
unresolved create/power-on intents defer heavy sharing only when their recorded
GPU count and canonical region prove compatibility with that exact demand.

Scheduling policy is a versioned immutable snapshot. An authenticated explicit
reload parses and validates the scheduler configuration and model catalog. The
only reloadable fields are the data-only `scheduling` settings and each model
location's `estimated_startup_seconds`; any credential, lease/fence timing,
endpoint, provider, model revision/backend/path, or other runtime change rejects
the entire candidate without exposing configuration values.

A live-placement exception applies when changes are limited to
`prefer_autodl_over_foreign_processes`, `preferred_site_ids`, model startup-cost estimates, and/or
removing sites from `foreign_capacity_sharing_site_ids`. Narrowing this
allowlist filters future offers only; it does not revoke existing leases.
Per-site idle-timeout overrides also use this live exception because disposal
still requires exact, idle, unleased replicas at the site.
Adding sharing sites or combining this with other settings still requires draining.
The next dispatch boundary durably activates that candidate without waiting for
existing attempts, leases, or provider operations. Their identities and state
remain unchanged; only subsequent placements use the new order. Institutional
actor-maintenance cooldowns remain intact. Any other changed field still requires draining.

Operators can set an ordered site preference live through
`PUT /api/v1/policy/site-preference`, the `site-preference` CLI command, or the
Sites detail actions in `top`. The request carries `expected_generation` so
concurrent edits conflict instead of silently overwriting each other. Only
registered, non-retired site IDs are accepted; offline preferences remain latent
until eligible. An empty order restores automatic measured-cost placement.
The order is persisted in the same atomic policy store and promoted while holding
the placement/offer lock, without waiting for active jobs or rewriting leases.
It is a rank within existing contention/cost bands, not affinity or an override
of health, reservations, foreign-process exclusions, gang admission, or model
compatibility. With an explicit order, ordinary and idle whole-replica replacement
candidates share their contention band so warmth cannot defeat the operator's
choice. A site without eligible capacity falls back immediately; all unlisted
sites retain measured-cost ordering. Queue priority, fairness and starvation
protection remain unchanged. Existing input-stage hysteresis remains in force;
the edit does not forcibly retarget transfers already underway.
Reloading YAML that omits `preferred_site_ids` preserves the live choice;
an explicit value, including `[]`, deliberately replaces it.

The workbench reorders directly in the existing Sites table, without a new tab
or editing dialog. Arrows select; Space picks up the selected site and starts
a captured draft. A visible `↕` marker and footer indicate that ordinary arrows
now move that exact site. Space drops it so arrows select another site.
Shift+arrows remain an optional direct shortcut, not a required terminal feature.
The site's details also offer Reorder sites to pick it up and return to the
table with mouse move/Auto/Save/Cancel controls. Enter saves once; Esc or leaving the
tab discards an unsaved draft. A row click drops the picked-up site and selects
the clicked row, never saving. Space and ordinary arrow overrides are scoped to
the Sites table and do not intercept text entry, buttons, details, or modal
controls. Starting a reorder clears the filter and captures all registered
sites and the policy generation; background refresh updates metrics but cannot
rewrite the draft. New sites remain unranked automatic fallbacks. Existing
explicit ranks precede unranked sites; a reorder ranks the complete captured
list. Offline sites remain visible and latent. Auto resets the draft order,
but neither it nor a local move performs an API write before explicit Save.
Unchanged drafts do not submit a mutation. A changed policy generation or
removed draft target disables Save; server-side CAS also protects races after
the last refresh. Conflicts require cancel/refresh/explicit retry, not blind
replay. Disruptive management actions retain separate review and confirmation.
This client-only change does not affect the placement bands or require a
scheduler/site-agent restart.

A valid changed candidate is first persisted, together with the still-active
snapshot, in an owner-only atomic state file under the scheduler global root.
It becomes `pending`; except for the priority-only change above, new batch and
interactive dispatch pauses, and queued/new
submissions remain queued. Existing offered, accepted, starting, running,
checkpointing, and committing attempt leases finish naturally. The same barrier
also waits for an in-flight or durably unresolved provider dispatch/startup;
provider and kernel reconciliation continues in the background. Reload never
cancels, preempts, moves, or rewrites a job, actor, lease, kernel, or provider
operation.

Policy staging/promotion and the placement-through-offer transaction share one
total order. An old-generation dispatch therefore either commits an offered
lease which the barrier counts, or finishes before the pending generation is
staged. When the blocker count reaches zero, the scheduler atomically persists
and activates the pending snapshot, clears generation-local dispatch hints, and
resumes dispatch under that generation. Identical retries are idempotent; a
different candidate conflicts while one is pending. A restart restores both
active and pending snapshots and resumes the drain instead of silently adopting
edited YAML. Operator status exposes active/pending generations, load
timestamps, drain state, and current blocker count. Arbitrary-code hot patching
is unsupported.

### 5.1 Timed operator GPU reservations

An authenticated operator may exclude one exact physical GPU for a bounded
interval. The request names a site plus either the current device index or the
stable NVML UUID; the scheduler resolves an index to the UUID before committing
the intent. The durable record retains the UUID, index snapshot, requester,
reason, creation/expiry times, state, and an optional idempotency key. It does
not reference the heartbeat GPU row because heartbeat reconciliation replaces
inventory rows and a disappeared device must not erase the audit record.

Reservation creation and offer creation share the scheduler's dispatch lock and
database ordering. A live `pending` or `ready` reservation overlays current
inventory as unavailable before another offer can select the UUID. `pending`
means an existing exact lease or actor still touches the device; `ready` means
Topodrome owns no allocation there. Expiry and cancellation are durable,
idempotent state transitions that remove the placement overlay. Index reuse or
a new device at the same index never silently retargets an existing record.

Operators may extend an unexpired reservation in place via
`POST /gpu-reservations/{id}/extend`. The request contains an aware
`expected_expires_at` and positive `additional_seconds`. Under the dispatch
lock and database row lock, compare the old expiry and update it atomically.
An identical already-applied target expiry is a harmless replay; any other
concurrent edit conflicts. Expired/canceled records cannot be revived, and the
new expiry cannot exceed one year from now. Identity, force mode, readiness and
eviction attribution stay unchanged; there is no temporary admission gap.

A default reservation is an admission fence, not preemption. Existing batch and
interactive leases finish normally. When every claim on a resident actor has
ended, the scheduler may send a deterministic, generation-bound eviction for
the exact site-published actor ID, replica-key hash, and ordered GPU membership.
The site again proves that the complete replica is idle and unclaimed before
atomically fencing and disposing it. A multi-GPU actor touching the reserved
UUID is released as one indivisible replica; its other members then become
available. Ambiguous or changed identity remains fenced or is rejected rather
than causing broad process cleanup.

Reservation cleanup overrides the queued-demand warm floor and therefore may
leave zero institutional actors. It does not relocate or speculatively prewarm
an actor. A later job still checks every other clean Lab1/Jupyter gang before
invoking AutoDL demand; memory-safe foreign-sharing institutional capacity
remains the fallback when the
provider is unavailable or its startup fails.
Reservation expiry before an active lease ends cancels the pending cleanup; it
never sends a late eviction for an expired intent. Ordinary users see only the
sanitized reserved bit in cluster inventory. Reservation creation, cancellation,
reason, requester, and audit history are operator-only.

Explicit `force: true` / `gpu-reserve --force` is the operator-approved
exception to active-work preservation. It persists an exact GPU reservation
and selects only attempts with active physical claims on that UUID. For a
multi-GPU replica this preempts the entire attempt/gang, never an individual
shard. The final send re-locks the reservation, rechecks its lifetime and exact
active claim, and binds delivery to the authenticated site connection generation.
Cancellation or expiry before delivery prevents a new preemption command;
already delivered preemption cannot be undone by canceling the reservation.

Batch preemption uses the existing checkpoint grace (30 seconds by default,
configurable per job), followed by the exact-owned bounded force-stop path.
The site never signals foreign processes or guesses process identity. A durable
attempt-to-reservation attribution allows a PREEMPTED batch attempt to retry
without consuming the normal failure budget, even if ordinary preemption retries
were disabled. Explicit user cancellation still wins. Existing committed
checkpoints and logs remain available; uncheckpointed progress may be lost.
Placement retains the original job's site/region affinity and other constraints,
so a retry may remain queued rather than run on another site. Interactive hosts
and their sessions are stopped instead of migrated. Existing reservations are
not upgraded implicitly, and normal idle reservations retain their old semantics.

After band, fast-class, and estimated-ready ordering, it scores the remaining
candidates, preferring:

1. an already warm compatible actor;
2. locally cached model weights;
3. locally cached immutable asset availability;
4. more stable VRAM headroom;
5. lower device utilization and queue depth;
6. lower expected transfer cost;
7. higher recent reliability.

The scheduler sends the chosen site an assignment containing the desired GPU
UUIDs and a short-lived fencing token. The site agent either atomically accepts
the lease or rejects it. Only an accepted lease can launch work.

Every successful lease renewal rotates that short-lived token without changing
the gang's ordered GPU claims or fencing epochs. The scheduler updates the
attempt and lease token digests atomically with the lease expiry, then returns
the fresh token and both expiries in the correlated ACK. The site agent adopts
the token only after authenticating that ACK and serializes renewal with
attempt, terminal, and kernel reports through one per-attempt fencing lock. It
delivers any earlier durable reports before renewal, so no queued old-token
event can trail a completed rotation. A lost ACK is retried with the same
durable message coordinates; the scheduler reconstructs the same grant without
extending the lease twice. Any other use of the prior token is stale
immediately. Long-running jobs therefore remain authorized through repeated
rotation rather than by lengthening the original token lifetime.

For a warm multi-GPU replica, placement accepts only the exact ordered member
set when every GPU reports the same live replica identity, runtime fingerprint,
and idle state. It never assembles a model from unrelated warm actors. For a
cold gang, each selected GPU is still scored and fenced independently, but all
claims share one lease and transaction. Queue age, explicit priority, and optional
`estimated_runtime_seconds` remain properties of the job rather than of a slot,
preventing a partially admitted gang from blocking capacity. Eligible queued jobs
and retries are scanned in descending priority, ascending estimated runtime
(unknown last), submission time, then job ID order. Pagination follows that same
key and skips unplaceable jobs without blocking subsequent candidates. Estimates
are positive finite expected work seconds per attempt, excluding queue wait and
model startup; retries retain the original estimate. They are scheduling hints,
not deadlines or preemption triggers. They remain in immutable job metadata and
are omitted from worker offers. After `scheduling.starvation_seconds` of eligible
waiting (default 1,800 seconds), aged work takes precedence over all younger work
in oldest-wait-first order, ignoring priority and runtime within that aged band.
Retry backoff is not eligible wait. Submission priority is per job, not per user.
The dispatch query, pagination cursor, and dashboard use the same ordering.

An aged job needing an unavailable gang may stage its inputs against a prospective
placement. Only Topodrome-owned occupancy is treated as potentially releasable;
foreign-process policy, reservations, health and model/affinity constraints still
apply. Once inputs are ready, the oldest such gang is excluded from younger
backfill until real capacity is available. Other GPUs remain usable. This is a
soft dispatch exclusion, never a partial lease or preemption, and cannot promise
resources that never become eligible.

### 5.2 Job arrays and observed timings

Array elements may override the complete `model` address and explicitly selected
resource fields. Omission inherits the shared model; explicit null means model-free
execution. Source/assets resolve once, while every child's registered model is
independently pinned before the all-or-none submission transaction. Resource and
parallelism checks apply to each child, including its own automatic model VRAM
default. These remain independent ordinary jobs, not one long sweep lease.
For an admitted array child, a bounded read-only queue lookahead may attach one
other eligible child's exact model as an advisory hint. The site reads only its
verified local weights through a site-wide bounded read-ahead pool, without CUDA,
downloads, permanent copies, queue reordering or readiness claims. Application
buffers, bytes, available-memory floor and a cooperative deadline bound this
optional optimization; the kernel controls reclaimable file-cache residency.

An array is a durable owner-scoped parent plus 1–1,000 independent ordinary jobs,
not a multi-site gang or a new execution protocol. A shared `JobSpec` and bounded
per-index kwargs expand at submission; immutable source/assets are resolved once
and pinned for every child in the same transaction. Metadata limits are 4 MiB for
the request and 32 MiB after expansion. Parent idempotency is owner-scoped and
checks a request fingerprint before re-resolving inputs. A changed replay conflicts.
The scheduler injects `array_index` into entry-point kwargs; users may not override
it. Child IDs, indices, checkpoints and retry limits remain independent/stable.

Admissions lock the parent before checking `max_parallel`, and hold that lock
through lease creation. Offered/accepted/active leases count; releasing leases
count while any physical claim remains active. Detached GPU-free finalizers do
not consume this allowance. Array holds and lowered ceilings block new admissions
only. Durable parent cancellation is reconciled through normal child cancellation
after scheduler restart, without rewriting successful siblings. All array reads
and controls preserve ordinary owner isolation. `POST/GET /arrays`, paginated
`GET /arrays/{id}`, `PATCH /arrays/{id}` and `POST /arrays/{id}/cancel` expose these
semantics. Dependencies are deliberately deferred; authorized agents use completion
notifications to decide follow-up submissions.

Schema `0018_job_arrays` also records non-overlapping scheduler-observed job phase
durations and each attempt's compute-completion timestamp. Queue/array wait, hold,
input staging, assignment, startup, compute, checkpointing, finalization, backoff
and cancellation remain distinct across retries; terminal totals freeze. Existing
history without phase data is untracked, never guessed. Reports are elapsed
observations, not remaining-time estimates or exact GPU-profiler measurements.
Workbench job rows expose compact Wait / Run / I/O, with full phase and per-attempt
breakdowns in details. Arrays expand/collapse inline in the Jobs table, retaining
other rows, stable child identities, and parent grouping through sorting/filtering. Completion
notifications observe aggregate durable success or the first terminal failure or
cancellation, never a retryable attempt failure; they do not cancel siblings.

## 6. Global experiment filesystem

Most experiments operate under an authoritative filesystem hosted by Lab1.
Lab1 accesses storage through its native local path. Remote AutoDL, Jupyter and
SQZ sites use one authenticated HTTP transport and a verified site-local mirror.
There is no protocol fallback or FUSE dependency. Persistent connections serve
tiny files immediately; identity-fenced Range streams serve large files.
Source trees use bounded standard ZIP wire envelopes when small, and paged
HTTP metadata plus independently verified member streams otherwise.
Missing immutable inputs are exposed only after local fsync, SHA-256 validation
and atomic publication. New outputs remain local until the durability barrier.

Every site maps its physical storage to a configured logical root, for example:

```text
<global-root>/
  sources/
  jobs/
  artifacts/
    catalog/
  notebooks/
```

The physical Lab1 root and each site's local namespace are configuration. In
direct-mirror mode, `global_root` is the site-local cache root; it is not a
second authoritative filesystem. Experiment code should resolve it through the
Topodrome context or environment and must not hardcode a machine-specific path.

HTTP mirror caches live on fast site-local disk and enforce a configured
minimum-free-space floor before materialization. Metadata, streams, prefix
writes and health checks have bounded memory, concurrency and timeouts.
Logical paths reject traversal, symlinks and special files. The agent reports
storage unavailable and refuses new work when Lab1 or its local cache is unsafe.
Whole-tree caches are separate from individual member caches: the presence of
one child never proves a complete directory has been materialized.

Storage separates read materialization (`resolve`) from canonical local write
locations (`resolve_writable`). Attempt workspaces, logs, checkpoints, staging
and manifest writers use the latter; it never downloads a remote job tree or
returns an immutable directory-cache view. The same canonical paths feed
`flush_and_verify`. Writable paths reject symlink aliases and special files;
retrying a job must not relocate its new attempt under a previous input cache.

Source-upload admission is an immutable member of the active scheduler policy
generation. Updated clients discover it before reading and base64-encoding a
tree, while the scheduler remains authoritative if a reload races an upload.
Production permits 256 MiB per file and 1 GiB decoded per bundle, within fixed
process-safety ceilings. A policy reload may raise or lower admission without a
service restart; it never revalidates or invalidates a content-addressed source
accepted under an earlier generation.

The global filesystem is appropriate for:

- experiment code;
- reusable immutable assets, including large final model-analysis products;
- job manifests;
- notebook documents;
- logs;
- compact checkpoints;
- reduced statistics, tables, and figures;
- final results.

It is not the execution path for:

- model weights;
- multi-gigabyte activation dumps;
- high-frequency training checkpoints;
- temporary tensor shards;
- Ray's object store.

Large temporary data is written to site-local scratch space. An experiment must
reduce, summarize, or explicitly export it before the attempt is considered
portable.

Closing a local cache file does not prove Lab1 durability. The mirror's
`flush_and_verify` first synchronizes and hashes local bytes. HTTP uploads
journal the exact source digest, destination and owned private prefix before
sending. Checksummed contiguous extents start at 128 KiB and adapt toward two
seconds of transfer, never exceeding 8 MiB. ACKs follow authority file and
directory fsync. After restart, the stored prefix and local source prefix must
match in size and independently observed SHA-256 before resume. Whole-file
verification precedes exact-sibling atomic publication; final independent
verification precedes receipt retirement and scheduler completion. Lost ACKs
recover idempotently without resending a verified complete output.
Immutable payloads reject conflicting replacement; current manifest pointers
retain the existing fenced-client semantics. These enrolled-agent byte APIs
are not user authorization or job-completion authority.

Authority-local digest requests return size and SHA-256 without downloading
outputs back to expensive compute. Bounded hash capacity rejects unsafe paths
and detected modifications; an unavailable verifier fails closed.
Bulk downloads require exact Range/Content-Range, Content-Length and If-Match
opaque source identity. Their complete bytes must match an accepted content pin
or independent authority digest before atomic cache publication. Unknown or
changed identities, corrupt bytes and ignored range requests cannot become
visible. Prefixes and pending results remain protected through recovery.

The default immediate threshold is 1 MiB (`storage.small_file_max_bytes`, hard
ceiling 4 MiB; zero uses HTTP streams for every file). Eight persistent
short-request connections and independent server I/O slots avoid process,
handshake and batching delays. Small source envelopes verify the original
manifest, exact regular member set and every checksum before publication;
they are not stored as another archive. HTTP keepalive defaults to 300 seconds.

Bulk HTTP streams/prefix writes have separate bounded pools and server slots.
For SSH-routed deployments, a second independently owned local forward is
mandatory (`bulk_ssh_local_forward`, `storage.bulk_url`): separate ports,
state directories and outer TCP connections, with ControlMaster reuse disabled.
Both routes reach the same authority. They are checked every five seconds;
a failed bulk route does not tear down healthy control. Physical link capacity
and disk contention remain shared; this does not promise a latency SLA.

Opt-in `storage.bulk_reconnect` measures sustained slow bulk traffic using a
second explicitly configured, pinned loopback SSH slot. A fresh route is
accepted only after bounded, integrity-checked bidirectional link probes show
an improvement (or verify recovery from an unavailable old link). Existing
HTTP streams drain on their original generation; durable upload extents and
verified download prefixes retain their normal resume semantics. At most two
owned generations exist, failed attempts observe cooldown, and control/jobs/
actors are untouched. Reconnection does not guarantee a different network
route. See `topodrome/docs/HTTP_TRANSPORT.md` for limits and configuration.

At most three recognized link-recovery attempts re-enter verified partial/
receipt recovery under one total deadline. Authentication, integrity, caller
cancellation, expired publication authority and disk failures never downgrade
or retry blindly. The mirror owns pins, exact-path locks, capacity accounting
and durability; HTTP owns byte transport. Native Lab1 access is unchanged.

HTTP upload extents may negotiate independent level-1 gzip bodies through
`Accept-Encoding` response headers. Only useful compression is selected; raw
SHA-256, offsets, acknowledgements and blobs remain unchanged. The authority
bounds encoded and decoded lengths, verifies a complete single gzip member and
its decoded checksum before writing, and rejects corruption without protocol
downgrade. Prefix metadata/publication has capacity separate from slow upload
body reception, preserving status/resume access under WAN loss. See
`topodrome/docs/HTTP_TRANSPORT.md` for limits and validation.
AutoDL's stdlib-only bootstrap seed helper uses the same HTTP read protocol and
protected enrollment before dependency installation. The bundle requires uv,
not external transfer commands. SSH remains for tunnels and scoped remote administration.

An authenticated GET returns exact bounded bytes and a SHA-256 ETag. Pinned inputs
check the expected size/hash before local fsync and atomic cache exposure, without
separate remote-stat round trips. PUT atomically receives, verifies and commits
one output, returning the authority-observed size/hash only after file/namespace
fsync. Blob publication folds its durable claim into this same request before
payload publication and is serialized with blob GC. Output authority is checked
before reception and again under job/attempt/lease locks at local commit; no
network reception holds those locks. Terminal log-outbox delivery remains allowed
for the exact originating site/attempt. Immutable files reject conflicting retries;
only the current job publication pointer may replace prior bytes. Symlinks, special
files, unsafe paths, unowned writes, expired leases and malformed uploads fail
closed. Only unsupported endpoints/ineligible objects downgrade to the bulk path,
never authentication, integrity or authority failures. A lost acknowledgement is
not success; a byte-identical retry is safe. No schema or storage-layout change is
required. Reverse proxies must preserve comparable upstream keepalive settings.

Checkpoint snapshots are self-contained: finalization
verifies only the newest snapshot, not every superseded snapshot. Retention
cleanup must preserve the newest verified recovery state and any checkpoint
still pinned by an active restore; it does not remove workload logs, metrics,
or final artifacts.

During live execution, the site writer rotates full checkpoint snapshots only
after the new snapshot has been atomically published, fsynced and hashed. It
retains the newest two by default (`checkpoint_keep_snapshots`, site setting,
minimum two), plus all snapshots selected for verification. Selection durably
pins the exact filename under a short attempt-local cross-process lock before
any network I/O; rotation takes that same lock non-blockingly and skips on
contention. Pins survive crashes and failed uploads. Rotation never traverses
other attempts, so scheduler-selected restore inputs remain intact. Temporary
files, unknown files, symlinks and multiply-linked files are not pruned.
Snapshot count is a best-effort bound: pins and I/O failures may exceed it.
The in-memory checkpoint-record history independently retains only two payloads.
No periodic upload is added; a local snapshot is not advertised as portable
until the existing verification barrier succeeds.

Authoritative terminal-job retention separately expires disposable checkpoints
and unpinned artifacts. Named asset versions, active consumer inputs, active
restore references, unfinished finalization and live leases are protected.
See `topodrome/docs/STORAGE_RETENTION.md` for defaults and audited rollout scope.

## 7. Global assets

Every artifact set from a successfully committed attempt is an immutable input
source. A consumer may reference an exact committed file or directory prefix
directly:

```yaml
assets:
  lens: job://job_a1b2c3/Llama-3-8B-Instruct-tuned-lens/
```

The scheduler authorizes the producer job, requires its successful published
commit manifest, cross-checks that manifest against durable artifact rows, and
replaces the submitted reference with a complete content pin. The pin contains
the file/directory kind, tree digest, total bytes, and each member's exact
attempt path, relative path, size, and SHA-256. A site agent rejects an unpinned
reference, so a mutable catalog name is never resolved during execution.

For reusable names, publication creates an immutable private or shared manifest:

```text
asset://private/lens@v1
asset://shared/lens@v1
```

Private names are scoped to the authenticated owner. Shared names are visible to
all whitelisted users. Publishing writes only catalog metadata beneath
`artifacts/catalog/v1`; artifact bytes remain in immutable shared blobs under
blobs-v1 (verified attempt paths under the legacy layout). Repeating an identical
publication is idempotent, while attempting to
reuse a name and version for different content is a conflict.

Asset deletion is an exact, authorized reference operation, not immediate payload
erasure. `DELETE /api/v1/assets/{name}/{version}` writes/fsyncs a `deleted.json`
tombstone before unlinking the active manifest. Readers hide tombstoned versions;
repeated deletion is idempotent and completes a crash-interrupted unlink. The
tombstone preserves audit metadata but does not pin bytes, and the deleted
name/version cannot be rebound. Shared mutations require the publisher or an
operator; private operations address the caller's namespace. Deletion shares
publication/submission/GC concurrency protection. Already accepted consumers,
other references and artifact retention continue to protect payloads until they
independently expire. No undelete or forced payload-purge operation is exposed.

Editable optional tags live in separate atomically replaced `tags.json` metadata.
Asset listings default to the latest nondeleted publication per visibility/name
in the caller's authorized namespace. Selection uses publication time with a
deterministic reference tie-break, before tag filtering and limiting. The API
and SDK `all_versions` option and CLI `--all-versions` expose prior versions
without changing storage or immutable input pins. The workbench groups history
under each latest asset row: Space/disclosure expands it independently, and
Enter opens the exact version. It fetches only metadata on expansion, bounds
history to 1,000 returned versions with explicit truncation/loading/error states,
and preserves exact-reference details and selection through refresh/collapse.
`PATCH /api/v1/assets/{name}/{version}/tags` applies set additions/removals under
the same authorization and lock. At most 32 normalized lowercase tags of 1–64
ASCII letters/digits or `_ . : / -` are allowed. Repeated `tag` filters on the
list API match all requested tags. Tag changes do not alter content/manifest
digests, versions, input pins, permissions or retention. CLI/SDK expose deletion,
tagging and filtering without GPU use; the workbench displays/searches tags.

Direct uploads use `asset-upload NAME VERSION LOCAL_PATH` or the synchronous/
asynchronous SDK's `upload_asset`. Authenticated owners stream a bounded JSON
file manifest (eight-byte big-endian length prefix) followed by concatenated
raw file bodies to `POST /api/v1/asset-uploads/{name}/{version}`. This transport
is separate from source-bundle JSON. It accepts at most 10,000 regular files,
256 GiB per file, 1 TiB total, and 16 MiB of metadata, with a 1-GiB storage floor,
two concurrent uploads, and a six-hour request ceiling. File hashes are checked
while streaming; no whole payload is buffered in memory. Symlinks, special
files, escaping/duplicate/conflicting paths, truncation, and trailing bytes are
rejected. Empty directories are not preserved.

Each upload stages under a unique private directory, fsyncs files/directories,
and atomically promotes its payload under `artifacts/uploads/<upload-id>/files`
before publishing a catalog manifest. Uploaded manifests carry `upload_id`
instead of producer job/attempt IDs. Catalog authorization and immutable
name/version semantics are identical for both origins. No model, job, attempt,
or lease is created by upload. An uncertain retry with identical content is
publication-idempotent, but transport does not yet resume partial bytes.
Known failed/redundant staging is cleaned; a process crash or ambiguous catalog
publication may leave an unreferenced payload for operator recovery, never a
catalog entry pointing at deliberately deleted bytes. Editing means publishing
a new version from a local copy, not mutating an existing pin.

`topodrome download REFERENCE DESTINATION` and both SDKs' `download` method
retrieve either a `job://JOB_ID/path` file/tree or an exact
`asset://private|shared/name@version` without creating a job, acquiring a GPU,
or registering an asset. Only durable succeeded-job outputs are exposed;
job ownership and private/shared catalog permissions are checked before
resolving exact immutable members. Public metadata never includes authority
filesystem or blob paths. Upload and job-published assets use the same engine.

`POST /api/v1/downloads` creates an owner-bound, fsynced reference beneath
`.topodrome-downloads`. Creation, renewal, closing and GC share the storage
serialization boundary; active download members survive artifact expiry and
asset deletion without a permanent asset reference. References expire after
15 minutes without renewal, with an absolute 24-hour lifetime. The client
renews every minute, including during local I/O; active server streams renew
too. Expired references no longer pin bytes, and a new download prunes expired
metadata. Admission is bounded to 16 live references per owner, 128 overall,
10,000 members and 16 MiB of metadata per reference. Corrupt pin metadata makes
collection fail closed. Clients close references on completion/error; process
death releases protection by expiry. Restarting the scheduler preserves live pins.

Authenticated `GET /api/v1/downloads/{download-id}/files/{member-index}` reads
descriptor-pinned regular files beneath the authority, refusing symlinks and
size changes. Open-ended byte ranges, SHA-256 ETags and If-Match permit resume
without accepting changed content. Responses are private/no-store. Client
copies stream through a destination-specific, owner-private staging tree,
locked against concurrent transfers and bound to the server/ref/manifest.
Retrying the same URI and destination rehashes the existing prefix, resumes
missing bytes and verifies every completed member's exact size/SHA-256 before
atomic exposure at a new destination. Existing files are never overwritten;
failed/canceled transfers retain partials for retry, while success removes
staging. Editing changes only the local copy. Downloads do not extend ordinary
retention beyond their live transfer pin; a permanent asset still requires
explicit publication. The older named `asset-download` interface remains
available but is not the resumable unified interface.

The site stages asset inputs only after placement and before model preparation.
Local authoritative files become a read-only, content-addressed view without a
second payload copy. A direct-mirror site downloads only missing members; each
large-file transfer has a configurable multi-hour timeout, reserves cache
capacity, and keeps a remote-identity-bound partial across retry or agent restart.
Completion verifies the exact pinned size and SHA-256. Later jobs reuse the
verified cache and advertise its full immutable cache key to placement. That key
binds the content digest to the pinned file/directory view, so equal content with
different view semantics cannot create a false locality hit. Locality remains a
bounded preference that cannot override GPU safety.

Asset payloads are not constrained by source-bundle upload limits and may be
several GiB. They remain a poor fit for high-churn activation dumps or frequent
checkpoint shards: those should stay in site-local scratch and be reduced before
explicit artifact commit. Model weights remain under the separate model-cache
policy in Section 8.

## 8. Model storage

Model addresses consist of a registered `repo_id` (`owner/repository`), an exact
`revision`, and optional `subfolder`. The catalogue's default or registered ref
labels are pinned before submission commits; explicit unknown selections are
rejected, never replaced by default weights. Additional operator-registered
snapshots support checkpoint experiments through ordinary jobs/arrays. Local
logical repositories need not be uploaded to HF. HF acquisition uses full commit
IDs, never moving branch labels. All actor/capability/placement identities include
the subfolder. Old aliases are replaced, not maintained as a compatibility layer.
Model discovery exposes registered ref labels and the default-snapshot marker
alongside per-site immutable capabilities, with exact repository filtering.
This metadata comes from the scheduler's catalogue, joined by repository, commit
and subfolder; discovery performs no Hub lookup and exposes no private locations.
See `topodrome/docs/MODEL_ADDRESSES.md` for catalogue and manifest examples.

The cross-site production catalog supports Meta-Llama-3-8B-Instruct,
Gemma-3-12B-it, OLMo-2-1124-7B-Instruct, and Qwen3-4B-Instruct-2507.
Llama-70B was retired on 2026-09-15 and re-enabled for Lab1 only on
2026-09-16 after an idle agent/scheduler restart. It requires a two-GPU
model-sharded gang with a 70-GiB per-GPU requirement. Jupyter and AutoDL remain
unsupported. Generic atomic sharding and distributed-job support are unchanged.

Model weights are deliberately separate from the global experiment filesystem.
Each site-specific model service owns acquisition, verification, caching,
loading, and eviction.

The scheduler records only the model capability and exact revision. A site may
use a Hugging Face cache, a local mirror, provider storage, or another source as
long as it satisfies the advertised identity.

The preference within a site is:

1. reuse a warm actor;
2. load from fast local SSD;
3. stage from site-persistent storage;
4. obtain the model from its configured origin.

The site agent verifies model identity before advertising `cached` or `warm`.
For a read-only preseeded checkpoint that lacks Hugging Face's internal
local-dir metadata, an operator may configure a separately protected revision
manifest. It must be an absolute regular file owned by root or the agent user
and not writable by group or other users. Before advertising the model, the
site hashes every required config, tokenizer, index, and weight file against
that trust anchor. This avoids reacquiring identical weights solely to recreate
metadata while preserving exact-revision verification.

## 9. AutoDL provider lifecycle

AutoDL instances are replaceable compute attached to persistent regional
storage.

### 9.0 Manual SSH registration

AutoDL uses only manual SSH registration. When `autodl.manual_ssh` is enabled, operator `topodrome autodl register`
accepts an SSH connection command and a hidden password. It inspects the
container architecture, libc, GPU inventory and shutdown support, then installs
the configured immutable uv bootstrap into an instance-specific directory.
Regional model caches and runtime seeds remain reusable. Readiness still requires
the authenticated agent heartbeat, GPU admission and storage checks, not SSH
reachability or successful launch alone.

This mode makes no provider API calls, provisions no instances and never powers
one back on automatically. The operator rents/starts it externally and registers
it again. Each registration has a distinct site identity, private SSH credentials,
TOFU-pinned host key and container ownership marker. Credentials live outside
global storage and never enter job payloads or provider journal rows.

Five-minute idle shutdown retains the full generation-bound drain, durability,
owned-runtime stop and replacement-connection exclusion protocol. Its journaled
power-off action uses AutoDL's `/usr/bin/shutdown` over pinned SSH. An ambiguous
command outcome remains fenced and requires operator attention; it is not blindly
replayed or presented as provider-confirmed power-off. A successful command means
the container accepted shutdown, not independent provider billing verification.
Stopped/released capacity is omitted from active inventory, not deleted from
audit history. Provider API renting, power-on and replacement are removed.

SSH identity and residual CUDA-process checks precede the journal's CALLING
transition. A busy preflight submits no shutdown and is retried at most three
times. STOPPING reconciliation resumes unsubmitted intents and commits already
acknowledged commands under the site registration lock. Ambiguous submitted
commands become an explicit fenced ERROR requiring operator attention, never an
indefinite STOPPING spinner or an unchecked replay.

Operator `topodrome autodl unregister --site <manual-site-id>` removes disconnected
manual capacity from active inventory and disables enrollment. It refuses fresh
heartbeats, connected agents, in-flight controller tasks, active leases or kernels.
It preserves audit/job/artifact records and performs no SSH shutdown or provider
operation: provider power/billing state remains unknown. Re-registration is allowed.
The manual-provider reconciliation loop also unregisters previously running instances
after `autodl.manual_offline_unregister_seconds` without a heartbeat (default 600).
The same disconnected, stale-heartbeat and no-active-work guards apply; automatic
cleanup never expires leases itself or signals remote processes. A returning instance
must be registered again. Unresolved startup/shutdown operations are not auto-removed.

### 9.1 Storage

AutoDL's file storage is mounted at `/root/autodl-fs` and is shared by instances
in the same AutoDL region. It is used for:

- regional model caches;
- provider-specific bootstrap state;
- large regional checkpoints that should survive instance release;
- data that would be expensive to transfer again within the region.

Active computation uses `/root/autodl-tmp` or another fast local disk. The site
agent stages model weights from `/root/autodl-fs` to local disk before loading
them when local I/O is materially faster.

The Lab1 global filesystem is accessed separately through the verified HTTP
mirror. Its cache belongs on `/root/autodl-tmp`, not on the limited
regional filesystem. Lab1 remains the authoritative location for normal
experiment code, immutable global assets, notebooks, portable checkpoints, and compact
final results.

### 9.2 Idle shutdown

An AutoDL site becomes idle only when all of the following are true:

- no job is running;
- no assignment is queued locally;
- no interactive kernel is active;
- no required artifact flush is pending;
- no model or asset staging operation is active.

Any activity in that list resets the idle timer. Production uses 300 seconds.
An operator can request earlier release. All shutdowns require generation-bound
drain and durability acknowledgements, followed by stopping owned Ray/agent
processes and a journaled SSH shutdown. Ambiguous commands are never blindly
replayed. A failed manual startup is surfaced for operator attention.

### 9.3 Explicit release and forced retry

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

Automatic refresh includes disconnected `starting_failed` and `error` instances,
even if no agent heartbeat ever arrived. The configurable offline grace (default
600 seconds) uses durable heartbeat/registration/failure timestamps and is
rechecked under lock before removal. A connected agent, active lease, or
nonterminal kernel prevents removal. This retires only enrollment and active
inventory; it retains audit history and never infers provider power or billing
state, nor replays an ambiguous shutdown.

Idle shutdown remains automatic after five minutes, with required transfers and
active work blocking shutdown. There is no automatic renting, power-on, replacement,
pool limit, provider token, or `autodl on/off` command. Rent and register another
instance when more capacity is needed. GPU type/count comes from inspected inventory.

The Python clients expose `release_autodl(site_id, force=False)` (await for the
async client). The operator-only endpoint is
`POST /api/v1/providers/autodl/registrations/{site_id}/release` with `{"force": false}`.

## 10. Jupyter sessions

An interactive Jupyter kernel is represented as a leased Topodrome workload.
A representative request is:

```python
topodrome.kernel(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    resources={"gpu": 1},
    mode="raw-model",
)
```

The scheduler selects a site, acquires a GPU lease, and returns a Lab1-facing
Jupyter URL. Remote sites create an outbound relay or reverse tunnel, so the
browser never connects directly to Ray or an inaccessible site port.

The native scheduler relay exposes a session-scoped Jupyter Gateway kernel
surface. An explicit connect request returns a short-lived bearer descriptor;
minting another descriptor rotates its access generation, and every REST
request and WebSocket frame revalidates that generation. The relay supports
Jupyter's classic framing and the v1 kernel WebSocket framing. It maps each
durable Topodrome kernel ID to a deterministic UUID-shaped gateway alias because
stock Jupyter Server registers native kernel channel routes only for that shape.
The alias is scoped by the enclosing session gateway and grants no authority by
itself.

For IDEs such as PyCharm that require a normal Jupyter Server URL, the scheduler
can opt in to supervising one installed stock Jupyter Server per live Topodrome
kernel. For an ordinary owner, this server owns notebook contents and Jupyter
sessions under a mode-0700 owner subdirectory of the configured notebooks root;
the operator retains the legacy root itself. It delegates only kernel operations
to the scoped scheduler gateway. Topodrome does not reimplement the stock
contents or sessions APIs and does not expose Ray. JupyterLab browser assets are
an optional deployment package, not a scheduler dependency.

The IDE-facing server has one stable public port and authentication token for
the live kernel session, except that granting/rotating or revoking and later
re-granting the owner's whitelist credential deliberately rotates that token.
The API and SDK return a query-free server URL, the separate token, and a
pasteable tokenized URL with no-store response headers. The public URL must be
routed through deployment TLS and access controls. The token is derived from a
stable scheduler-only secret plus a non-secret owner-access revision and is
never persisted, logged, or placed in process arguments.

The stock server authenticates to its upstream gateway with a separate HMAC
capability scoped to the Topodrome session and a random child-process
generation. Only the non-secret generation, exact PID/start identity, port, and
activity timestamps are persisted in a mode-0700 scheduler runtime directory.
The capability exists only in the child's minimal environment. A small
Jupyter-Server kernel-client adapter forwards the configured authorization
header on native WebSockets, matching the stock REST gateway behavior. The
scheduler accepts the capability only while that exact desired process
generation is live, the owner remains authorized, and the kernel/relay binding
is current. Stop, owner revocation, or replacement clears the generation before
terminating the exact process, so a delayed request from an old child cannot
reach the kernel. Owner revocation does not stop the underlying kernel or any
batch work; a later re-grant permits the owner to resume through newly issued
credentials.

Ordinary user connect-token expiry or rotation and transient upstream failure
must not restart a healthy IDE server. A graceful scheduler restart deliberately
leaves detached active servers running and recovers their same PID, port, and
user token from non-secret state. A dead or repeatedly unresponsive child is
replaced with a new bridge generation. Explicit managed-server deletion, direct
kernel stop, terminal kernel state, and disconnected idle expiry revoke the
bridge and terminate only the exact managed process; permanent service shutdown
may request controller-wide termination.

Deleting a stock Jupyter notebook session is a metadata detach only. It must
not delegate kernel deletion to the gateway, release the Topodrome lease, or
terminate the managed server; those remain explicit Topodrome lifecycle
operations.

Two modes are supported:

- `service`: a lightweight kernel submits structured operations to a shared
  model actor;
- `raw-model`: an exclusive kernel owns a GPU and a private model actor, allowing
  arbitrary hooks, inspection, activation modification, and adapter work.

A raw-model kernel cannot safely share mutable model state with batch jobs.
Notebook files live on the global filesystem. Live Python objects, CUDA state,
and uncommitted tensors are not migratable; after preemption, a replacement
kernel restores from the notebook and explicit checkpoints.

## 11. Artifacts, checkpoints, and retries

Each job has one stable identity and multiple possible attempts:

```text
jobs/<job-id>/
  manifest.json
  attempts/<attempt-id>/
  committed/
```

An attempt writes incomplete outputs with a temporary suffix or into an
attempt-specific staging directory. Required outputs are checksummed and
committed through a small manifest. Downstream jobs and retries consume only
committed outputs.

Retry behavior is transparent at the workspace level:

- the same `job_id` is retained;
- the new run receives a new `attempt_id`;
- committed outputs appear at the same logical paths;
- the script may detect completed shards and skip them;
- missing work is recomputed.

Portability is explicit:

- `portable`: all required state is available from the global filesystem;
- `site-affine`: required state exists only at one site;
- `region-affine`: required state exists in a provider's regional storage;
- `recomputable`: migration may discard uncommitted local work.

At approximately 1 Mbps between sites, the scheduler strongly prefers data and
artifact locality. It does not migrate large state merely to reduce queue time.

## 12. Foreign-user pressure and preemption

The site agent periodically inspects NVML process, utilization, and memory data.
For latched 99--100% legacy utilization (observed with driver bug 5446845),
AutoDL sites enable independent-counter fallback for every GPU model; other
sites retain the RTX 5090-only fallback. A site
may use independently measured GPM graphics/SM activity only when no compute or
graphics process is visible, allocated VRAM is at most 16 MiB (NVML v2 excludes
driver-reserved memory; unavailable v2 retains the conservative v1 value), and both metrics are
valid and at most 10% over two 200-ms windows. It rechecks memory/processes
after sampling and rounds measured activity upward. Unsupported/failed/invalid
GPM measurements retain the original conservative utilization. No GPU reset,
driver change, or blanket foreign-process exemption is performed. Heartbeats
and final local admission consume the same corrected inventory measurement.
It distinguishes Topodrome-owned processes from foreign processes using tracked
PIDs, actor identities, and leases.

The known Jupyter `gpu_idle_load.py` worker is an explicit exception: a direct
Python invocation whose script basename is exactly `gpu_idle_load.py` is
classified as cooperative and self-yielding, not as ordinary foreign pressure.
When Topodrome reserves that GPU, it starts an owned CUDA sentinel and waits for
the loader to notice the competing CUDA process and exit by itself. The agent
then re-probes the complete gang before loading a model. Topodrome never sends a
signal to, terminates, or kills the loader. If it does not yield within the
configured handoff timeout, admission fails and the process remains untouched;
similar names and all unmatched processes remain ordinary foreign work.

Transient activity does not immediately preempt work. Pressure must satisfy a
configurable sustained threshold. When policy requires yielding:

1. the site agent marks the attempt `PREEMPT_REQUESTED`;
2. the runtime checkpoints at the next supported boundary;
3. required files are committed;
4. the actor is stopped after a configurable grace period;
5. the GPU lease is released;
6. the scheduler requeues the same job with a new attempt.

If graceful checkpointing times out, the process may be terminated and the job
resumes from its last committed state. Topodrome never terminates foreign
processes.

Explicit job cancellation is separate from retryable pressure preemption. The
site first requests cooperative cancellation, waits the operator-owned
`cancellation_grace_seconds` (30 seconds in production, independent of the
job's checkpoint settings), and then force-stops only the exact tracked actor
or complete actor gang after re-proving process ownership. The attempt becomes
`canceled`, is not retried, and no foreign process is signalled.

Failure cleanup must not synchronously wait for multi-GiB checkpoint uploads.
Recovery waits at most five seconds for a verified checkpoint reference, and
explicit cancellation stops that wait immediately. Otherwise it retains the
previous committed reference; unverified checkpoint files remain local and
are never advertised as portable. Mirror operations serialize only for the exact
same logical path; unrelated sources, assets, logs, and checkpoints never share
a hash-stripe lock. Locks remain alive while held or awaited and are reclaimed
when unused.

## 13. Concurrency

Multiple clients may submit concurrently. The central queue, PostgreSQL leases,
and fencing tokens prevent duplicate ownership.

Inventory updates, offer creation, and reservation creation lock the site row
before physical GPU rows (gang rows in UUID order). This prevents heartbeat
replacement of GPU inventory from deadlocking with child-row foreign-key checks.
An offer proven unsent by the exact-generation control registry is withdrawn
transactionally and requeued immediately, retaining valid staged inputs and the
job's failure budget. A send exception is ambiguous: its claims remain fenced
until acceptance or normal expiry. Recovery never revokes an accepted attempt.

Within one model actor, mutating experiments run sequentially. Independent GPUs
may host separate actors for concurrent work. Multi-GPU training acquires all
required GPU leases atomically before creating its Ray worker group.

Model-sharded inference likewise acquires all required leases atomically, but
creates one group-level actor and invokes the experiment once. A warm sharded
replica occupies every member slot between jobs and returns to the queue only as
one reusable unit. Failure or foreign-user pressure on any member dirties and
disposes the entire replica before the gang lease is released.

Interactive raw-model sessions hold exclusive leases until shutdown or idle
expiry. Service-backed notebooks may share the ordinary actor queue.

## 14. Security

- AutoDL and other provider credentials live only in the scheduler's secret
  environment or an external secret manager.
- The scheduler's unauthenticated development mode is restricted to loopback
  listeners and loopback public origins. The built-in scheduler and managed-
  Jupyter HTTP listeners are plaintext and always restricted to loopback, in
  development and production. Production requires a bearer API token and HTTPS
  scheduler and managed-Jupyter public origins published through a TLS reverse
  proxy; invalid combinations fail during configuration loading.
- Credentials are never placed in source control, job manifests, PostgreSQL log
  fields, or the shared filesystem.
- The scheduler's operator bearer remains the administrative credential. Each
  whitelisted Unix account receives a separate revocable bearer whose digest,
  rather than plaintext, is kept in an owner-only local credential store.
- Every job and kernel session records its owner. Direct status, attempt, log,
  cancellation, kernel-connect, and managed-notebook operations return the same
  not-found response for a missing resource and another user's resource.
  Operators retain cross-owner access. Cluster-wide site, GPU, model, and
  provider-free inventory remain shared read-only views, while ordinary status
  snapshots include only the caller's job and kernel rows.
- Job idempotency is scoped to the authenticated owner. The scheduler hashes
  the owner and caller-provided key into the existing globally unique database
  field, avoiding a blocking uniqueness-constraint rewrite while allowing two
  users to choose the same client key independently.
- Site agents use individual enrollment identities and authenticated outbound
  connections.
- Remote storage HTTP uses site enrollment, never ordinary-user credentials.
  Plaintext endpoints must be explicit loopback routes inside pinned SSH
  forwards; non-loopback endpoints require TLS. Dedicated forward-capable
  site keys permit only the scheduler's loopback endpoint and are never
  personal login keys. Restrict them to the scheduler's exact forwarding endpoint.
  Credentials stay in owner-only files and request headers, never URLs or argv.
- Lease tokens are short-lived and scoped to one attempt.
- Direct kernel-relay URLs are short-lived and generation-revocable. Managed
  IDE server credentials remain stable only while their session/process and
  owner authorization are desired. Revoking a whitelist entry rejects further
  relay access and terminates only that owner's exact managed IDE child; it
  does not cancel the underlying kernel or batch work.
- Each ordinary user's managed stock Jupyter Server roots its Contents API in
  that owner's private notebook subdirectory. The operator retains the legacy
  global notebook root. This prevents accidental cross-user browsing through
  the IDE API, but trusted submitted code still shares the site's Unix account
  and is not a hostile-code filesystem boundary.
- Ray dashboards and internal control ports remain site-local.
- Experiment code is trusted research code in the first implementation. Strong
  multi-tenant compute and filesystem sandboxing is explicitly out of scope for
  version one. API ownership prevents accidental cross-user control; it is not
  a security boundary against malicious submitted code.
- Checkpoint references bind the job-scoped storage path to the SHA-256 digest
  verified at commit time. Resume re-verifies both authoritative storage and
  the bytes read before deserializing the trusted job's checkpoint. The agent
  stages and verifies bytes without unpickling, passing only a site-local path
  and pinned digest to the leased actor. `context.restore_checkpoint()` rehashes
  and deserializes inside that worker, caching the decoded value for that attempt.
  CUDA checkpoint tensors must never be restored in the coordinating agent.

Any provider token previously pasted into a chat or terminal log must be rotated
before it is configured in Topodrome.

### 14.1 Initial integration-test credentials

The initial remote JupyterLab test server has the following non-secret base
endpoint:

```text
http://103.237.28.246:7798
```

The Jupyter bearer token must be injected at runtime:

```text
TOPODROME_TEST_JUPYTER_TOKEN
```

For the initial Lab1 implementation, these settings are stored outside the
repository at:

```text
/home/xiangkun/.config/topodrome/secrets.env
```

The parent directory has mode `0700` and the file has mode `0600`. The file uses
dotenv syntax and currently defines:

```text
TOPODROME_TEST_JUPYTER_URL
TOPODROME_TEST_JUPYTER_TOKEN
```

The scheduler service loads this file through its service configuration or
settings loader. The Jupyter test credentials are restricted to integration tooling. AutoDL uses
private per-registration SSH credentials outside global storage; no developer API
token is loaded or used. Credentials must never enter jobs, logs, or model actors.

Production terminal bootstrap and managed IDE serving use distinct secret
names in that same protected file:

```text
TOPODROME_JUPYTER_BOOTSTRAP_URL
TOPODROME_JUPYTER_BOOTSTRAP_TOKEN
TOPODROME_MANAGED_JUPYTER_SERVER_TOKEN
```

The first two are available only to the terminal-bootstrap controller. The
managed-server value is stable key material used only to derive IDE and
process-generation bridge capabilities. None of these production values may
reuse the integration-test fields.

They must not be written into this document, committed configuration, example
commands, test snapshots, process arguments, database rows, or logs. Local
development loads them from the permission-restricted file above. Production
deployment should use a secret manager or systemd credentials. Tests must redact
request authorization headers and URLs containing a `token` query parameter.

Because the initial values were transmitted through conversation text, they
must be rotated before they are used for implementation testing.

## 15. Command-line surface

The System workbench tab consumes operator-only `GET /api/v1/system` diagnostics.
Storage telemetry is a single-flight, five-minute background sample with a
20-second / 250,000-entry cooperative scan budget and a 50,000-row-per-table
diagnostic metadata cap. It counts allocated regular-file bytes by unique inode
within the active central root, excluding symlinks/submounts. External databases,
remote caches/models and archived stores outside that root are out of scope.
Filesystem free space is explicitly host-wide, not Topodrome-only.
Cleanup estimates reuse the authoritative reference-aware eligibility planner
at the sample time and +24 hours, using a read-only blob-index snapshot. Viewing
does not initialize an index, hash payloads, acquire the publication/GC write
lock, or delete files. Shared hardlinks only count as reclaimable if all links
are eligible. Estimates assume unchanged references, do not override GC budgets
or disabled maintenance, and never authorize deletion; actual GC still rechecks
under its existing locks. Sampling failure retains labeled stale usage; unknown
eligibility is not zero. Runtime health is refreshed independently of disk scans.

Interactive terminals use an optional Textual workbench for `top`; `--once`
retains the plain snapshot and `--plain` the legacy monitor. The workbench is an
API client, never an alternate authority: confirmed actions use the existing
owner/operator-checked endpoints and exact full IDs/UUIDs. Background refresh
cannot retarget a confirmation. Reads and mutations are asynchronous; mutation
timeouts are uncertain outcomes, never automatic retry triggers. Offline site
metadata stays visible while obsolete GPU rows are hidden. The UI exposes no
Jupyter-server controls and does not change dispatch, leases, or site liveness.
The interactive surface is a human management console: job submission and asset
mutation stay in the agent-facing CLI/SDK. Its write actions are confirmed GPU
reservations, reservation extension/cancellation, manual AutoDL registration/release/removal,
live site preference, and policy reload. Page-local actions use the same SDK endpoints, with no direct
database or shell administration. The command palette is temporarily removed;
storage cleanup remains a CLI operation. Forms require a separate review and
confirmation step, and detail pages render curated fields, never raw API dumps.
AutoDL registration is a primary page-header action on Sites and AutoDL (`n`),
including when the AutoDL list is empty. It accepts an SSH command and masked
password, then requires explicit review and confirmation. Accepted registrations
appear as starting before their first agent heartbeat; provider startup state
and agent readiness remain distinct. There is no automatic rental or replay of
an uncertain registration request.
Job arrays and GPUs share native disclosure arrows and Space to expand/collapse
member jobs or read-only foreign processes in place. Space on a child collapses
its parent. Enter or clicking a data row opens details, never toggles expansion.
Array details show a live aggregate summary; attempts and logs belong to individual
members and remain in their own detail tabs. GPU Summary includes the same process
list as the expanded rows. Arrow keys continue switching tabs/selecting rows. Process refresh
preserves focus and GPU identity; recognized cooperative idle workers are labeled
separately. Device utilization is never attributed to an individual process.
Confirmation dialogs default to the affirmative button after review; initial
form inputs remain editable and forced eviction remains opt-in. Asynchronous
impact checks disable confirmation until complete, then focus Confirm only if
the user has not navigated elsewhere. Completion never applies an action itself.
Opening a resource's details displays its applicable actions directly. Background
list changes cannot retarget open details; missing rows disable their actions.
Reserved-GPU details show operator-only reservation metadata and direct Extend /
Cancel actions, joined by exact site and GPU UUID. Stale/failed metadata reads
disable these actions. Dialogs capture the reservation ID and expected expiry;
background replacement cannot retarget a confirmation. Ordinary users retain
only the sanitized reserved bit in shared inventory.
Workbench focus intent is bound to the visible resource and exact reservation,
not a table row number. Initial loading, confirmed mutations, and transient read
failures share this handling; subsequent user navigation cancels focus return.
Status/reservation reads begun before a completed mutation are discarded and refetched. Queued
action events from replaced controls cannot open forms for a new target. Empty
or removed views retain a visible keyboard target, without changing permissions
or bypassing confirmation.
On the Jobs list, explicitly toggling history with `h` focuses the resulting
rows (or the tab bar if still empty), so arrow navigation works immediately.
This does not redirect focus from open details, text fields, or modal dialogs.
Mouse tab selection (including the current tab), keyboard tab navigation, and
filter completion hand focus to the visible content; Down/Enter on a tab label
also enters its content without opening or executing a row action. Focus is
applied after layout with a navigation-bound intent, not an unconditional queued
callback. Superseded pane-focus events cannot reactivate an earlier tab. Empty
lists return focus to newly loaded rows only while their automatic fallback is
still current; deliberate navigation, typing, or a modal cancels that intent.
Returning from a modal rechecks that the previous focus is still visible and
usable. Text-only help/report dialogs start on their scrollable content, and
late report responses cannot open a dialog over subsequent navigation.
The log pane polls its bounded 64-KiB tail independently of summary/attempt
reads, at the configured refresh interval (two seconds by default). It displays
its own last successful fetch age and failure/staleness indicator; the global
LIVE badge describes inventory freshness, not log delivery or workload health.
Logs initially follow the tail using native scroll anchoring. Scrolling up freezes
the visible snapshot while one latest tail continues to be fetched; End or scrolling
back to the bottom resumes following. Refresh, tab changes, and resizing do not
steal focus. Late replies from a previous selection cannot replace current logs.
Complete, schema-valid progress records become one advisory last-reported meter
above the logs, never a job-completion signal. Invalid/partial records and ordinary
output remain visible; raw stored logs and the CLI log API are unchanged. Scrollbar
thumbs use foreground glyphs so the terminal-default background filter cannot
erase their interiors or their native mouse hit/drag behavior.
Forced-reservation review performs bounded, owner-scoped status/attempt reads to
identify jobs on the exact GPU UUID. This is advisory, not an atomic preemption
target lock; unavailable or stale information must be explicit. The server's
reservation fencing and permissions remain authoritative.

The intended command vocabulary is:

```bash
topodrome submit experiment.yaml --wait
topodrome status <job-id>
topodrome logs <job-id>
topodrome cancel <job-id>
topodrome kernel --model llama3.1-8b-it
topodrome notebook-server <kernel-session-id>
topodrome top --once
topodrome sites
topodrome gpus
topodrome models
topodrome autodl
```

The Python SDK and CLI call the same versioned scheduler API.
The authenticated `GET /api/v1/status` endpoint supplies the scheduler-wide
read-only snapshot used by `topodrome top`. The managed notebook-server
command returns an IDE-ready URL plus separate token and tokenized URL.

The dashboard snapshot filters terminal kernels before its row limit and orders
executing jobs, queued/retry-wait jobs, and terminal history in that order before
the job limit. GPU inventory, active kernels, and active AutoDL instances retain
space before the jobs table expands into remaining page rows. This is presentation only; durable history,
kernel lifecycle, and dispatch ordering remain unchanged.

`submit --wait` follows merged attempt stdout/stderr live on CLI stdout; state
notices and the terminal JSON stay on stderr. `Ctrl-C` detaches the follower
without canceling the job. The site agent publishes at most 64 KiB per fenced
live-log message with generation, offset, and digest coordinates. These
idempotent messages are outside the PostgreSQL event journal. The scheduler
stores at most 64 MiB of advisory bytes per attempt in private sidecars and
retains those bytes while their job/attempt records exist, including after
failure or cancellation without a final upload. Only orphan sidecars expire
after 24 hours. Reaching the cap explicitly stops that attempt's live
publisher instead of creating a retry loop; final authoritative log
publication and verification continue normally.

The final `attempt.log` remains authoritative. A terminal report carries its
logical path, exact length, and SHA-256 (or explicitly verifies that no log
exists); the scheduler independently verifies that descriptor before writing
durable per-attempt completion metadata and deleting advisory copies. The
ranged attempt-log API returns offsets, an anchor digest, replacement state,
and this completion bit so retry-aware clients can drain without duplicate
bytes. After terminal job state the CLI waits for completion only for its
bounded `--log-grace` interval (10 seconds by default); late authoritative
output remains readable with `topodrome logs`.

Python stdout/stderr capture writes through immediately, including output
without a newline. Ordinary exceptions and SystemExit retain their formatted
traceback; SystemExit is converted into an attempt failure instead of exiting
the hosting service. A driver-observed worker death is appended to the local
attempt log before recovery I/O. SIGKILL cannot produce a Python traceback or
recover bytes that never reached capture, but already written/streamed output
remains readable and is not discarded merely because the job failed.

`topodrome logs` and the SDK's `logs()` use the same live/verified-final source
selection as the ranged API. They tail database-authorized attempts in attempt
number order, including remote jobs whose final-log directory does not exist
on Lab1 yet. The combined tail is byte-bounded and returned with `no-store`;
unverified final uploads never supersede the live view.

## 16. Planned repository structure

Workload `context.report_progress` writes bounded validated structured records to
the existing fenced live-log channel, at most one per two seconds. The scheduler
extracts only complete records from accepted bytes into attempt-scoped advisory
sidecars, tolerating split/replayed chunks and discarding prior-generation progress.
Owner-filtered status includes only the latest active attempt's progress; no schema
migration or new agent message is needed. Stage/count/optional total/message and
approximate ETA never change job state or prove artifact durability. Missing updates
after 30 seconds are marked stale. Log-channel limits also limit progress delivery.
Dashboard job IDs show four characters after `job_`, for display only, not resolution.

### Compute completion and non-blocking finalization

For batch workloads, the agent releases the entire GPU gang as soon as user code
has returned and the runtime is confirmed idle. Disposable/private runtimes are
stopped first; reusable actors become warm and may immediately execute another
job. The agent retains immutable attempt paths and compact artifact registrations,
never a right to manipulate the released actor from the finalizer.

The ordered, fenced `committing` event carries `gpu_released: true`. PostgreSQL
deactivates that lease's GPU claims atomically and retains the lease in `releasing`
as renewable **finalization authority**. Its historical claim epochs authenticate
only attempt events/logs/renewals, not physical GPU ownership. Terminal reports
and expiry affect only that attempt's claims, never a successor using the GPU.
Every authenticated renewal updates both the in-memory token and the exact
acknowledged lease expiry, including renewals before GPU release. Detached
finalizers enforce that latest deadline, never the original offer deadline.
Unconfirmed renewals cannot extend it. A terminal renewal response cancels only
the detached transfer, not its formerly owned runtime.
The job API remains `running` until durable success/failure; `top` shows
`finalizing` and zero occupied GPUs. Legacy committing reports without this flag
retain their original GPU ownership semantics.

Finalizers upload artifacts, latest checkpoint and logs with the existing
authoritative verification barriers. Mirror uploads report bytes and distinct
verification phases. Finalizers remain in the site's active-attempt set and set
`artifact_flush_pending`, so drain, idle shutdown and instance cleanup cannot
discard pending outputs. Before detaching, the agent fsyncs registered output
snapshots and a private compute-completion receipt. Once the scheduler acknowledges
the detached `committing` transition, agent restart recovers **upload-only** authority
for that exact attempt. It does not reload a model or repeat user code. Transport
failures retry with bounded exponential backoff (1–30 seconds); completed remote
files are independently verified and reused rather than uploaded again.
Untyped local filesystem failures (including missing files and denied access)
are not network retries: they take the normal diagnostic failure path while
preserving completed output/recovery receipts.

Expired detached leases pause I/O and have a configurable recovery grace
(`finalization_recovery_grace_seconds`, default 24 hours). Recovery requires the
authenticated original site, the same still-running job/committing attempt, and
inactive historical GPU claims. Expired upload messages are acknowledged as no-ops
so they cannot block the ordered event spool; they never extend authority. Terminal
or canceled jobs cannot be revived. A permanently lost site/disk, expiry of the
recovery grace, or a crash before the scheduler's compute-completion acknowledgement
can still require normal job retry. Local uncommitted output is never cache-GCed.
GPU pressure/reservations do not preempt already-detached finalizers. User
cancellation targets transport only, never a subsequently reused actor.

Site configuration bounds concurrent finalization I/O with
`finalization_concurrency` (default 2) and stops new local admissions at
`finalization_backlog_limit` (default 4). Already-computing attempts may also finish,
so this admission threshold can be exceeded by the finite in-flight GPU workload.
An I/O outage cannot create an unbounded chain of newly admitted jobs.

### Input staging, cancellation, and cache lifetime

The scheduler first records a bounded input-stage intent (at most two outstanding
stages per site) with immutable asset/source/checkpoint identities. Jobs remain
queued, with no attempt or GPU claims, while the selected site downloads and verifies
inputs. Placement may be reconsidered before any lease exists, using remaining
input bytes, recent measured throughput, model startup, queue delay, and comparable
workload compute/finalization history. Existing contention/cost/fast-GPU bands,
affinity, priority and physical admission remain binding. A ready report
permits a fresh placement/physical admission check and an atomic gang lease; it is
not itself GPU authority. Stage generations/deadlines fence stale reports, expire
orphan transfers, and make replay safe. Another compatible job may bypass one whose
inputs are not ready. Source staging and checkpoint restoration do not upload
immutable inputs back to the authoritative store.

The default adaptive input policy requires both 30 seconds and 20% estimated
remaining-time savings to retarget, with a durable 60-second cooldown after a
switch. If the old site has no eligible GPU and another does, the savings test
is bypassed, but the cooldown remains. A full destination staging queue preserves
the old intent. Retargeting atomically changes the stage generation before sending
an exact-generation cancellation to the old site and a new stage command to the
destination. Late reports cannot train estimates or authorize an offer. A lost
cancellation is bounded by the old stage's deadline. Cached immutable bytes and
identity-bound partials survive; active attempts and kernels never participate.
Relative transfer TTLs avoid relying on synchronized host clocks.

Scheduler observations persist as clipped EWMAs in PostgreSQL. Cold runtime
preparation is measured with a site monotonic clock per site/model/parallelism;
warm reuse does not train the cold-start estimator. Direct-mirror downloads count
new bytes and transfer time, excluding resumed prefixes and verification time.
Fresh in-progress metrics override historical bandwidth. Samples older than seven
days are ignored; missing samples use catalog startup estimates and conservative
configured transfer rates. Native authoritative storage incurs local reads, not
a network download. Exact-workload compute and finalization timings can refine
completion estimates; without comparable history the submitted runtime hint is
used, or a common unknown-work cost. Unmeasured return I/O is not predicted from
an invented output size. History is advisory, not a completion/deadline guarantee.
These settings live under `scheduling.input_placement`; schema revision is
`0017_adaptive_inputs`. Only authenticated, non-replayed state transitions train
attempt costs. No user code, arguments or credentials are stored in measurement keys.

HTTP streams, transfer-lock waits, and streaming hash loops honor explicit
cancellation. Owned SSH tunnels are supervised independently of requests; foreign
PIDs are never signaled. Failed/canceled computation releases its GPU without waiting for remote
log storage: a separate private, fsynced log outbox retries upload and authoritative
registration. `log_complete` stays false until verified. Logs and finalizer receipts
count as pending I/O for provider idle shutdown and graceful release.

Only verified remote **asset replicas** are eligible for automatic eviction.
Active staging/attempt users pin both content views and shared physical member paths.
Authority verification occurs outside the short pin/delete lock; pins and local file
identity are rechecked before deletion. Local authoritative storage, models, source
trees, partial downloads, checkpoints and uncommitted job outputs are excluded.
Site settings default to `asset_cache_max_age_seconds: 604800`,
`asset_cache_max_bytes: 107374182400`, and `asset_cache_gc_interval_seconds: 600`.
These are best-effort bounds on verified asset entries, not a total disk quota:
active pins and files outside this cache may exceed them. An unavailable authority
defers eviction rather than risking the last recoverable copy.

Topodrome will be implemented as a new top-level directory:

```text
topodrome/
  README.md
  ARCHITECTURE.md
  AGENTS.md
  pyproject.toml
  configs/
  examples/
  src/topodrome/
    scheduler/
    site_agent/
    runtime/
    client/
    jupyter/
    storage/
    providers/
      autodl/
  tests/
    unit/
    integration/
```

The root-level `TOPODROME_ARCHITECTURE.md` remains the cross-project design
authority. The copy inside `topodrome/` may link back to it or replace it once
Topodrome becomes an independently versioned project.

The existing single-GPU Ray service remains untouched during initial
implementation. Its synchronous module-runner behavior may inform the generic
model runtime, but Topodrome must not depend on its project-specific name or
configuration.

## 17. Instructions for coding agents

Agents implementing Topodrome must preserve these invariants:

1. Do not create one Ray cluster spanning multiple physical sites.
2. Do not trust Ray's logical GPU count as proof that a shared physical GPU is
   available.
3. Do not expose Ray ports as the cross-site public API.
4. Do not put model weights in the Lab1 global experiment filesystem.
5. Do not store large binary artifacts in PostgreSQL.
6. Do not mark a remote-backed output durable until it has been flushed and
   verified.
7. Do not release an AutoDL instance until required local state is durable.
8. Never rent, power on, or replace AutoDL capacity automatically. Only explicitly
   SSH-registered capacity is schedulable; retain audit records after shutdown.
9. Do not run concurrent model-mutating experiments in the same actor.
10. Do not assume an interactive kernel's live process state can migrate.
11. Keep provider-specific behavior behind a provider interface.
12. Keep physical paths and credentials in configuration, not experiment code.
13. Preserve stable `job_id` values across retries and use a fresh `attempt_id`
    for every execution.
14. Preserve existing experiment directories and unrelated worktree changes.
15. Never evict, replace, migrate, or restart an actor that owns an active
    batch lease or interactive kernel session as ordinary maintenance. An explicit
    operator forced GPU reservation or forced AutoDL release is the maintenance exception: use the
    fenced checkpoint/owned-stop protocol in section 5.1, not direct PID killing.
    A detached finalization lease has no GPU/actor ownership and cannot block or
    authorize manipulation of an actor subsequently leased to a different job.
16. Count actor-policy capacity by unique physical GPU membership. While
    runnable demand is queued, preserve the configured reusable-actor floor
    after every ordinary targetless institutional release. With no queued
    demand ordinary idle expiry may reach zero; an explicit timed GPU
    reservation may override the floor even with queued demand. Active work
    requires the explicit forced-reservation exception in invariant 15.
17. Persist the authenticated owner of every job and kernel session. Ordinary
    user credentials must not reveal whether another user's direct resource ID
    exists, and must never authorize its logs, credentials, cancellation, or
    stop operations.
18. Resolve every user-facing asset reference at submission and persist its
    exact content pin. Never let a site agent resolve a mutable asset name.
19. Do not copy committed payload bytes merely to publish an asset name. Remote
    caches may materialize lazily, but must verify the scheduler-pinned size and
    SHA-256 before exposing a completed view.

Before changing one of these invariants, update this document and obtain an
explicit architecture decision.

## 18. Initial acceptance criteria

The first complete implementation is accepted when it can demonstrate:

- submission from Lab1 through both Python and CLI;
- PostgreSQL-backed job and lease recovery after scheduler restart;
- accurate discovery of free and foreign-occupied Lab1 GPUs;
- execution through a site-local Ray actor without repeated model loading;
- oversized inference through one warm, reusable, atomically leased multi-GPU
  replica, with verified use of every member and one entrypoint invocation;
- concurrent submissions queuing safely on one actor;
- queue-driven whole-replica model replacement, a one-GPU warm floor only
  while demand is queued, and 20-minute idle expiry to zero when it is not;
- a timed exact-GPU reservation that can safely evict the last idle actor,
  never evicts after cancellation/expiry, and drives later unplaceable demand
  through the ordinary AutoDL fallback;
- execution from a terminal-only Jupyter site through an outbound agent;
- use of the Lab1 global filesystem through a verified HTTP mirror;
- direct and named private/shared assets pinned before dispatch;
- multi-GiB asset members downloaded only on first remote use, resumed safely
  after interruption, and reused without a second payload copy;
- checkpointed retry with stable `job_id` and new `attempt_id`;
- graceful preemption and rescheduling;
- AutoDL idle power-off after its configured interval (five minutes in production);
- manual SSH registration and graceful/forced release without provider API calls;
- recovery of same-region AutoDL state through `/root/autodl-fs`;
- creation and connection of both service-backed and raw-model Jupyter kernels;
- IDE attachment through a stock Jupyter Server, including contents, sessions,
  and authenticated native WebSocket execution without restarting the server
  when an ordinary short-lived connect credential expires;
- per-owner job/kernel visibility, owner-scoped managed notebook roots, and
  whitelist revocation that removes API/relay/IDE access without cancelling
  running work;
- immediate revocation of an old managed-server bridge generation after stop or
  process replacement, with credentials absent from arguments and durable state;
- an authenticated scheduler-wide status snapshot usable by the CLI dashboard;
- complete removal of provider credentials from logs and persisted job records.
