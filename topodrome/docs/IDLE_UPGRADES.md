# Idle-gated upgrades

The operator can queue an immutable release and let an independent updater apply
it when the affected services are idle. This is code/configuration deployment,
not scheduling-policy hot reload. It never preempts work or closes submissions.

```bash
topodrome upgrade queue /ssd/operator/releases/new-release/plan.yaml --pause targets
topodrome upgrade status
topodrome upgrade cancel upgrade_EXACT_ID
topodrome upgrade pause upgrade_EXACT_ID --scope targets
```

`--pause targets` is the default. Preparation happens while scheduling continues;
after preparation, new assignments and input staging avoid the named sites.
Other sites continue working. A scheduler upgrade affects every site, so its
target pause is cluster-wide. `--pause all` stops new dispatch cluster-wide;
`--pause none` waits for natural idleness without draining. In every mode, the
final idle check atomically fences the affected scope before any live change.
Jobs/arrays/kernel requests remain accepted and queued; running jobs, interactive
kernels, finalizers and pending logs finish normally. No timeout force-stops them.
Continuous demand may indefinitely delay a natural-idle update.
Before live mutation, `upgrade pause ID --scope none|targets|all` can change the
drain scope without changing the frozen release. `none` resumes dispatch while
waiting. The update uses an optimistic comparison to reject conflicting edits;
after apply starts, its safety fence cannot be relaxed.

## Release plan

Plans are protected, service-account-owned regular files (0600) on Lab1. Steps
are trusted **operator deployment programs**, never job-provided hooks. Declare
all affected site IDs and `scheduler: true` if central services, global protocol,
or database schema change. The updater cannot infer a script's side effects:
scripts must touch only the declared Topodrome installation/configuration and
prove exact process ownership. No foreign process, model, or user file cleanup.

```yaml
name: Enable measured bulk tunnel renewal on SQZ
scheduler: false
sites: [sqz]
files: [topodrome.whl, deploy.py]
prepare:
  - argv: [/absolute/path/to/python, '{release}/deploy.py', prepare]
    timeout_seconds: 300
apply:
  - argv: [/absolute/path/to/python, '{release}/deploy.py', apply]
    timeout_seconds: 300
verify:
  - argv: [/absolute/path/to/python, '{release}/deploy.py', verify]
    timeout_seconds: 60
rollback:
  - argv: [/absolute/path/to/python, '{release}/deploy.py', rollback]
verify_rollback:
  - argv: [/absolute/path/to/python, '{release}/deploy.py', verify-rollback]
require_new_agent_session: true
health_timeout_seconds: 180
```

Preparation may build a new uv environment and check connectivity, but must not
modify a running installation. Apply should switch a fully staged release and
restart only the idle service. Verify must check the intended installed build and
configuration, not just SSH reachability. Rollback is optional, but if supplied
it requires separate verification and a prior release compatible with the update
admission fence. Do not assume arbitrary database migrations are reversible.
When reusing locked dependencies through an isolated environment, preparation
must also validate required console entry points beside that environment's
interpreter (including Ray), for both the new and rollback installations.
Importable dependencies alone do not prove that a site agent can start.
The agent session ID is the durable event-spool stream ID and normally survives
a process restart. Do not erase or rotate an undelivered spool to satisfy this
check. For a restarting release that retains the spool, set
`require_new_agent_session: false` only when the frozen verification step independently
checks a changed exact-owned process identity, the intended interpreter/release,
and its installed build/configuration. Fresh post-claim heartbeats and all idle
checks remain mandatory. Non-restarting updates may also disable the stream-ID check.

Independent site releases should use separate single-site plans. A plan naming
several sites deliberately waits for all of them and holds them as one scope;
use that only when their changes truly need coordinated application. The runner
currently accepts one pending plan, so apply independent idle sites first, then
queue the busy site's draining release. Preserve holds during the handoff with
exact, non-forced operator reservations; remove only those temporary reservations
after the replacement upgrade's dispatch fence is active.

The scheduler snapshots the plan and hashes/copies its declared files to an
owner-private release under `<global-root>/.topodrome-upgrades/releases/`.
`{release}` expands to that snapshot. Subsequent source edits cannot change a
queued update. Each declared file is bounded to 1 GiB, plans to 128 KiB, and the
journal to 1 MiB/100 records; all declared files together must fit within 1 GiB
and leave a 1-GiB filesystem reserve. One pending release is allowed to avoid competing
maintenance fences. Repeating an explicit `--idempotency-key` with the same
inputs is safe; changed inputs conflict. Release files/audit are retained, never
silently garbage-collected. All local data/staging should live on `/ssd` here.

## Independent runner

Run the updater as the scheduler's Unix service account, independently of the
scheduler process. For this deployment:

```bash
topodrome-updater --config /home/xiangkun/.config/topodrome/scheduler.yaml \
  --env-file /home/xiangkun/.config/topodrome/scheduler.env
```

Install it as a separate systemd user service (example in
`configs/topodrome-updater.service.example`). The process holds an exclusive
ownership lock. It uses the operator bearer from protected settings, not a job
or user credential. Commands use argv without a shell and a minimal environment;
scheduler credentials, command output and arbitrary exception text are not
published in status. Deployment scripts load required secrets from protected
files and must not print them. They should be bounded and idempotent for explicit
operator recovery, but live apply is never automatically replayed after a crash.

## Safety and recovery

Before apply, the scheduler checks durable attempts, leases (including GPU-free
finalization), kernels, in-flight input staging, pending artifacts and provider
operations/tasks. Every target must also have a fresh heartbeat from its exact
current connection, healthy storage, and no local computation, assignment,
kernel, staging or output/log-delivery activity. Warm actors and SQZ2's owned
idle-compute helper are not running jobs. Offline/stale/unknown targets block,
never authorize a guessed shutdown. Ready, unused input caches are preserved;
their completed staging records do not cause a drain deadlock.
An explicitly scheduled idle service restart may dispose its idle warm actors;
this is planned code deployment, not discretionary warm-pool eviction. Active
work remains protected, and cached model files are not removed.

The idle check and final fence share the admission lock with offers and kernel
attachments. Provider idle shutdown and speculative recovery avoid fenced sites.
The fence and release state survive both scheduler and updater restarts.
Completion requires successful plan verification, healthy fresh post-claim
heartbeats, the requested stream-ID change or the frozen exact-process/build
proof described above, and a changed scheduler process identity for scheduler
releases. Only then is dispatch resumed.

An apply/verification failure attempts the explicitly declared rollback while
remaining fenced. An interrupted/ambiguous apply or rollback becomes
`needs_attention`: the updater **does not blindly repeat it or remove the fence**.
After inspection, explicitly request one of:

```bash
topodrome upgrade recover upgrade_EXACT_ID --action verify
topodrome upgrade recover upgrade_EXACT_ID --action rollback
```

Verification is safe to retry; rollback requires a declared rollback plan. If the
scheduler itself cannot start, restore its compatible release through the same
operator-owned deployment mechanism, then request verification. Do not delete
the journal to bypass an uncertain upgrade. Cancellation is allowed only before
live mutation; it cancels maintenance, not jobs. Status exposes sanitized phases,
affected scope, release digest and exact idle blockers, also in System telemetry.
