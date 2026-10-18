# Failure diagnostics and rollout, 2026-09-14

The remote Jupyter Ray log recorded host-memory-pressure worker deaths.
Inspection of the live agent with `py-spy dump --nonblocking` showed postmortem
checkpoint uploads inside `_latest_checkpoint`, lasting about two hours. Those
uploads delayed terminal reports even though the corresponding GPU processes
no longer existed. Cancellation could not kill an already-missing worker.

Recovery stopped only two positively identified child transfer processes
uploading checkpoints for those dead attempts. No checkpoint files were deleted,
no Ray head or site agent was restarted, and no live experiment was signalled.
Both previously stuck canceled Jupyter jobs now report `canceled`; Gemma's
configured retry launched normally. Historical output is accessible through
`topodrome logs JOB_ID` (63,308 bytes for the old Llama-70B attempt).

## Implemented fixes

- Python output writes through without requiring a newline. Exceptions retain
  formatted tracebacks; SystemExit becomes a workload failure, not service exit.
- Driver-observed worker failure diagnostics are appended before recovery I/O.
- Checkpoint recovery waits at most five seconds and stops waiting immediately
  on explicit cancellation. Only verified references can become retry inputs;
  a background transfer or surviving local file is not proof of durability.
- Mirror transfers now use exact-path locks instead of hash stripes. A slow
  checkpoint cannot block an unrelated source, asset, or log. Same-path holders
  and waiters retain the same reentrant lock; unused locks are reclaimed.
- Scheduler-side streamed log copies remain while their job/attempt records
  exist, even after failure, until a verified final log supersedes them.
  Existing per-attempt live byte caps remain in force; orphan cleanup is unchanged.

## Activation status

**Current rollout: completed 2026-09-15, Asia/Singapore (2026-09-14 UTC).**
After confirming both institutional sites had no nonterminal jobs or kernels,
Lab1 and Jupyter agents were restarted. Both reported ready with fresh outbound
heartbeats and eight GPUs each at 16:13 UTC. The scheduler remained running.
The exact-path lock source matched across the local source and AutoDL package;
AutoDL's persistent next-start bootstrap is
`2b56f404ba3153283b864b88176b4165f2187287cfd5c2f2adaf2592a098cf00`.
Its maintenance boot was powered off through the journaled lifecycle. No new
GPU workload test was submitted; the hermetic suite passed 982 tests (seven
opt-in integration tests skipped).

Jupyter startup exposed a missing directory:
`/mnt/model/ckpts/Meta-Llama-3-70B-Instruct`. Its site-local 70B catalog entry
was temporarily removed so the other four available models could be served.
The previous catalog is preserved at
`/home/dpsk_a2a/.config/topodrome/models.yaml.before-missing-70b-20260915`.
At that point live capabilities omitted Jupyter 70B but still included Lab1
70B. The subsequent operator decision retired 70B across all sites: do not
restore its catalog entry. No model files were deleted or downloaded.

The notes below describe the earlier staged rollout, not remaining deployment
work.

The scheduler-only restart activated retention without restarting any site
agent. The four touched implementation modules passed mypy; lint passed; the
complete unit/component suite passed 952 tests. Regression cases include
tracebacks, SystemExit, abrupt process exit without a newline, a stalled
checkpoint, cancellation during recovery, and retained terminal-job output.

Lab1 has the new source on disk. The three worker-side modules were atomically
installed into the Jupyter release; previous copies are preserved under
`/home/dpsk_a2a/.local/share/topodrome/stages/failure-logs-20260914/previous`.
Jupyter was restarted while idle on 2026-09-14; at 12:46 UTC it reported ready,
a fresh heartbeat, and all eight GPUs. Startup first encountered a transient
control-tunnel failure, then a stale local two-GPU claim. After the exact old
owned Ray process group had stopped, recovery verified the group and endpoint
absent, acquired wrapper locks, and backed up the fence database to
`/home/dpsk_a2a/.local/share/topodrome/stages/failure-logs-20260914/fences-before-stale-recovery.sqlite`.
The normal post-runtime-stop reconciliation cleared stale claims/allocations
while preserving fencing counters. No workload files or logs were deleted.
The backup permits inspection/recovery; do not restore stale claims into a
running agent. This was an operational recovery, not a generalized automatic
repair for every missing-ownership-record case.

Lab1 still reported two running jobs, with both Ray workers present, and was
deliberately not restarted. Its agent-side activation remains pending an idle,
drain-safe restart. Already-imported Python code is not hot-patched; fresh
worker imports are not proof that existing workers changed code.

AutoDL's next-start bootstrap was rebuilt with these worker modules and the
existing native regional runtime seed. No AutoDL workload or institutional
model test was submitted for this change. Do not claim that worker-side
activation is complete across currently running institutional agents.

## Follow-up: source blocked by a checkpoint stripe

`job_12ffd869671e4819a047e683a30eafd3` reached running state but its source
materialization thread waited on a hash-stripe lock. The owning transfer was
an old checkpoint upload for failed attempt
`attempt_6da2a2c81a1145f78b986e447f88512f`; the Qwen actor was alive and idle.
Separating only log locks had left sources/assets vulnerable to the same bug.

Exact-path weakly held locks replace both stripe tables. Regression tests hold
a checkpoint upload open while downloading a colliding source/asset, check
same-path serialization and reentrancy, and verify unused lock reclamation.
The unit/component suite passed 956 tests; mypy and lint passed.

Operational recovery validated the failed attempt and the exact uploader PID,
parent, source, and remote temporary target, then used a pidfd to terminate
only that upload. Its local checkpoint remains intact. The original job had
already been canceled by its caller; it now reports canceled with readable
diagnostics. Replacement `job_fcf07cc2b3ce477e8c4a50373639cf56` was left alone
and produced workload logs.

The exact-path implementation was initially staged in Lab1's source and the
Jupyter release pending idle restarts. Those restarts and the AutoDL bootstrap
rebuild are now complete as recorded above. Do not hot-patch live lock tables,
because existing waiters would retain the old locks.
