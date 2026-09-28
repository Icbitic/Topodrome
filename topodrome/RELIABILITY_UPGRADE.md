# Transfer/lifecycle reliability upgrade

Requested 2026-09-26. Deployment is a separate, idle-guarded operation; running
jobs must not be interrupted. JupyterLab is excluded from deployment for now.

Implemented:

1. Owned, cancellable subprocess transfers and cancellation-aware locks/hash loops.
2. Durable finalization journal, capped retry backoff, authenticated restart recovery;
   a separate durable outbox preserves failed/canceled logs without holding GPUs.
3. Scheduler-controlled input staging without GPU claims; bounded staging admission,
   cancellation and fresh physical admission after inputs are ready.
4. Explicit lifecycle/transfer progress and cancellation-wait diagnostics in status/top.
5. Pinned, age/size-bounded eviction of verified remote asset replicas only.
6. Fault-injection/regression tests, documentation, and safe deployment.

Safety requirements: never mutate user job specifications, signal foreign processes,
   trust uploads without authoritative verification, delete uncommitted output,
   resume a terminal/canceled attempt, or allocate GPU authority for I/O-only work.
   Credentials must not enter journals under global storage, job payloads or logs.

Validation must cover cancellation during download/lock wait/hash/verification,
lost connections, long transfers across lease renewal, agent restart while uploading,
duplicate/replayed staging commands, shared asset users, cache eviction races,
and a successor running on a finalizer's former GPU.

## Validation and rollout — 2026-09-26

- `uv run pytest tests/unit tests/component -q --tb=short`: **1,059 passed**,
  two upstream deprecation warnings, 93.42 seconds.
- `uv run ruff check src tests migrations`: passed.
- Global Topodrome skill validation: passed; tutorial, architecture and skill
  explain input staging, detached finalization, late terminal logs and cache reuse.
- Migration `0016_input_staging`: clean-database upgrade and schema consistency
  check passed; production PostgreSQL migration applied successfully.
- Coordinated scheduler, Lab1 and SQZ deployment passed database idle checks
  before and after freezing dispatch. No running jobs were interrupted.
- Lab1 and SQZ returned to `ready`, with fresh heartbeats. JupyterLab was
  deliberately excluded and remains offline. No AutoDL instance was rented.
- Future manual AutoDL registrations use bootstrap
  `32df12284490423997340edde386adfc5d7e5e602ac069727b65c4d5d7cfb533`;
  the existing verified runtime seed was preserved unchanged.

Live checks used Qwen3-4B, one GPU per job, one attempt, no additional downloads:

| Job | Site | Expected result | Submission to terminal |
| --- | --- | --- | --- |
| `job_a2b358cdd372420a9a9da43461360019` | Lab1 | Succeeded; committed `result.json` and `payload.bin` | 15.0 s |
| `job_fa8f9f9819a445d8849191d212e5030b` | SQZ | Deliberately failed; before-death marker and exception in complete authoritative logs | 31.0 s |
| `job_753092f1862545d09243896176c5e312` | SQZ | Succeeded; consumed/validated Lab1 payload and committed `result.json` | 34.9 s |

The SQZ consumer was observed as `finalizing`, **zero GPUs**, while uploading
its manifest and logs. All three attempts ended with complete verified logs.
Final checks found no nonterminal jobs/kernels/leases, no pending finalization
or terminal-log receipts on Lab1/SQZ, and no scheduler/Lab1 warning entries
during the smoke checks. Foreign workloads and existing reservations were not
changed. Smoke artifacts remain available as deployment evidence.

## Operational boundaries

- Restart recovery and transport faults are covered by hermetic fault-injection
  tests; live smoke jobs did not deliberately disconnect/restart a busy agent.
- Upload-only recovery requires acknowledged compute completion and retained
  site-local output. Permanent disk/site loss, a crash before that acknowledgement,
  or expiry of the configurable 24-hour recovery grace can still require job retry.
- Cache GC covers verified remote asset replicas only (default seven days /
  100 GiB). It is not a quota for models, source, checkpoints, partial transfers,
  authoritative storage or uncommitted output; those bytes are not silently deleted.
- JupyterLab still needs this site-agent rollout when it becomes available;
  AutoDL bootstrap is updated but was not retested on paid hardware.

## Adaptive input placement — 2026-09-27

Deployed schema `0017_adaptive_inputs`, scheduler, Lab1 and SQZ after the idle
guard passed. Active jobs, foreign processes and reservations were untouched.

- Portable queued/retry-wait jobs may retarget input staging to a materially
  faster eligible site before any GPU lease. Default savings thresholds are
  30 seconds **and** 20%, with a persisted 60-second post-switch cooldown.
- Exact stage cancellation/generation fencing rejects old ready reports and
  preserves job identity/retry budget. Full target staging queues do not cancel
  the old transfer. Running attempts and kernels are not migrated.
- Persistent, clipped EWMA measurements now supply download and cold runtime
  preparation estimates, with fresh live transfer metrics taking precedence.
  Cold preparation includes actor/runtime setup, not just reading model weights.
  Comparable workload compute/finalization costs contribute when available.
  Seven-day-old history is ignored; missing data uses documented fallbacks.
- Relative stage TTLs tolerate host clock skew. Cached members and resumable
  prefixes are retained; resumed prefixes do not inflate measured bandwidth.
- `uv run pytest tests/unit tests/component -q --tb=short`: **1,073 passed**,
  two upstream deprecation warnings, 93.41 seconds. New tests cover mid-transfer
  Lab1 availability, stale reports, cooldown/thresholds, affinity, disabled
  policy, destination capacity, near-complete transfers, clock skew, resumed
  bytes, stale/outlier history, and replay/repeated-state measurement deduplication.
- Ruff, skill validation, clean migration and Alembic schema check passed.

Live Qwen3-4B checks both succeeded in one attempt with complete verified logs:

| Job | Site | Elapsed | Committed outputs |
| --- | --- | --- | --- |
| `job_0cd41b8f20834f9d90633837a8a153cb` | Lab1 | 15.004 s | `result.json`, `payload.bin` |
| `job_e8c550696ee44eb49a3eb56b36d04b65` | SQZ | 36.497 s | `result.json` |

SQZ validated Lab1's 8,126,464-byte input; it reported zero GPU ownership during
finalization. Read-back confirmed persisted startup/compute/finalization samples
on both sites and a download-rate sample on SQZ. These small smoke measurements
are collection-path evidence, not a general hardware/network benchmark.
The mid-transfer availability change was fault-simulated in scheduler tests,
not forced by reserving or interfering with production GPUs.

The AutoDL bootstrap was rebuilt as
`5af833ce08d69e5bc610ba662172500504b9d6cd788ff77656f13ff8e2bef747`,
preserving the existing runtime seed. No paid instance was rented. JupyterLab
remains excluded/offline. Tutorial, example scheduler configuration and the
user-level Topodrome skill document the new behavior.
