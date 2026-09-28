# Job arrays

Arrays run independent shards, seeds or parameter combinations. They do not split
arbitrary Python code or create a cross-site distributed-training group.

## Submit

```bash
topodrome submit job.yaml --array 0-99 --max-parallel 4 --idempotency-key eval-shards-v1
```

The entry point must accept `run(context, array_index, **kwargs)`. Use the index
to select a deterministic shard or seed. Every element is an ordinary job with
its own full job ID, attempts, checkpoint, logs and outputs; indices do not change
on retry. Elements may run on different eligible sites. Resource requirements
apply **per element**: four parallel two-GPU elements may occupy eight GPUs.

For different parameters per element, `topodrome submit-array sweep.yaml` accepts:

```yaml
source_dir: ./experiment
max_parallel: 2
job:
  name: lens-seeds
  entrypoint: run.py
  model: meta-llama-3-8b-instruct
  resources: {gpu: 1, min_vram_gb: 20}
  kwargs: {steps: 1000}
  required_outputs: [metrics.json]
elements:
  - index: 0
    kwargs: {seed: 42}
  - index: 1
    kwargs: {seed: 43}
```

Element kwargs override shared kwargs; `array_index` is reserved and injected
automatically. Instead of `elements`, use `indices: "0-99"` (also `0-20:2` or
`1,3,7`). Arrays contain 1–1,000 unique nonnegative indices. Request metadata is
bounded to 4 MiB and expanded pinned job metadata to 32 MiB. Source uploads happen
once; immutable asset bytes are shared, not copied per element.

All children and input pins commit atomically. Use a stable idempotency key for
uncertain submission retries; the CLI prints its generated key when omitted.
Replay with the same manifest/source/parameters and key returns the same array.
An altered request conflicts instead of submitting duplicates. SDK calls accept
an explicit key; persist it before making a request that may be interrupted.

## Inspect and control

```bash
topodrome arrays
topodrome array ARRAY_ID                 # counts and first 100 members
topodrome array ARRAY_ID --after 99      # indices greater than 99
topodrome array-set ARRAY_ID --max-parallel 3
topodrome array-set ARRAY_ID --hold      # pause future admissions only
topodrome array-set ARRAY_ID --resume
topodrome logs CHILD_JOB_ID
topodrome cancel CHILD_JOB_ID            # one element
topodrome array-cancel ARRAY_ID          # all unfinished elements
```

The response's `next_after` is the next pagination cursor; do not assume indices
are consecutive. Use full IDs for commands and asset references, never the four
characters shown by `top`.

Concurrency is enforced transactionally by the scheduler across restarts and
includes offered/starting work. Reducing the ceiling or holding an array does not
evict admitted work. GPU-free finalization releases the compute allowance, while
per-site upload concurrency and backlog bounds remain in force. Input prefetch
uses the existing bounded staging pipeline and does not reserve a GPU.

A failed element does not cancel siblings. Automatic retry stays per element;
successful results remain completed. After retry exhaustion, explicitly submit a
new array containing only the failed indices that the user wants rerun. Do not
resubmit an entire sweep after one failure or an uncertain HTTP response.
Cancellation persists and is reconciled after a caller disconnect or scheduler
restart. A canceled array cannot be resumed. Completed results retain their normal
artifact lifetimes; publish outputs needed permanently as assets.

## Python clients

```python
from topodrome.client.client import Client
from topodrome.domain.arrays import JobArraySpec

spec = JobArraySpec(
    job={"entrypoint": "run.py", "model": "meta-llama-3-8b-instruct",
         "resources": {"gpu": 1, "min_vram_gb": 20}},
    elements=[{"index": i, "kwargs": {"seed": 40 + i}} for i in range(10)],
    max_parallel=2,
)
with Client() as client:
    array = client.submit_array(spec, source_dir="experiment", idempotency_key="seeds-v1")
    detail = client.array(array["array_id"], after=-1, limit=100)
```

Both clients expose `submit_array`, `array`, `arrays`,
`update_array(id, max_parallel=..., held=...)`, and `cancel_array`.
Await these methods on `AsyncClient`; results are dictionaries. Owners can read
and control only their own arrays/elements; operators retain administrative access.

## Notifications and following

`--wait` follows aggregate counts, not interleaved element logs. Ctrl-C detaches
without cancellation. It waits until every element is terminal and exits nonzero
if any failed/canceled. Read one element's logs using its job ID.

`--notify-codex SESSION` on either submission command, or
`topodrome codex-notify ARRAY_ID --thread SESSION`, sends once when all elements
succeed or the first terminal failure/cancellation is observed. Retry-wait is not
failure. Remaining elements continue after an early alert; inspect their status
before starting follow-up work that needs the entire array. The bridge reads one
aggregate endpoint rather than polling every child. Unsubscribe with
`topodrome codex-unsubscribe ARRAY_ID` without canceling work. Normal local bridge,
profile, explicit opt-in, and uncertain-delivery rules in the tutorial still apply.

Dependencies are deliberately out of scope. After a completion notification, an
agent may submit follow-up jobs already authorized by the user, using committed
`job://` outputs or published assets. Notifications grant no new authorization.

## Workbench and timing

Arrays appear as summary rows in the Jobs table. Enter or click a parent to
expand/collapse its indexed elements inline; other jobs remain visible, and
multiple arrays can be expanded together. Enter on an element opens its usual
details; Esc returns to the same table. Press `h` to include finished elements.
Filtering and sorting keep children with their parent, and refresh preserves
expansion and selection. Submission and array mutations stay in the CLI/SDK.

Job rows show Wait / Run / I/O when the terminal is at least 58 columns wide;
narrower layouts retain the full breakdown in details. Wait sums queue, array allowance, hold and retry
backoff; Run is compute; I/O sums input staging and finalization. Details show
assignment, startup, checkpointing, cancellation and untracked history separately.
These non-overlapping totals cover all attempts and freeze at terminal completion.
Attempt details show startup, computation and finalization for that attempt.
They are scheduler-observed durations, not estimated remaining time or exact
kernel-profiler measurements. Finalization includes upload/verification and I/O
waiting. Phases not recorded before this upgrade are marked untracked, never
guessed. A completed computation is not success until finalization is durable.
