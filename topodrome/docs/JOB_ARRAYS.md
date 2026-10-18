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
  model: meta-llama/Meta-Llama-3-8B-Instruct
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

### Checkpoint sweeps and different models

An element may override `model` and resource fields. Omitted `model` inherits
the base job; explicit `model: null` requests a model-free GPU job, not inheritance.
A model override is a complete model address (include `repo_id`); omitted revision
selects that repository's registered default. Resource overrides merge only the
specified fields with the base resources. Every expanded job must independently
satisfy GPU/parallelism and model-free VRAM requirements.

```yaml
source_dir: ./experiment
max_parallel: 2
job:
  name: olmo-checkpoint-evaluation
  entrypoint: evaluate.py
  model: allenai/Olmo-3-1025-7B
  resources: {gpu: 1, min_vram_gb: 16}
  required_outputs: [metrics.json]
elements:
  - index: 0
    model: {repo_id: allenai/Olmo-3-1025-7B, revision: stage1-step1000}
  - index: 1
    model: {repo_id: allenai/Olmo-3-1025-7B, revision: stage1-step2000}
  - index: 2
    model: EleutherAI/pythia-160m
```

Use `run(context, array_index, **kwargs)` and the supplied model/tokenizer. Do not
manually download or replace model weights inside the entry point. Each selection
is resolved to its own immutable registered commit before the array commits;
an unknown repository/revision rejects the entire submission. Source and asset
resolution still happen once. Retries keep the selected checkpoint, child ID,
index and ordinary output isolation. Replaying an accepted idempotency key does
not re-resolve moving defaults. `max_parallel` still counts children, not GPUs.

For all registered checkpoints, discover once and submit once with either SDK:

```python
from topodrome.client.client import Client

repo = "allenai/Olmo-3-1025-7B"
with Client() as client:
    snapshots = {}
    for row in client.models(repo_id=repo):
        snapshots.setdefault((row["revision"], row.get("subfolder")), row)
    elements = []
    ordered = sorted(snapshots.items(), key=lambda item: (item[0][0], item[0][1] or ""))
    for index, ((revision, subfolder), row) in enumerate(ordered):
        model = {"repo_id": repo, "revision": revision}
        if subfolder is not None:
            model["subfolder"] = subfolder
        elements.append({"index": index, "model": model,
                         "kwargs": {"checkpoint_refs": row["refs"]}})
    result = client.submit_array(
        job={"entrypoint": "evaluate.py", "model": repo,
             "resources": {"gpu": 1, "min_vram_gb": 16},
             "required_outputs": ["metrics.json"]},
        elements=elements, max_parallel=2, source_dir="experiment",
        idempotency_key="olmo-evaluation-v1",
    )
```

The entry point should accept `checkpoint_refs` and record it in provenance.
Execution order is not training-step order; use the full returned labels when
analyzing results. Repeated sites/warm replicas are not additional checkpoints.
For multiple repositories, query `client.models()` once, filter the intended
repository IDs, and deduplicate by repository/revision/subfolder.

### Reusable checkpoint workers

The site may retain a compatible idle Transformers worker across children or
ordinary jobs. Only after the next exact GPU gang is physically admitted does
it replace weights. Compatible, local safetensors checkpoints with matching
functional configuration/layout retain parameter storage; changed configuration,
untracked hooks or unsupported weight layouts reconstruct the model in the same
worker. Changed runtime/resource profiles and private/distributed/adapter workers
use normal worker disposal instead. Any partial switch failure disposes the exact
worker before claims can be released. The new checkpoint identity is published
only after a successful load and CUDA synchronization. Separate child contexts,
imports, outputs, retries, cancellation and priorities are unchanged.
Liveness reconciliation and replacement are serialized per actor; a loading
worker is neither advertised under its old checkpoint nor evicted as stale.

For mixed-model arrays, a bounded advisory next-model hint can warm the operating
system's reclaimable file cache while the current child computes. It reads only
already verified local model files: no download, GPU allocation, permanent copy,
or admission of the hinted job. The scheduler may choose another child/site, so
prefetch is best-effort, not a promise or dependency. Site settings default to
`checkpoint_prefetch_max_bytes: 17179869184` (16 GiB per checkpoint),
`checkpoint_prefetch_concurrency: 1` (site-wide), and
`checkpoint_prefetch_memory_floor_bytes: 17179869184` (16 GiB available RAM).
Zero maximum bytes disables read-ahead. Readers have an 8-MiB application buffer
and a 120-second cooperative deadline; the OS decides page-cache residency.

Keep all useful analyses of one checkpoint in its child and batch model calls.
Worker reuse avoids repeated process/CUDA setup, not the need to read distinct
weights. It does not bypass physical GPU safety, queue ordering, or fairness.

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
    job={"entrypoint": "run.py", "model": "meta-llama/Meta-Llama-3-8B-Instruct",
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
