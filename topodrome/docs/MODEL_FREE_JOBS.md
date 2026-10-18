# GPU jobs without a preloaded model

Use an explicit YAML `model: null` (Python `model=None`). This is a GPU batch
job, not a CPU job or a model alias. Topodrome skips model catalog lookup,
acquisition and Transformers loading. Your code may construct its own model or
call another training/experiment library inside `run(context, **kwargs)`.

```yaml
entrypoint: gpu_compute.py
source_dir: examples
model: null
resources:
  gpu: 1
  min_vram_gb: 2
  cpu: 2
  memory_gb: 4
required_outputs: [gpu-result.json]
retry:
  max_attempts: 2
```

`min_vram_gb` must be positive and cover the **peak per-GPU** memory requirement
of your code, including any model/optimizer/activations it loads. Admission
adds the site's safety margin. This is a capacity check, not a CUDA allocator
quota; scripts must stay within their declared requirements. GPU-free file
access still uses the direct download/upload commands, not a batch job.

## Devices and execution modes

| Manifest | Entrypoint invocations | `context.devices` | Rank metadata |
| --- | --- | --- | --- |
| `gpu: 1`, `execution: single` (default) | Once | `(torch.device('cuda:0'),)` | rank 0, world size 1 |
| `gpu: N`, `execution: single` | Once, owning the complete gang | `cuda:0` through `cuda:N-1` | rank 0, world size 1 |
| `gpu: N`, `execution: distributed` | Once per GPU, N >= 2 | Each worker sees only `(cuda:0,)` | rank 0 through N-1, world size N |

All entries are real `torch.device` objects. `context.device` is the first
entry in `context.devices`; `context.local_rank` is zero for a native worker
with one visible GPU. `context.is_main_process` is `context.rank == 0`.
Never substitute host GPU indices or change `CUDA_VISIBLE_DEVICES`. The site
agent selects admitted physical UUIDs before Torch is imported in the worker.
The scheduler leases all GPUs atomically on **one site**. Two 80-GiB GPUs do
not become a transparent 160-GiB device: your library/code must implement
sharding, data parallelism or explicit placement.

For native distributed execution Topodrome provides `RANK`, `WORLD_SIZE`,
`LOCAL_RANK=0`, `MASTER_ADDR` and `MASTER_PORT`. Initialize a process group in
the entrypoint using that environment (or `context.rank/world_size`), operate
on `context.device`, and register `destroy_process_group` as a cleanup hook.
Do not invoke `torchrun` again inside each rank. Only rank zero owns shared
checkpoint/artifact commits and advisory progress. Other ranks may write
distinct files under the shared workspace for rank zero to register after
collective completion; scratch/output names must not collide. Preserve the
same collective ordering on every rank, including cooperative preemption.

## Context and lifecycle

`context.model` and `context.tokenizer` are both `None`. The rest of the ordinary
context is unchanged: workspace, committed immutable inputs, scratch, progress,
checkpoint interval/restore, artifact registration, cleanup and preemption.
Libraries should accept the supplied device(s) and use a thin callback adapter
for progress/checkpoints; Topodrome does not promise framework-specific APIs.
Declare extra dependencies in the existing hash-pinned `runtime.pip` contract,
not by installing packages from inside the entrypoint.

Every model-free attempt owns a private process or worker group. It is stopped
and confirmed before its GPU lease is released; it is never advertised as a
warm model or protected by the reusable-model floor. Upload finalization can
continue without occupying the GPUs. Retries retain the job ID, get a new
attempt ID and restore the scheduler-selected checkpoint when available.
Checkpoint resumable work at safe boundaries, and persist only final results
plus compact metrics when possible. Artifacts need not become permanent assets.

## Runnable examples

```bash
topodrome submit examples/gpu_compute.yaml --wait
topodrome submit examples/gpu_compute_gang.yaml --wait
topodrome submit examples/gpu_compute_distributed.yaml --wait
```

Each example writes a JSON device/result report and uses the ordinary context
progress, cleanup and checkpoint APIs. Existing preloaded-model jobs keep using
`model.parallelism`; do not add `execution` to those manifests. During a rolling
upgrade model-free jobs wait until an updated agent advertises support.
