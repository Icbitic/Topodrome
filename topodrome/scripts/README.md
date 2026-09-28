# Real-environment smoke tests

These harnesses are intentionally outside the hermetic test suite. They read
credentials only from the mode-checked file at
`/home/xiangkun/.config/topodrome/secrets.env` (or `TOPODROME_SECRET_FILE`) and
never pass credential values in process arguments. Run them from the
`topodrome/` directory.

The wrappers default `UV_CACHE_DIR` to
`/ssd/xiangkun/.cache/topodrome-uv` so large package caches do not consume the
small root filesystem. An existing explicit `UV_CACHE_DIR` is preserved.
The spawned site agent retains Ray state under its configured scratch root;
long pytest or deployment paths use the validated `ray_short_temp_root` alias
so Ray's AF_UNIX sockets remain below the kernel path limit.

## Lab1

The Lab1 smoke samples all GPU indexes 0–7 with NVML, treats every unregistered
process as foreign, and selects the card with the most physical free VRAM above
the 20 GiB admission threshold. It loads
`/ssd/models/Meta-Llama-3-8B-Instruct` into a site-local Ray actor, performs two
one-token calls, and proves that both calls used the same actor and a single
model load.

```bash
scripts/run_lab1_smoke.sh
```

Set `TOPODROME_LAB1_GPU_INDEX=0` to constrain the test to GPU 0. The script
skips safely when the selected GPU no longer has 20 GiB free; it never signals
or terminates foreign processes.

The dedicated sharding opt-in selects exactly two admitted cards, constrains
each Transformers device-map entry to 10 GiB by default, and proves one actor
load, both local CUDA ordinals in the model map, exactly two entrypoint calls
for two requests, warm reuse, and exact actor cleanup:

```bash
scripts/run_lab1_smoke.sh --model-sharded
```

Use `TOPODROME_LAB1_SHARDED_GPU_INDICES=0,2` to constrain the physical pair,
`TOPODROME_SHARDED_SMOKE_FREE_VRAM_GIB` to change admission, or
`TOPODROME_SHARDED_SMOKE_MAX_MEMORY_GIB` to change the per-GPU device-map cap.

The complete scheduler-to-agent smoke submits one real job through the Python
SDK and a second through the CLI, then proves both used the same live model
object. Its default remains the original single-GPU path:

```bash
scripts/run_lab1_system_smoke.sh
```

Its separate sharding opt-in starts the real scheduler and agent with exactly
two selected Lab1 GPUs, requests one atomic degree-2 gang with an 11 GiB
admission requirement and a 10 GiB Transformers cap per device, and checks both
lease claim sets, both warm members, the actor PID/object identity, the ordered
`CUDA_VISIBLE_DEVICES` evidence, and device-map use of local ordinals 0 and 1:

```bash
scripts/run_lab1_system_smoke.sh --model-sharded
```

Only cards at or below 5% utilization and with at least 11 GiB physically free
are eligible. Set `TOPODROME_LAB1_SYSTEM_SHARDED_GPU_INDICES=0,2` to constrain
the pair. The harness treats unregistered processes as foreign and never
signals them.

### Last live Lab1 verification

On 2026-09-08, the complete forced two-GPU scheduler/agent smoke passed in
46.58 seconds on physical GPUs 2 and 3. Each device had more than the required
11 GiB free; the explicit shared-capacity run allowed their existing
utilization while preserving the 10 GiB model-map cap. Both SDK and CLI jobs
used one warm sharded actor, all nine pre-existing GPU processes remained
present, and the post-run audit found no Ray, scheduler, or site-agent process
owned by the harness.

## Terminal-only Jupyter

The fast probe authenticates with a header, discovers physical GPU memory, and
verifies the server-local model at
`/mnt/model/ckpts/Meta-Llama-3-8B-Instruct`, including the config and safetensor
index checksums against Lab1:

```bash
scripts/run_remote_jupyter_smoke.sh
```

The full option uploads a credential-free source archive, uses the server's
installed `uv` module, creates a uv-managed venv over the existing CUDA 12.8
Torch/Ray stack, overlays the required Transformers 4.57.6 backend, and runs
the same two-call Ray/model smoke against the local model. It does not contact
Hugging Face from the Jupyter server or redownload the CUDA runtime.

```bash
scripts/run_remote_jupyter_smoke.sh --with-model
```

The forced two-GPU variant has the same model-map, reuse, entrypoint-count, and
cleanup assertions. It defaults to physical indices 2 and 3 and can be
constrained with `TOPODROME_JUPYTER_SHARDED_GPU_INDICES`:

```bash
scripts/run_remote_jupyter_smoke.sh --model-sharded
```

The remote model tree is treated as read-only. When it lacks Hugging Face's
internal revision metadata, the harness uploads the 3 KiB operator-owned Lab1
revision manifest into its disposable workspace. The verifier hashes every
required remote model file against that trust anchor before loading it; it
does not download or alter the 15 GiB checkpoint.

Every terminal name has the form `topodrome_it_<run>_<nonce>` and is journaled
locally before creation. Cleanup deletes only names that both match that strict
underscore-only pattern and occur in the journal for the exact Jupyter
endpoint. It cannot delete other users' terminals. After a killed test, retry
the probe or run:

```bash
UV_CACHE_DIR=/ssd/xiangkun/.cache/topodrome-uv \
uv run --extra jupyter python -m scripts.cleanup_jupyter_test_terminals
```

This deliberately does not touch older hyphenated probe terminals or any
unregistered name.

### Last live Jupyter verification

On 2026-09-08, both opt-in Jupyter tests passed against the configured remote
site. The inventory probe verified that the remote model's `config.json` and
safetensor index checksums match the Lab1 copy. The forced oversized-model
smoke selected physical GPUs 2 and 3, loaded the local 8B model across both
CUDA ordinals, completed two one-token requests with the same Ray actor ID,
and reported `reused=true` with exactly two actor executions. The latest rerun
completed in 183.14 seconds without downloading the checkpoint.

The post-run audit found no journaled harness terminal and no disposable run
workspace. The remaining `.ipynb_checkpoints` directory is Jupyter-managed.
After actor shutdown both cards returned to their prior ambient workload state.
That live harness did not assume ambient processes were scavengers, and no
foreign process was signalled. Production site configuration now recognizes
only a direct Python invocation of the exact `gpu_idle_load.py` basename as a
cooperative self-yielding load; all other processes remain foreign. This
verifies the managed Terminal-to-site-local Ray path; it is not a central
scheduler-to-agent end-to-end test.

## Paid AutoDL RTX 5090

This section describes the paid integration harness, not production instance
bootstrap. Production uses the generated, secret-free release documented under
"AutoDL production bootstrap" in the project README; its uv environment and all
runtime caches live under `/root/autodl-tmp`. Production can bind the same
Lab1-hosted `site-packages/` seed to the exact `uv.lock` and worker platform,
fetch it with the bundled rclone, and then require a frozen offline `uv sync`;
the seed is not stored in the 20 GiB Beijing-B regional filesystem.

First add `TOPODROME_HF_TOKEN` to the protected secret file. The paid smoke has
three independent cost bounds:

- an exact command-line consent flag before any provider mutation;
- a provider-side shutdown command installed at create/power-on time;
- a host-side deadline and a `finally` block that powers off all instances
  owned by the exact crash-resumable test run.

Provider mutation intents are committed to a persistent SQLite journal before
the API call. A killed harness resumes the same run and reconciles rather than
blindly creating another instance. Its fsync'd run state advances monotonically
from initial verification to retained restart or replacement, then to
acceptance-verified cleanup. The safe boundary records the exact initial UUID,
canonical create-region key, and marker digest; a replacement boundary also
records its fresh UUID. A restart after releasing the initial 2x5090 therefore
continues the replacement phase and can never reinterpret either released
capacity or its replacement as the initial instance. Final cleanup audits both
unresolved and terminal applied CREATE journal rows, so a provider UUID remains
recoverable even if the process died before returning it to caller-local state.
By default it requests exactly one `5090-p`,
checks the wallet and total instance hourly price, bootstraps with `uv`,
downloads the exact Llama revision from `https://hf-mirror.com` into
`/root/autodl-fs`, proves actor reuse, deletes its temporary credential file,
and retains the instance in the powered-off state.

```bash
scripts/run_autodl_5090_smoke.sh --i-understand-this-creates-paid-capacity
```

The two-GPU variant requires separate capacity and destructive-replacement
consent phrases plus an explicit total-instance price cap. It sends one official
Pro API create request with `req_gpu_amount=2`; it never creates or gangs two
single-GPU instances. The smoke launches one model-sharded actor, exposes both
admitted devices, and fails unless the Transformers device map uses local CUDA
ordinals 0 and 1. After verifying a durable regional marker, it releases only
that exact run-owned UUID, times creation of a fresh same-region 2x5090, verifies
the marker and model again, and finally powers the replacement off:

```bash
scripts/run_autodl_5090_smoke.sh \
  --gpu-count 2 \
  --max-total-hourly-cny 6.00 \
  --i-understand-this-creates-2x5090-paid-capacity \
  --i-understand-this-releases-and-replaces-2x5090-paid-capacity
```

The one-GPU defaults are a 2-hour hard watchdog and a ¥3.00/hour total-instance
ceiling. They may only be tightened or explicitly adjusted within the harness
limits:

```bash
TOPODROME_AUTODL_TEST_WATCHDOG_SECONDS=3600 \
TOPODROME_AUTODL_MAX_HOURLY_CNY=3.00 \
scripts/run_autodl_5090_smoke.sh --i-understand-this-creates-paid-capacity
```

The normal controller separately applies Topodrome's 10-continuous-minute idle
policy after checking all five activity gates. No smoke test creates dummy GPU
work merely to postpone that shutdown.

For a cold disposable worker, the harness can use a Lab1-built portable
`site-packages` seed while still letting `uv` own the project environment. Keep
that archive on Lab1 and pass its absolute path; it is copied only to
`/root/autodl-tmp`, never to the limited regional filesystem:

```bash
TOPODROME_AUTODL_RUNTIME_SEED=/ssd/xiangkun/.tmp/topodrome-runtime-site-packages.tar.gz \
scripts/run_autodl_5090_smoke.sh --i-understand-this-creates-paid-capacity
```

### Last live AutoDL verification

On 2026-09-08, the complete 2x5090 sharded release/replacement acceptance passed
in 1243.01 seconds. Initial capacity reached `running` in 22.881 seconds; the
initial bootstrap/model/reuse smoke took 561.277 seconds, including a 410.981
second upload of the 4.1 GiB ephemeral runtime seed. Provider-confirmed
power-off took 11.436 seconds. The exact run-owned instance was then released,
a distinct same-region 2x5090 reached `running` in 22.304 seconds, and its
replacement model/reuse smoke took 579.904 seconds, including a 417.458 second
runtime-seed upload.

The replacement recovered the SHA-256-verified Beijing-B marker and existing
Llama model from `/root/autodl-fs`; it did not download model weights. AutoDL's
observed rack sign was normalized to the documented `beijingDC2` create key
before release. The finalizer and an independent provider audit confirmed zero
running instances, exactly one stopped run-owned 2x5090 replacement, and the
untouched stopped legacy 1x5090. The two latest seed transfers measured about
79--80 Mbit/s.

The historical single-GPU verification remains useful for retained-instance
power-cycle timing:

On 2026-09-08, the retained-instance smoke passed in 2128.17 seconds. It
created one Beijing-B RTX 5090 worker, transferred the ephemeral runtime from
Lab1, acquired the exact Llama payload from the configured Hugging Face mirror
while excluding `original/*`, loaded it once into a Ray actor, and completed
two calls through that actor. It then powered the worker off, powered the same
retained worker back on, recovered the regional marker, reused both the
ephemeral runtime and persistent model, repeated the actor-reuse check, and
powered the worker off again.

A separate single-GPU live replacement test then durably journaled release of that worker,
created a distinct RTX 5090 in the same Beijing-B storage region, recovered the
regional marker and existing model without another model download, rebuilt
only the replaceable runtime under `/root/autodl-tmp`, and passed the same
two-call actor-reuse assertion. A final independent inventory check found
exactly one run-owned replacement in `stopped` state.

The earlier Lab1-to-AutoDL runtime transfer was approximately 97--115 Mbit/s;
the initial mirror download was approximately 18--20 MB/s. After removing only
the three hash-verified legacy runtime paths, regional persistent usage was
16,069,810,720 bytes (about 14.97 GiB), leaving the model and integration marker
intact. Provider UUIDs and credentials are deliberately omitted from this log.
