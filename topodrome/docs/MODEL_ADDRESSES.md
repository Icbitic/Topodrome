# Registered model repositories and checkpoints

Topodrome addresses models like Hugging Face: a namespaced repository ID,
an optional revision, and an optional subfolder. The old `alias` field and
unqualified model names are not supported.

```yaml
model:
  repo_id: meta-llama/Meta-Llama-3-8B-Instruct
  # revision omitted: pin the operator-registered default at submission
```

The shorthand `model: meta-llama/Meta-Llama-3-8B-Instruct` means the same thing.
Explicit `model: null` still requests GPU execution without a preloaded model.

For checkpoint experiments:

```yaml
model:
  repo_id: EleutherAI/pythia-1b
  revision: step20000
  # subfolder: checkpoints/step20000  # only when the registered snapshot uses it
```

Only operator-registered repositories **and snapshots** are allowed. A repository
name is not permission to fetch arbitrary revisions. Unknown selections fail
submission instead of silently running the default weights. Tags/branches in
`refs` are resolved from the registered catalog to its immutable revision before
dispatch; the scheduler makes no Hub lookup and workers receive no moving ref.
Changing the registered default/label later does not change an accepted job.
HF origins must be pinned to a full commit hash. Existing verified weights are
reused; renaming logical addresses does not download or move them.

Credential-free HTTPS model-page URLs also work:

```yaml
model: https://huggingface.co/EleutherAI/pythia-1b/tree/step20000
```

`hf-mirror.com` page URLs normalize identically. For revision names containing
slashes, use the explicit mapping form, since URL path components would be
ambiguous. Credentials, query strings, non-HF hosts, absolute subfolders and
path traversal are rejected. This convenience is not arbitrary URL loading.

## Discover checkpoint labels

Query the registered step/branch/tag labels instead of guessing them or reading
operator configuration:

```bash
topodrome models --repo EleutherAI/pythia-160m
```

Omit `--repo` to list all advertised model capabilities. `--repo-id` is an alias.
Both SDKs accept `client.models(repo_id="EleutherAI/pythia-160m")` (await it with
`AsyncClient`); the HTTP endpoint is `GET /api/v1/models?repo_id=...`.

Each site/capability row includes `repo_id`, immutable `revision`, `subfolder`,
`refs` (registered labels such as `step20000`), `is_default`, and `availability`.
Labels/defaults come from the scheduler's loaded catalogue, joined by repository,
commit and subfolder. Queries do not contact Hugging Face or expose model paths.
Unmatched site reports have no inferred labels and are not marked default.
An unknown canonical repository filter returns an empty list; malformed IDs
are rejected. Site availability is the reported model capability, not a promise
that a GPU is immediately schedulable.

Several sites or warm replicas can advertise the same snapshot. When generating
a sweep, deduplicate by `(repo_id, revision, subfolder)` and preserve the chosen
label and immutable commit in output provenance. For example:

```python
from topodrome.client import Client

with Client() as client:
    rows = client.models(repo_id="EleutherAI/pythia-160m")
    snapshots = {(row["revision"], row["subfolder"]): row for row in rows}
```

Use the full discovered label in `model.revision`, or its returned commit hash.
Step naming is repository-specific: Pythia uses names such as `step20000`, OLMo
Instruct uses `step_1000`, and OLMo base includes stage/ingredient/token qualifiers.
Do not treat a numeric step alone as a universal or unambiguous address.

## Operator catalog

Catalog keys are repository IDs. The root snapshot is the default; optional
`snapshots` register further immutable versions. Backend/dtype/VRAM defaults
apply to all snapshots of that repository. Each snapshot has its own site
locations and optional labels. For example (illustrative, not an installed model):

```yaml
models:
  local/training-run-a:
    revision: snapshot-20000-v1
    refs: [step20000]
    backend: transformers
    dtype: bfloat16
    min_vram_gb: 4
    sites:
      lab1:
        path: /ssd/models/training-run-a/step20000
        local_files_only: true
    snapshots:
      - revision: snapshot-40000-v1
        refs: [step40000]
        sites:
          lab1:
            path: /ssd/models/training-run-a/step40000
            local_files_only: true
```

Local repository IDs need no HF upload. Their revision IDs and files must be
operator-managed, immutable snapshots, never directories overwritten by a
training loop. Paths identify the repository snapshot root; `subfolder`, when
set, identifies a complete Transformers model directory beneath it. The revision
manifest remains at the snapshot root and uses repository-relative file names.
Separate revisions must not share a writable acquisition/cache destination.
Catalog snapshots/ref labels are unambiguous for each revision/subfolder pair.

For a downloadable snapshot, set `origin` to its HF repository, `revision` to
the full commit, `endpoint: https://hf-mirror.com`, and a private snapshot-specific
`persistent_path`. Acquisition filters to a selected subfolder when present.
Existing admission/verification checks and site-local caching still apply.
Catalog/model changes use the idle-gated upgrade workflow, not policy-only reload.

Actor identities, site capabilities, placement and warm reuse include repository,
exact revision **and subfolder**, plus the existing backend/dtype/runtime/gang
identity. Different checkpoints cannot accidentally share a loaded actor.

Use ordinary jobs or arrays for independent experiments. An array currently
shares one model selection across its elements: use a separate array per checkpoint
when sweeping both checkpoints and seeds/shards. This feature does not implement
automatic checkpoint discovery, a series job type, hot weight swapping, or
distributed training. The normal `context.model`, tokenizer, assigned devices,
progress, checkpoints and output APIs are unchanged.

## Existing deployment cutover

The five configured models now use their real repository IDs:

| Repository | Availability |
| --- | --- |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Registered institutional/AutoDL locations |
| `google/gemma-3-12b-it` | Registered institutional/AutoDL locations |
| `allenai/OLMo-2-1124-7B-Instruct` | Registered institutional/AutoDL locations |
| `Qwen/Qwen3-4B-Instruct-2507` | Registered institutional/AutoDL locations |
| `meta-llama/Meta-Llama-3-70B-Instruct` | Lab1 only, explicit two-GPU sharding |

Migration `0020_repository_models` converts existing job/kernel model metadata
and capability/actor identities, preserving job IDs, owners, outputs, reservations
and history. It refuses active leases and unknown old names. This is not runtime
backward compatibility: update manifests/clients before new submissions. Changed
manifests need new replay keys; never resubmit historical arrays to convert them.
Rollback requires the fenced pre-upgrade database/config backup, not a lossy
mapping from arbitrary new repositories back to old aliases.
