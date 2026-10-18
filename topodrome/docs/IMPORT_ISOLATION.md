# Experiment import isolation

Warm actors retain model weights and installed dependencies, not the previous
job's source modules. Each string entrypoint invocation receives a fresh import
scope, including retries of the same source bundle.

Both `run.py` and `package.module:run` support ordinary sibling, package,
namespace-package, transitive and lazy imports from the attempt workspace.
The scope lasts through asynchronous execution and actor cleanup callbacks.
Success, import errors and execution errors all restore the previous search path
and any installed modules temporarily shadowed by the source. Attempt-owned
modules and source-directory finder caches are removed afterward.

The public `resolve_entrypoint(..., source_root=...)` returns a synchronous
actor callable with a private module set. It restores those identities while
invoked, including while awaiting an async function; resolving another source
tree cannot change its helpers. Normal batch execution resolves anew per attempt.

Python import state is process-global. Local actors serialize source scopes;
production Ray actors use separate processes and remain parallel. Job threads
must finish before the entrypoint/cleanup returns. Do not retain job-defined
objects or hooks in the warm model between attempts. This is not a sandbox for
untrusted code, nor isolation of arbitrary changes to installed libraries.

Regression coverage: `tests/unit/test_runtime_import_isolation.py` checks warm
model reuse, helper collisions, retry state, package and lazy imports, async
execution, failure cleanup, resolved-callable identity, cleanup callbacks and
concurrent local actors.

Deployment requires an idle-gated site-agent/owned-Ray restart: changing the
files alone does not update Python already loaded in a warm worker. Use separate
site upgrade plans as described in `IDLE_UPGRADES.md`; running jobs finish before
their site's release is applied. No scheduler restart or schema change is needed.
