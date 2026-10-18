from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import ModuleType

import pytest

from topodrome.runtime import (
    ActorKey,
    ExperimentContext,
    LoadedModel,
    ModelKey,
    invoke_experiment,
    resolve_entrypoint,
)
from topodrome.runtime.actor import LocalModelActor
from topodrome.runtime.source_imports import SourceImports


def source(root: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def context(root: Path) -> ExperimentContext:
    return ExperimentContext(
        model=object(),
        tokenizer=None,
        device="cpu",
        job_id="job-test",
        attempt_id="attempt-test",
        workspace=root,
    )


def actor(model: object, loads: list[str]) -> LocalModelActor:
    key = ActorKey(
        site_id="test",
        gpu_uuid="test-gpu",
        model=ModelKey(repo_id="test/test", revision="test", backend="test", backend_version="1"),
    )

    def load(key: ModelKey) -> LoadedModel:
        loads.append(key.repo_id)
        return LoadedModel(model=model, device="cpu")

    return LocalModelActor(key, load, dirty_on_experiment_error=False)


def run(worker: LocalModelActor, root: Path, entrypoint: str = "run.py") -> object:
    return worker.run(entrypoint, job_id="job-test", attempt_id="attempt-test", workspace=root)


def test_same_warm_actor_loads_each_jobs_helpers_without_reloading_model(tmp_path: Path) -> None:
    loads: list[str] = []
    model = object()
    worker = actor(model, loads)
    path_before = sys.path[:]
    synthetic_before = {name for name in sys.modules if name.startswith("_topodrome_experiment_")}
    try:
        results = []
        for value in ("first", "second"):
            root = source(
                tmp_path / value,
                {
                    "run.py": "import td_job_helper\ndef run(context):\n"
                    "    return td_job_helper.VALUE, td_job_helper.__file__, context.model\n",
                    "td_job_helper.py": f"VALUE = {value!r}\n",
                },
            )
            results.append(run(worker, root))
            assert "td_job_helper" not in sys.modules
        assert results == [
            (value, str(tmp_path / value / "td_job_helper.py"), model)
            for value in ("first", "second")
        ]
        assert loads == ["test/test"]
        assert worker.describe().executions == 2
        assert sys.path == path_before
        synthetic_after = {n for n in sys.modules if n.startswith("_topodrome_experiment_")}
        assert synthetic_after == synthetic_before
    finally:
        worker.evict()


def test_retry_does_not_reuse_mutable_module_state(tmp_path: Path) -> None:
    root = source(
        tmp_path,
        {
            "run.py": "import td_job_helper\ndef run(context):\n"
            "    td_job_helper.COUNT += 1\n    return td_job_helper.COUNT\n",
            "td_job_helper.py": "COUNT = 0\n",
        },
    )
    worker = actor(object(), [])
    try:
        assert [run(worker, root), run(worker, root)] == [1, 1]
    finally:
        worker.evict()


@pytest.mark.parametrize("entrypoint", ["run.py", "td_job_package.entry:run"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_lazy_transitive_package_imports_stay_in_scope(
    tmp_path: Path, entrypoint: str, asynchronous: bool
) -> None:
    for value in ("first", "second"):
        head = "async def" if asynchronous else "def"
        wait = "    import asyncio\n    await asyncio.sleep(0)\n" if asynchronous else ""
        root = source(
            tmp_path / value,
            {
                "run.py": f"{head} run(context):\n{wait}"
                "    from td_job_package.helper import value\n    return value()\n",
                "td_job_package/__init__.py": "",
                "td_job_package/entry.py": f"{head} run(context):\n{wait}"
                "    from .helper import value\n    return value()\n",
                "td_job_package/helper.py": "def value():\n"
                "    from td_job_namespace.values import VALUE\n    return VALUE\n",
                # No __init__.py: namespace packages must be removed as well.
                "td_job_namespace/values.py": f"VALUE = {value!r}\n",
            },
        )
        assert invoke_experiment(entrypoint, context(root)) == value
        assert not any(n.startswith(("td_job_package", "td_job_namespace")) for n in sys.modules)


def test_resolved_callable_reinstalls_its_own_modules_and_supports_dataclasses(
    tmp_path: Path,
) -> None:
    resolved = []
    for value in ("first", "second"):
        root = source(
            tmp_path / value,
            {
                "run.py": "from __future__ import annotations\nfrom dataclasses import dataclass\n"
                "import td_job_helper\n@dataclass\nclass Result:\n    value: str\n"
                "def run(context):\n    import importlib, pickle\n"
                "    helper = importlib.import_module('td_job_helper')\n"
                "    assert helper is td_job_helper\n"
                "    return pickle.loads(pickle.dumps(Result(helper.VALUE))).value\n",
                "td_job_helper.py": f"VALUE = {value!r}\n",
            },
        )
        resolved.append((resolve_entrypoint("run.py", source_root=root), context(root)))
        assert "td_job_helper" not in sys.modules
    assert [function(ctx) for function, ctx in reversed(resolved)] == ["second", "first"]


def test_shadowed_installed_package_is_restored_and_other_dependencies_stay_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = ModuleType("td_job_package")
    package.__path__ = [str(tmp_path.parent / "installed")]
    old_helper = ModuleType("td_job_package.helper")
    old_helper.VALUE = "installed"
    package.helper = old_helper
    dependency = ModuleType("td_shared_dependency")
    monkeypatch.setitem(sys.modules, "td_job_package", package)
    monkeypatch.setitem(sys.modules, "td_job_package.helper", old_helper)
    monkeypatch.setitem(sys.modules, "td_shared_dependency", dependency)
    root = source(
        tmp_path,
        {
            "run.py": "def run(context):\n    from td_job_package.helper import VALUE\n"
            "    import td_shared_dependency\n    return VALUE, td_shared_dependency\n",
            "td_job_package/__init__.py": "",
            "td_job_package/helper.py": "VALUE = 'job'\n",
        },
    )
    assert invoke_experiment("run.py", context(root)) == ("job", dependency)
    assert sys.modules["td_job_package"] is package
    assert sys.modules["td_job_package.helper"] is old_helper
    assert package.helper is old_helper
    assert sys.modules["td_shared_dependency"] is dependency


@pytest.mark.parametrize("phase", ["import", "run"])
@pytest.mark.parametrize("error", ["RuntimeError", "SystemExit"])
def test_exception_cleans_partial_imports_and_restores_search_path(
    tmp_path: Path, phase: str, error: str
) -> None:
    statement = f"raise {error}('expected failure')\n"
    root = source(
        tmp_path,
        {
            "run.py": "import td_job_helper\n"
            + (
                statement + "def run(context): pass\n"
                if phase == "import"
                else "def run(context):\n    " + statement
            ),
            "td_job_helper.py": "VALUE = 'failing'\n",
        },
    )
    before = sys.path[:]
    expected = RuntimeError if error == "RuntimeError" else SystemExit
    with pytest.raises(expected, match="expected failure"):
        invoke_experiment("run.py", context(root))
    assert sys.path == before
    assert "td_job_helper" not in sys.modules
    root = source(
        tmp_path / "next",
        {
            "run.py": "import td_job_helper\ndef run(context): return td_job_helper.VALUE\n",
            "td_job_helper.py": "VALUE = 'healthy'\n",
        },
    )
    assert invoke_experiment("run.py", context(root)) == "healthy"


def test_cleanup_callbacks_can_import_the_same_attempt_modules(tmp_path: Path) -> None:
    model: list[int] = []
    root = source(
        tmp_path,
        {
            "run.py": "import td_job_helper\ndef run(context):\n"
            "    td_job_helper.COUNT += 1\n"
            "    def cleanup():\n        import td_job_helper\n"
            "        context.model.append(td_job_helper.COUNT)\n"
            "    context.add_cleanup(cleanup)\n",
            "td_job_helper.py": "COUNT = 0\n",
        },
    )
    worker = actor(model, [])
    try:
        run(worker, root)
        run(worker, root)
        assert model == [1, 1]
        assert "td_job_helper" not in sys.modules
    finally:
        worker.evict()


def test_local_actors_cannot_race_over_process_global_imports(tmp_path: Path) -> None:
    first_entered, release_first, second_attempted = Event(), Event(), Event()
    workers = [actor((first_entered, release_first), []), actor(object(), [])]
    roots = []
    for value in ("first", "second"):
        roots.append(
            source(
                tmp_path / value,
                {
                    "run.py": "import td_job_helper\ndef run(context):\n"
                    "    if isinstance(context.model, tuple):\n"
                    "        entered, release = context.model\n        entered.set()\n"
                    "        assert release.wait(5)\n"
                    "    import importlib\n"
                    "    return importlib.import_module('td_job_helper').VALUE\n",
                    "td_job_helper.py": f"VALUE = {value!r}\n",
                },
            )
        )

    def second() -> object:
        second_attempted.set()
        return run(workers[1], roots[1])

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(run, workers[0], roots[0])
            try:
                assert first_entered.wait(5)
                other = executor.submit(second)
                assert second_attempted.wait(5)
                assert not other.done()
            finally:
                release_first.set()
            assert (first.result(5), other.result(5)) == ("first", "second")
    finally:
        for worker in workers:
            worker.evict()


def test_async_worker_thread_import_and_search_path_replacement_are_cleaned(tmp_path: Path) -> None:
    root = source(
        tmp_path,
        {
            "run.py": "import asyncio, sys\nasync def run(context):\n"
            "    helper = await asyncio.to_thread(__import__, 'td_job_helper')\n"
            "    sys.path = ['unrelated']\n    return helper.VALUE\n",
            "td_job_helper.py": "VALUE = 'async'\n",
        },
    )
    path_object, before = sys.path, sys.path[:]
    assert invoke_experiment("run.py", context(root)) == "async"
    assert sys.path is path_object
    assert sys.path == before
    assert "td_job_helper" not in sys.modules
    assert not any(Path(p).is_relative_to(root) for p in sys.path_importer_cache)


def test_active_event_loop_rejection_does_not_leak_modules(tmp_path: Path) -> None:
    root = source(
        tmp_path,
        {
            "run.py": "import td_job_helper\nasync def run(context): return td_job_helper.VALUE\n",
            "td_job_helper.py": "VALUE = 'async'\n",
        },
    )

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="active event loop"):
            invoke_experiment("run.py", context(root))
        assert "td_job_helper" not in sys.modules

    asyncio.run(exercise())


def test_newly_imported_installed_dependency_remains_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = source(
        tmp_path / "installed", {"td_external_dependency.py": "IDENTITY = object()\n"}
    )
    monkeypatch.syspath_prepend(str(installed))
    try:
        identities = []
        for name in ("first", "second"):
            root = source(
                tmp_path / name,
                {
                    "run.py": "import td_external_dependency\ndef run(context):\n"
                    "    return td_external_dependency.IDENTITY\n",
                },
            )
            identities.append(invoke_experiment("run.py", context(root)))
        assert identities[0] is identities[1]
        assert "td_external_dependency" in sys.modules
    finally:
        sys.modules.pop("td_external_dependency", None)


def test_nested_source_invocation_restores_outer_helpers(tmp_path: Path) -> None:
    inner = source(
        tmp_path / "inner",
        {
            "run.py": "import td_job_helper\ndef run(context): return td_job_helper.VALUE\n",
            "td_job_helper.py": "VALUE = 'inner'\n",
        },
    )
    outer = source(
        tmp_path / "outer",
        {
            "run.py": "import td_job_helper\ndef run(context):\n"
            "    inner = context.model()\n    import td_job_helper\n"
            "    return inner, td_job_helper.VALUE\n",
            "td_job_helper.py": "VALUE = 'outer'\n",
        },
    )
    ctx = context(outer)
    ctx.model = lambda: invoke_experiment("run.py", context(inner))
    assert invoke_experiment("run.py", ctx) == ("inner", "outer")
    assert "td_job_helper" not in sys.modules


def test_scope_setup_error_does_not_leave_actor_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = actor(object(), [])

    def fail(_self: SourceImports) -> set[str]:
        raise OSError("source unavailable")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(SourceImports, "_names", fail)
            with pytest.raises(OSError, match="source unavailable"):
                run(worker, tmp_path)
        assert worker.is_reusable
        assert worker.describe().pending_runs == 0
        assert worker.describe().executions == 1
    finally:
        worker.evict()
