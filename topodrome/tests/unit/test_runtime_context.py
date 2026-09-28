from __future__ import annotations

from pathlib import Path

import pytest

from topodrome.runtime import (
    CheckpointRestore,
    CleanupError,
    ExperimentContext,
    PreemptionToken,
    invoke_experiment,
    resolve_entrypoint,
)
from topodrome.runtime.checkpoint import CheckpointManager


def make_context(tmp_path: Path, **overrides: object) -> ExperimentContext:
    values = {
        "model": object(),
        "tokenizer": object(),
        "device": "cuda:0",
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "workspace": tmp_path,
        **overrides,
    }
    return ExperimentContext(**values)


def test_context_exposes_attempt_services(tmp_path: Path) -> None:
    checkpoints = []
    artifacts = []
    token = PreemptionToken()
    context = make_context(
        tmp_path,
        preemption=token,
        checkpoint_handler=checkpoints.append,
        artifact_handler=artifacts.append,
    )
    output = tmp_path / "result.json"
    output.write_text("{}")

    checkpoint = context.checkpoint({"next_shard": 4}, name="progress", metadata={"portable": True})
    artifact = context.register_artifact("result.json", logical_path="metrics/result.json")

    assert checkpoint.sequence == 1
    assert checkpoints == [checkpoint]
    assert artifacts == [artifact]
    assert artifact.source == output
    assert not context.should_preempt()
    assert token.request("foreign GPU pressure")
    assert context.should_preempt()
    assert context.preemption_reason == "foreign GPU pressure"


def test_context_restores_only_the_scheduler_selected_checkpoint(tmp_path: Path) -> None:
    reference = "jobs/job-1/attempts/attempt-0/checkpoints/00000001-state.pkl"
    context = make_context(
        tmp_path,
        checkpoint_restore=CheckpointRestore(reference, {"next_shard": 4}),
    )

    assert context.has_checkpoint
    assert context.checkpoint_ref == reference
    assert context.restore_checkpoint() == {"next_shard": 4}

    fresh = make_context(tmp_path, attempt_id="attempt-2")
    assert not fresh.has_checkpoint
    assert fresh.restore_checkpoint({"next_shard": 0}) == {"next_shard": 0}


def test_checkpoint_interval_and_sequence_advance_only_after_persistence() -> None:
    current = [100.0]
    fail = [True]

    def persist(record: object) -> str:
        if fail[0]:
            fail[0] = False
            raise OSError("temporary storage failure")
        return "jobs/job/attempts/attempt/checkpoints/00000001-state.pkl"

    manager = CheckpointManager(
        persist,
        interval_seconds=10,
        clock=lambda: current[0],
    )
    assert not manager.due()
    current[0] = 110.0
    assert manager.due()

    with pytest.raises(OSError):
        manager.save({"step": 1})
    assert manager.records == ()
    assert manager.due()

    saved = manager.save({"step": 1})
    assert saved.sequence == 1
    assert not manager.due()
    assert manager.latest_reference is not None


def test_cleanup_is_lifo_and_can_be_disarmed(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    calls: list[str] = []
    context.add_cleanup(calls.append, "first")
    canceled = context.add_cleanup(calls.append, "canceled")
    context.add_cleanup(calls.append, "last")
    canceled.cancel()

    context.close()
    context.close()

    assert calls == ["last", "first"]
    assert context.closed


def test_cleanup_attempts_every_callback_and_reports_all_failures(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    calls: list[str] = []

    def fail(name: str) -> None:
        calls.append(name)
        raise RuntimeError(name)

    context.add_cleanup(fail, "first")
    context.add_cleanup(fail, "second")

    with pytest.raises(CleanupError) as caught:
        context.close()

    assert calls == ["second", "first"]
    assert len(caught.value.failures) == 2


@pytest.mark.parametrize("logical_path", ["/absolute", "../escape", "a/../../escape", ""])
def test_artifact_logical_paths_cannot_escape(tmp_path: Path, logical_path: str) -> None:
    output = tmp_path / "output"
    output.write_text("data")
    context = make_context(tmp_path)

    with pytest.raises(ValueError):
        context.register_artifact(output, logical_path=logical_path)


def test_source_relative_python_entrypoint_defaults_to_run(tmp_path: Path) -> None:
    experiment = tmp_path / "experiments" / "probe.py"
    experiment.parent.mkdir()
    source = "def run(context, *, value):\n    return (context.job_id, context.model, value)\n"
    experiment.write_text(source)
    context = make_context(tmp_path, model="loaded-model")

    result = invoke_experiment("experiments/probe.py", context, {"value": 7})
    resolved = resolve_entrypoint("experiments/probe.py:run", source_root=tmp_path)

    assert result == ("job-1", "loaded-model", 7)
    assert resolved.__module__.startswith("_topodrome_experiment_")


@pytest.mark.parametrize("reference", ["../outside.py", "nested/../../outside.py"])
def test_source_entrypoint_rejects_traversal(tmp_path: Path, reference: str) -> None:
    with pytest.raises(ValueError):
        resolve_entrypoint(reference, source_root=tmp_path)


def test_source_entrypoint_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_entrypoint(str(tmp_path.parent / "outside.py"), source_root=tmp_path)
