# Tokenizer vocabulary symbols below are not credentials.
# ruff: noqa: S106
from dataclasses import replace

import pytest

from tests.unit.test_runtime_ray_actor import _FakeRay
from topodrome.runtime import (
    ActorKey,
    DetachedRayActorAdapter,
    ModelKey,
    ModelRegistration,
    RayActorSpec,
    TransformersLoader,
)
from topodrome.runtime.actor import ActorState, ActorUnavailableError, LoadedModel, LocalModelActor


def key(revision="first"):
    return ActorKey(
        "test-site",
        "GPU-A",
        ModelKey("test/olmo3", revision, "transformers", "configured", dtype="float32"),
    )


@pytest.fixture
def tiny_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0")
        ),
        unk_token="t0",
        bos_token="t2",
        eos_token="t1",
        pad_token="t0",
    )
    config = transformers.Olmo3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
    )
    paths = []
    for index in range(2):
        torch.manual_seed(index + 7)
        model = transformers.Olmo3ForCausalLM(config)
        path = tmp_path / str(index)
        model.save_pretrained(path, max_shard_size="2KB")
        tokenizer.save_pretrained(path)
        paths.append(path)
    loaders = [
        TransformersLoader(ModelRegistration(key(str(i)).model, local_path=path), device="cpu")
        for i, path in enumerate(paths)
    ]
    return torch, transformers, paths, loaders


def test_real_olmo_sharded_safetensors_replace_in_place_and_match_fresh_logits(
    tiny_checkpoints, monkeypatch
):
    torch, transformers, _, loaders = tiny_checkpoints
    fresh = loaders[1](key("1").model)
    ids = torch.tensor([[2, 5, 7, 9]])
    with torch.no_grad():
        expected = fresh.model(ids).logits
    fresh.close()
    actor = LocalModelActor(key("0"), loaders[0])
    model = actor._loaded.model
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    # A stale model-dependent rotary buffer must not survive the switch.
    for name, buffer in model.named_buffers():
        if name.endswith("inv_freq"):
            buffer.zero_()
    model._cache = object()
    model.config._name_or_path = "old-checkpoint"
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_, **__: pytest.fail("compatible switch reconstructed the model"),
    )
    assert actor.replace_model(key("1"), loaders[1], expected=key("0")) == "weights"
    assert actor._loaded.model is model
    assert {name: value.data_ptr() for name, value in model.named_parameters()} == pointers
    assert not hasattr(model, "_cache")
    with torch.no_grad():
        actual = actor.run(
            lambda context: context.model(ids).logits.detach(), job_id="j", attempt_id="a"
        )
    torch.testing.assert_close(actual, expected)
    assert actor.key == key("1") and actor.describe().model_switches == 1
    assert actor._loaded.tokenizer.name_or_path == str(loaders[1].registration.local_path)
    assert model.name_or_path == str(loaders[1].registration.local_path)
    assert model.config._name_or_path == model.name_or_path
    assert model.config._commit_hash == "1"
    assert actor.replace_model(key("0"), loaders[0], expected=key("1")) == "weights"
    assert model.config._commit_hash == "0"
    actor.evict()


def test_configuration_change_reconstructs_in_same_worker(tiny_checkpoints):
    _, _, paths, loaders = tiny_checkpoints
    import json

    config = json.loads((paths[1] / "config.json").read_text())
    config["rope_theta"] = 20000
    (paths[1] / "config.json").write_text(json.dumps(config))
    actor = LocalModelActor(key("0"), loaders[0])
    original = actor._loaded.model
    identity = actor.actor_id
    assert actor.replace_model(key("1"), loaders[1], expected=key("0")) == "reload"
    assert actor.actor_id == identity and actor._loaded.model is not original
    assert actor._loaded.model.config.rope_theta == 20000
    actor.evict()


def test_untracked_hook_does_not_leak_into_next_checkpoint(tiny_checkpoints):
    _, _, _, loaders = tiny_checkpoints
    actor = LocalModelActor(key("0"), loaders[0])
    old = actor._loaded.model
    old.register_forward_hook(lambda *_: None)
    assert actor.replace_model(key("1"), loaders[1], expected=key("0")) == "reload"
    assert actor._loaded.model is not old
    assert not actor._loaded.model._forward_hooks
    actor.evict()


def test_different_architecture_rebuilds_model_without_replacing_worker(tiny_checkpoints, tmp_path):
    torch, transformers, paths, loaders = tiny_checkpoints
    target = tmp_path / "pythia"
    config = transformers.GPTNeoXConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        bos_token_id=2,
        eos_token_id=1,
    )
    model = transformers.GPTNeoXForCausalLM(config).eval()
    model.save_pretrained(target)
    transformers.AutoTokenizer.from_pretrained(paths[0]).save_pretrained(target)
    next_key = replace(key("pythia"), model=replace(key("pythia").model, repo_id="test/pythia"))
    loader = TransformersLoader(ModelRegistration(next_key.model, local_path=target), device="cpu")
    actor = LocalModelActor(key("0"), loaders[0])
    identity = actor.describe()
    assert actor.replace_model(next_key, loader, expected=key("0")) == "reload"
    assert isinstance(actor._loaded.model, transformers.GPTNeoXForCausalLM)
    assert actor.describe().actor_id == identity.actor_id
    assert actor.describe().worker_pid == identity.worker_pid
    tokens = torch.tensor([[2, 5, 7]])
    with torch.no_grad():
        torch.testing.assert_close(actor._loaded.model(tokens).logits, model(tokens).logits)
    actor.evict()


def test_ray_switch_requires_explicit_owned_actor_and_preserves_process(tiny_checkpoints):
    _, _, _, loaders = tiny_checkpoints
    ray = _FakeRay()
    adapter = DetachedRayActorAdapter(ray)
    first = adapter.get_or_create(RayActorSpec(key("0"), "0"), loaders[0])
    previous = first.describe()
    second = adapter.get_or_create(
        RayActorSpec(key("1"), "0"), loaders[1], expected_switch_actor_id=previous["actor_id"]
    )
    assert second.raw_handle is first.raw_handle
    assert second.describe()["actor_id"] == previous["actor_id"]
    assert second.describe()["last_switch_kind"] == "weights"
    assert not second.reused  # Startup accounting must not treat a switch as a warm hit.
    second.evict(force=True)


class SwitchLoader:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False

    def __call__(self, _):
        return LoadedModel(model={"weight": 1}, close=lambda: setattr(self, "closed", True))

    def replace(self, loaded, previous):
        loaded.model["weight"] = 2
        if self.fail:
            raise RuntimeError("partial copy")
        return True


def test_partial_switch_never_publishes_new_identity_and_disposes():
    old = SwitchLoader()
    actor = LocalModelActor(key(), old)
    with pytest.raises(RuntimeError, match="partial copy"):
        actor.replace_model(key("second"), SwitchLoader(True), expected=key())
    assert actor.key == key() and actor.state is ActorState.STOPPED
    assert old.closed
    with pytest.raises(ActorUnavailableError):
        actor.run(lambda context: 1, job_id="j", attempt_id="a")


def test_pending_run_and_changed_resource_identity_cannot_switch():
    actor = LocalModelActor(key(), SwitchLoader())
    actor._reserve()
    with pytest.raises(ActorUnavailableError):
        actor.replace_model(key("second"), SwitchLoader(), expected=key())
    actor._pending_runs = 0
    actor._state = ActorState.IDLE
    with pytest.raises(ActorUnavailableError):
        actor.replace_model(
            replace(key("second"), runtime_fingerprint="different"), SwitchLoader(), expected=key()
        )
    assert actor.state is ActorState.IDLE and actor.key == key()
    actor.evict()
