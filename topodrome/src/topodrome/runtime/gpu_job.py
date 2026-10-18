"""Minimal, process-private resources for a GPU job without a preloaded model."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from .actor import LoadedModel
from .model_key import ModelKey


@dataclass(frozen=True, slots=True)
class GPUJobLoader:
    """Construct devices inside the visibility-isolated worker, never the agent.

    No Transformers imports, model acquisition, weights, or dummy CUDA allocation.
    The process itself is disposed at attempt completion, not kept in a warm pool.
    """

    visible_gpu_count: int = 1

    def __post_init__(self) -> None:
        if self.visible_gpu_count < 1:
            raise ValueError("GPU jobs require at least one assigned GPU")

    def __call__(self, key: ModelKey | None) -> LoadedModel:
        if key is not None:
            raise ValueError("GPUJobLoader requires a null model identity")
        torch = importlib.import_module("torch")
        if torch.cuda.device_count() != self.visible_gpu_count:
            raise RuntimeError("worker CUDA visibility does not match its admitted GPU allocation")
        devices = tuple(torch.device("cuda", index) for index in range(self.visible_gpu_count))
        return LoadedModel(model=None, tokenizer=None, device=devices[0], devices=devices)
