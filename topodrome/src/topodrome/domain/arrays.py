"""Bounded, immutable array submissions; no executable parameter expansion."""

from typing import Any

from pydantic import Field, model_serializer, model_validator

from .specs import JobSpec, ModelRequirement, ResourceRequirements, StrictModel


class ArrayElement(StrictModel):
    index: int = Field(ge=0, le=2**31 - 1)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    model: ModelRequirement | None = None
    resources: ResourceRequirements | None = None

    @model_serializer(mode="wrap")
    def preserve_inheritance(self, handler):
        value = handler(self)
        # Omission inherits; explicit model:null requests a model-free job.
        # Preserve that distinction through both SDKs and idempotent replay.
        for name in ("model", "resources"):
            if name not in self.model_fields_set:
                value.pop(name, None)
        if self.resources is not None:
            value["resources"] = self.resources.model_dump(mode="json", exclude_unset=True)
        return value


class JobArraySpec(StrictModel):
    job: JobSpec
    elements: tuple[ArrayElement, ...] = Field(min_length=1, max_length=1000)
    max_parallel: int = Field(default=1, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_elements(self):
        if len({element.index for element in self.elements}) != len(self.elements):
            raise ValueError("array indices must be unique")
        if self.job.workload_kind.value == "interactive":
            raise ValueError("arrays support batch jobs, not interactive kernels")
        if len(self.model_dump_json().encode()) > 4 * 1024**2:
            raise ValueError("array parameter metadata exceeds 4 MiB")
        expanded_bytes = 0
        for element in self.elements:
            expanded_bytes += len(self.element_spec(element).model_dump_json().encode())
            if expanded_bytes > 32 * 1024**2:
                raise ValueError("expanded array metadata exceeds 32 MiB; use smaller arrays")
        return self

    def element_spec(self, element: ArrayElement) -> JobSpec:
        value = self.job.model_dump()
        if "model" in element.model_fields_set:
            value["model"] = element.model.model_dump() if element.model is not None else None
        if "resources" in element.model_fields_set:
            if element.resources is None:
                raise ValueError("element resources must be a resource mapping, not null")
            value["resources"].update(element.resources.model_dump(exclude_unset=True))
        value["kwargs"] = {**self.job.kwargs, **element.kwargs}
        # The normal entrypoint receives array_index as a keyword argument.
        # It is immutable through retry and works with existing site agents.
        if "array_index" in value["kwargs"]:
            raise ValueError("array_index is reserved for the scheduler in array jobs")
        value["kwargs"]["array_index"] = element.index
        return JobSpec.model_validate(value)


class ArrayUpdate(StrictModel):
    max_parallel: int | None = Field(default=None, ge=1, le=1000)
    held: bool | None = None

    @model_validator(mode="after")
    def nonempty(self):
        if self.max_parallel is None and self.held is None:
            raise ValueError("provide max_parallel or held")
        return self
