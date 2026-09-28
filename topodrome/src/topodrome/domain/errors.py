from __future__ import annotations


class TopodromeError(Exception):
    """Base class for errors safe to expose through the public API."""

    code = "topodrome_error"


class InvalidTransitionError(TopodromeError):
    code = "invalid_transition"


class NotFoundError(TopodromeError):
    code = "not_found"


class ConflictError(TopodromeError):
    code = "conflict"


class AdmissionRejectedError(TopodromeError):
    code = "admission_rejected"


class StaleFenceError(TopodromeError):
    code = "stale_fence"


class DurabilityError(TopodromeError):
    code = "durability_error"


class ProviderError(TopodromeError):
    code = "provider_error"


class ProviderCapacityError(ProviderError):
    code = "provider_capacity_unavailable"


class DatabaseUnavailableError(TopodromeError):
    code = "database_unavailable"
