"""Recover upload-only authority, never resurrect GPU ownership or terminal jobs."""

import hashlib
from datetime import timedelta

from topodrome.domain.enums import AttemptState, JobState, LeaseState
from topodrome.domain.specs import FenceClaim, JobSpec, utc_now
from topodrome.scheduler.input_staging import aware


async def expired_upload_message(session, repository, authority, site_id, payload):
    """ACK obsolete upload-only messages so they cannot poison the durable spool.

    This is a no-op, not permission to perform I/O or change an attempt state.
    The sender must obtain a fresh authenticated upload grant separately.
    """
    attempt_id = str(payload["attempt_id"])
    token = str(payload["fencing_token"])
    signed = authority.authenticate(token, attempt_id=attempt_id, site_id=site_id)
    attempt = await repository._get_attempt(session, attempt_id, lock=True)
    lease = attempt.lease
    if attempt.site_id != site_id:
        raise ValueError("upload message site mismatch")
    if (
        attempt.state is not AttemptState.COMMITTING
        or lease is None
        or lease.state is not LeaseState.RELEASING
        or any(c.active for c in lease.claims)
    ):
        return False
    if {(c.gpu_uuid, c.epoch) for c in signed.claims} != {
        (c.gpu_uuid, c.epoch) for c in lease.claims
    }:
        raise ValueError("upload message claim mismatch")
    return (
        signed.expires_at <= utc_now()
        or aware(lease.expires_at) <= utc_now()
        or attempt.fencing_token_digest != authority.digest(token)
    )


async def recover(
    session, repository, authority, site_id, payload, settings, receipt, *, replayed=False
):
    attempt = await repository._get_attempt(session, str(payload["attempt_id"]), lock=True)
    lease = attempt.lease
    now = utc_now()
    if attempt.site_id != site_id or attempt.job_id != payload.get("job_id"):
        raise ValueError("finalization identity mismatch")
    if (
        attempt.state is not AttemptState.COMMITTING
        or lease is None
        or lease.state is not LeaseState.RELEASING
        or any(c.active for c in lease.claims)
        or attempt.job.state is not JobState.RUNNING
        or aware(lease.expires_at) + timedelta(seconds=settings.finalization_recovery_grace_seconds)
        <= now
    ):
        return {"accepted": False, "state": attempt.state.value}
    claims = tuple(FenceClaim(gpu_uuid=c.gpu_uuid, epoch=c.epoch) for c in lease.claims)
    token, token_payload = authority.issue(
        attempt_id=attempt.id,
        site_id=site_id,
        claims=claims,
        ttl=timedelta(seconds=settings.fencing_token_ttl_seconds),
        now=aware(receipt.acknowledged_at),
        token_id=hashlib.sha256(f"finalization:{receipt.message_id}".encode()).hexdigest(),
    )
    if token_payload.expires_at <= now or (
        replayed and attempt.fencing_token_digest != authority.digest(token)
    ):
        return {"accepted": False, "retry": True}
    if not replayed:
        lease.expires_at = now + timedelta(seconds=settings.lease_ttl_seconds)
        lease.fencing_token_digest = attempt.fencing_token_digest = authority.digest(token)
        lease.updated_at = attempt.updated_at = now
    return {
        "accepted": True,
        "assignment": {
            "job_id": attempt.job_id,
            "attempt_id": attempt.id,
            "lease_id": lease.id,
            "spec": JobSpec.model_validate(attempt.job.spec).model_dump(
                mode="json", exclude={"name", "description", "estimated_runtime_seconds",
                                      "artifact_retention_seconds"}
            ),
            "gpu_uuids": [c.gpu_uuid for c in claims],
            "fence_epochs": {c.gpu_uuid: c.epoch for c in claims},
            "claims": [c.model_dump(mode="json") for c in claims],
            "fencing_token": token,
            "expires_at": lease.expires_at.isoformat(),
            "lease_ttl_seconds": settings.lease_ttl_seconds,
            "checkpoint_ref": attempt.checkpoint_ref,
        },
    }
