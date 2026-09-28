from __future__ import annotations

import pytest

from topodrome.runtime_references import (
    validate_job_working_dir_reference,
    validate_pinned_runtime_requirements,
    validate_published_ray_working_dir,
)


def test_runtime_pip_requires_exact_version_and_distribution_digest() -> None:
    pinned = f"einops==0.8.0 --hash=sha256:{'a' * 64}"
    assert validate_pinned_runtime_requirements((pinned,)) == (pinned,)

    for invalid in (
        "einops",
        "einops==0.8.0",
        f"einops>=0.8.0 --hash=sha256:{'a' * 64}",
        f"einops==0.8.0 --hash=sha256:{'A' * 64}",
        "--extra-index-url=https://packages.invalid/simple",
        f"einops @ https://packages.invalid/einops.whl --hash=sha256:{'a' * 64}",
    ):
        with pytest.raises(ValueError, match="exact"):
            validate_pinned_runtime_requirements((invalid,))


def test_runtime_pip_rejects_duplicate_and_unbounded_input() -> None:
    pinned = f"einops==0.8.0 --hash=sha256:{'a' * 64}"
    with pytest.raises(ValueError, match="duplicates"):
        validate_pinned_runtime_requirements((pinned, pinned))
    with pytest.raises(ValueError, match="256-item"):
        validate_pinned_runtime_requirements(
            tuple(
                f"package-{index}==1 --hash=sha256:{index:064x}"
                for index in range(257)
            )
        )


def test_runtime_working_directories_are_content_addressed_at_each_boundary() -> None:
    job_reference = f"sha256:{'b' * 64}"
    ray_reference = f"gcs://_ray_pkg_{'c' * 16}.zip"
    assert validate_job_working_dir_reference(job_reference) == job_reference
    assert validate_published_ray_working_dir(ray_reference) == ray_reference
    assert validate_published_ray_working_dir("/srv/topodrome/source") == (
        "/srv/topodrome/source"
    )

    for invalid in (
        "https://public.invalid/source.zip",
        "s3://public-bucket/source.zip",
        "gcs://_ray_pkg_mutable-name.zip",
        "relative/source",
    ):
        with pytest.raises(ValueError):
            validate_published_ray_working_dir(invalid)
    with pytest.raises(ValueError, match="sha256"):
        validate_job_working_dir_reference("https://public.invalid/source.zip")
