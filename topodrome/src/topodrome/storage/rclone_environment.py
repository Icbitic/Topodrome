"""Minimal environment passed to site-owned rclone subprocesses."""

from __future__ import annotations

import os
from collections.abc import Mapping

_SAFE_RUNTIME_KEYS = frozenset(
    {
        "ALL_PROXY",
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "NO_PROXY",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "XDG_CACHE_HOME",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


def rclone_process_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Keep runtime settings and explicitly rclone-scoped credentials only.

    Provider, site-enrollment, fencing, Hugging Face, and Jupyter credentials
    may coexist in the site-agent process.  None is needed by rclone.  Backends
    that require environment credentials must use rclone's ``RCLONE_*`` names
    (or the permission-restricted file named by ``RCLONE_CONFIG``).
    """

    values = os.environ if source is None else source
    environment = {
        key: value
        for key, value in values.items()
        if key in _SAFE_RUNTIME_KEYS or key.upper().startswith("RCLONE_")
    }
    environment.setdefault("PATH", os.defpath)
    return environment


__all__ = ["rclone_process_environment"]
