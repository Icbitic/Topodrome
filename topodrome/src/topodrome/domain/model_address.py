"""Path-free repository/checkpoint addresses shared by manifests and catalogues."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlsplit

_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


def validate_repo_id(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2 or any(
        _COMPONENT.fullmatch(part) is None
        or part.endswith((".", ".git"))
        or ".." in part
        or "--" in part
        for part in parts
    ):
        raise ValueError("model repo_id must be a namespaced repository: owner/repository")
    return value


def validate_revision(value: str) -> str:
    if (
        not value
        or len(value) > 200
        or value == "configured"
        or value != value.strip()
        or any(ord(char) < 33 or ord(char) == 127 for char in value)
        or any(char in value for char in "\\?#:")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("model revision must be an explicit, safe checkpoint identifier")
    return value


def validate_subfolder(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not value
        or len(value) > 500
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("model subfolder must be a canonical relative POSIX directory")
    return value


def normalize_model_address(value: Any) -> Any:
    """Accept a repository string or a credential-free HF model page URL.

    Slash-containing revision names are unambiguous in the mapping form, not
    in a ``/tree/<revision>/...`` page URL. No network access is needed here.
    """
    if isinstance(value, str):
        payload = {"repo_id": value}
    elif isinstance(value, dict):
        payload = dict(value)
    else:
        return value
    address = payload.get("repo_id")
    if not isinstance(address, str) or "://" not in address:
        return payload
    parsed = urlsplit(address)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"huggingface.co", "hf-mirror.com"}
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("model URL must be a credential-free HTTPS Hugging Face model page")
    path = unquote(parsed.path, errors="strict").removeprefix("/").removesuffix("/")
    parts = path.split("/")
    if len(parts) < 2 or (len(parts) > 2 and (len(parts) < 4 or parts[2] != "tree")):
        raise ValueError("expected a Hugging Face repository or /tree/revision model page")
    payload["repo_id"] = validate_repo_id("/".join(parts[:2]))
    if len(parts) > 2:
        fields: dict[str, str | None] = {"revision": validate_revision(parts[3])}
        if len(parts) > 4:
            fields["subfolder"] = validate_subfolder("/".join(parts[4:]))
        for field, selection in fields.items():
            if payload.get(field) not in {None, selection}:
                raise ValueError(f"model URL and explicit {field} disagree")
            payload[field] = selection
    return payload


def model_label(repo_id: str, revision: str | None, subfolder: str | None = None) -> str:
    label = repo_id if revision is None else f"{repo_id}@{revision}"
    return label if subfolder is None else f"{label}:{subfolder}"
