"""Enrollment identities for manually registered AutoDL instances."""

import hashlib
import hmac
import re


def valid_manual_site(site_id: str) -> bool:
    return re.fullmatch(r"autodl-manual-[a-f0-9]{24}", site_id) is not None


def manual_token(base_token: str, site_id: str) -> str:
    if not valid_manual_site(site_id):
        raise ValueError("invalid manual AutoDL site")
    return hmac.new(
        base_token.encode(), ("topodrome-manual:" + site_id).encode(), hashlib.sha256
    ).hexdigest()
