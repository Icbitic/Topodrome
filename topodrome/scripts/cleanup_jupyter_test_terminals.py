"""Remove only Jupyter terminals recorded as created by this test harness."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scripts.integration_support import ManagedJupyterClient
from topodrome.config import load_secret_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--secret-file",
        type=Path,
        default=Path("/home/xiangkun/.config/topodrome/secrets.env"),
    )
    return parser


async def _main() -> int:
    args = _parser().parse_args()
    secrets = load_secret_bundle(args.secret_file)
    if secrets.test_jupyter_url is None or secrets.test_jupyter_token is None:
        raise RuntimeError("protected secret file lacks the Jupyter test credentials")
    async with ManagedJupyterClient(
        secrets.test_jupyter_url,
        secrets.test_jupyter_token,
    ) as client:
        removed = await client.cleanup_registered_terminals()
    print(f"removed {len(removed)} registered Topodrome integration terminal(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
