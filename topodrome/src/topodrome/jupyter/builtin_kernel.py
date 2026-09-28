"""Marker entrypoint for content-addressed interactive assignments.

The site agent intercepts interactive assignments and hosts their model actor
through the outbound relay. Executing this file as an ordinary batch source is
always an orchestration error and fails closed.
"""

from __future__ import annotations

from typing import Any


def run(context: Any, **kwargs: Any) -> None:
    del context, kwargs
    raise RuntimeError("the built-in kernel entrypoint requires the interactive site runtime")
