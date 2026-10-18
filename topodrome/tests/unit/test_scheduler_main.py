from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from topodrome.scheduler import main as scheduler_main

pytestmark = pytest.mark.unit


def test_scheduler_cli_keeps_uvicorn_on_the_redacting_root_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        api_host="127.0.0.1",
        api_port=8042,
        api_keepalive_seconds=300,
        validate_api_bind_host=lambda host: host,
    )
    application = object()
    invocation: dict[str, Any] = {}

    monkeypatch.setattr(sys, "argv", ["topodrome-scheduler"])
    monkeypatch.setattr(scheduler_main, "SchedulerSettings", lambda: settings)
    monkeypatch.setattr(scheduler_main, "create_app", lambda actual: application)
    monkeypatch.setattr(scheduler_main, "configure_logging", lambda: None)
    monkeypatch.setattr(
        scheduler_main.uvicorn,
        "run",
        lambda actual, **options: invocation.update(app=actual, **options),
    )

    scheduler_main.main()

    assert invocation == {
        "app": application,
        "host": "127.0.0.1",
        "port": 8042,
        "timeout_keep_alive": 300,
        "log_config": None,
    }
