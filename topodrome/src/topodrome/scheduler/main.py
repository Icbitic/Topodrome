from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from topodrome.config import SchedulerSettings
from topodrome.logging import configure_logging
from topodrome.scheduler.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Topodrome central scheduler")
    parser.add_argument("--config", type=Path, default=os.environ.get("TOPODROME_CONFIG"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    arguments = parser.parse_args()
    settings = (
        SchedulerSettings.from_yaml(arguments.config) if arguments.config else SchedulerSettings()
    )
    configure_logging()
    bind_host = settings.validate_api_bind_host(arguments.host or settings.api_host)
    app = (
        create_app(settings, settings_path=arguments.config)
        if arguments.config is not None
        else create_app(settings)
    )
    uvicorn.run(
        app,
        host=bind_host,
        port=arguments.port or settings.api_port,
        # Sporadic small files must not reopen a TCP/SSH channel after the
        # server's usual five-second idle timeout. Proxy deployments must keep
        # their upstream idle timeout aligned with this setting.
        timeout_keep_alive=settings.api_keepalive_seconds,
        # Uvicorn's default logging configuration installs an unfiltered,
        # non-propagating access handler after ``configure_logging``.  Keep the
        # root redacting handler so short-lived kernel query credentials never
        # enter access logs.
        log_config=None,
    )


if __name__ == "__main__":
    main()
