"""Optional Logfire setup shared by all agent scripts."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_logfire() -> bool:
    """
    Configure Logfire when enabled via environment variables.

    Environment variables:
    - ENABLE_LOGFIRE / LOGFIRE_ENABLED: on/off switch (default: off)
    - LOGFIRE_TOKEN: optional write token
    """

    load_dotenv(override=True)

    enabled = _env_flag("ENABLE_LOGFIRE") or _env_flag("LOGFIRE_ENABLED")
    if not enabled:
        return False

    try:
        import logfire  # type: ignore
    except ImportError:
        print("[logfire] ENABLE_LOGFIRE is on, but package `logfire` is not installed.")
        return False

    token = os.getenv("LOGFIRE_TOKEN")
    kwargs = {"token": token} if token else {}

    try:
        logfire.configure(**kwargs)
        logfire.instrument_openai()
    except Exception as exc:
        print(f"[logfire] Failed to initialize: {exc}")
        return False

    return True
