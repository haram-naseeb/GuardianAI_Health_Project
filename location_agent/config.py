"""Configuration for the GuardianAI Location Agent.

Everything is driven by environment variables so no secret is ever
hardcoded. The Google Maps API key is never written to logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; plain environment variables still work.
    pass

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class LocationAgentConfig:
    """Runtime settings for the Location Agent."""

    google_maps_api_key: str | None = None
    mock_mode: bool = False
    static_mode: bool = False
    search_radius_m: int = 10_000
    max_facilities: int = 10
    request_timeout_s: int = 10

    @classmethod
    def from_env(cls) -> "LocationAgentConfig":
        return cls(
            google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY") or None,
            mock_mode=_env_bool("LOCATION_AGENT_MOCK", default=False),
            static_mode=_env_bool("LOCATION_AGENT_STATIC", default=False),
            search_radius_m=_env_int("LOCATION_SEARCH_RADIUS_M", 10_000),
            max_facilities=_env_int("LOCATION_MAX_FACILITIES", 10),
            request_timeout_s=_env_int("LOCATION_REQUEST_TIMEOUT_S", 10),
        )
