"""GuardianAI Location Agent package.

Provides location intelligence for GuardianAI: finds, routes to, and
ranks nearby emergency facilities. The agent is advisory only and must
never gate the emergency call path.
"""

from .agent import LocationAgent
from .config import LocationAgentConfig
from .models import (
    Facility,
    LocationResult,
    LocationValidationError,
    UserLocation,
)

__all__ = [
    "LocationAgent",
    "LocationAgentConfig",
    "Facility",
    "LocationResult",
    "LocationValidationError",
    "UserLocation",
]
