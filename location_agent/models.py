"""Data models and input validation for the Location Agent.

Design rule: never invent data. Any field the maps provider does not
supply stays None, and invalid input is rejected with a clear error.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Error codes used in structured error responses.
# ---------------------------------------------------------------------------
ERROR_MISSING_COORDINATES = "MISSING_COORDINATES"
ERROR_INVALID_COORDINATES = "INVALID_COORDINATES"
ERROR_SEARCH_FAILED = "FACILITY_SEARCH_FAILED"
ERROR_NO_FACILITIES = "NO_FACILITIES_FOUND"
ERROR_API_QUOTA = "API_QUOTA_EXCEEDED"
ERROR_TIMEOUT = "REQUEST_TIMEOUT"
ERROR_MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
ERROR_UNKNOWN = "UNKNOWN_ERROR"

SOURCE_GOOGLE_MAPS = "google_maps"
SOURCE_MOCK = "mock"
SOURCE_STATIC_PK = "static_pk"


class LocationValidationError(ValueError):
    """Raised when the input location is missing or invalid."""


def validate_coordinates(latitude: Any, longitude: Any) -> tuple[float, float]:
    """Validate raw coordinates and return them as floats.

    Raises LocationValidationError when coordinates are missing, not
    numeric, not finite, or outside their valid ranges.
    """
    if latitude is None or longitude is None:
        raise LocationValidationError(
            "Missing coordinates: both latitude and longitude are required."
        )

    if isinstance(latitude, bool) or isinstance(longitude, bool):
        raise LocationValidationError(
            "Coordinates must be numbers, not booleans."
        )

    try:
        lat = float(latitude)
        lng = float(longitude)
    except (TypeError, ValueError) as error:
        raise LocationValidationError(
            f"Coordinates must be numeric, got latitude={latitude!r}, "
            f"longitude={longitude!r}."
        ) from error

    if not (math.isfinite(lat) and math.isfinite(lng)):
        raise LocationValidationError("Coordinates must be finite numbers.")

    if not -90.0 <= lat <= 90.0:
        raise LocationValidationError(
            f"Latitude {lat} is outside the valid range [-90, 90]."
        )

    if not -180.0 <= lng <= 180.0:
        raise LocationValidationError(
            f"Longitude {lng} is outside the valid range [-180, 180]."
        )

    return lat, lng


@dataclass
class UserLocation:
    """The user's current location, exactly as provided by the caller."""

    latitude: float
    longitude: float
    accuracy_m: float | None = None
    timestamp: str | None = None

    @classmethod
    def create(
        cls,
        latitude: Any,
        longitude: Any,
        accuracy_m: Any = None,
        timestamp: str | None = None,
    ) -> "UserLocation":
        lat, lng = validate_coordinates(latitude, longitude)

        if accuracy_m is not None:
            try:
                accuracy_m = float(accuracy_m)
            except (TypeError, ValueError) as error:
                raise LocationValidationError(
                    f"accuracy_m must be numeric, got {accuracy_m!r}."
                ) from error
            if not math.isfinite(accuracy_m) or accuracy_m <= 0:
                raise LocationValidationError(
                    "accuracy_m must be a positive number when provided."
                )

        return cls(
            latitude=lat,
            longitude=lng,
            accuracy_m=accuracy_m,
            timestamp=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "latitude": self.latitude,
            "longitude": self.longitude,
        }
        if self.accuracy_m is not None:
            result["accuracy_m"] = self.accuracy_m
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        return result


@dataclass
class Facility:
    """One emergency facility candidate with route information."""

    name: str
    latitude: float
    longitude: float
    facility_type: str
    address: str | None = None
    place_id: str | None = None
    rating: float | None = None
    source: str = SOURCE_GOOGLE_MAPS
    distance_km: float | None = None
    travel_time_minutes: float | None = None
    route_status: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocationResult:
    """Structured output consumed by the Coordinator.

    The schema is intentionally stable: success responses carry the
    recommendation plus alternatives; error responses carry an error
    code and always set fallback_available=True because emergency
    calling never depends on this agent.
    """

    status: str  # "success" | "error"
    current_location: UserLocation | None = None
    recommended_facility: Facility | None = None
    alternatives: list[Facility] = field(default_factory=list)
    searched_at: str | None = None
    source: str = SOURCE_GOOGLE_MAPS
    error_code: str | None = None
    message: str | None = None
    fallback_available: bool = True

    @classmethod
    def success(
        cls,
        current_location: UserLocation,
        recommended: Facility,
        alternatives: list[Facility],
        source: str,
    ) -> "LocationResult":
        return cls(
            status="success",
            current_location=current_location,
            recommended_facility=recommended,
            alternatives=alternatives,
            searched_at=datetime.now(timezone.utc).isoformat(),
            source=source,
        )

    @classmethod
    def failure(cls, error_code: str, message: str) -> "LocationResult":
        return cls(
            status="error",
            searched_at=datetime.now(timezone.utc).isoformat(),
            error_code=error_code,
            message=message,
            fallback_available=True,
        )

    def to_dict(self) -> dict[str, Any]:
        if self.status == "success":
            return {
                "status": self.status,
                "current_location": (
                    self.current_location.to_dict()
                    if self.current_location
                    else None
                ),
                "recommended_facility": (
                    self.recommended_facility.to_dict()
                    if self.recommended_facility
                    else None
                ),
                "alternatives": [f.to_dict() for f in self.alternatives],
                "searched_at": self.searched_at,
                "source": self.source,
            }
        return {
            "status": self.status,
            "error_code": self.error_code,
            "message": self.message,
            "fallback_available": self.fallback_available,
        }
