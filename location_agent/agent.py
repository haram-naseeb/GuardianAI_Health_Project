"""GuardianAI Location Agent.

Finds nearby emergency facilities, calculates road distance and
estimated travel time for each candidate, ranks them deterministically,
and returns a structured LocationResult for the Coordinator.

Safety rules:
- Never invent facilities, addresses, coordinates, or travel times.
- Every failure path returns a structured error result; the agent never
  raises into the Coordinator.
- This agent enhances emergency coordination. It never gates or delays
  the emergency call path.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import LocationAgentConfig
from .geo import haversine_km
from .models import (
    ERROR_API_QUOTA,
    ERROR_INVALID_COORDINATES,
    ERROR_MALFORMED_RESPONSE,
    ERROR_MISSING_COORDINATES,
    ERROR_NO_FACILITIES,
    ERROR_SEARCH_FAILED,
    ERROR_TIMEOUT,
    ERROR_UNKNOWN,
    SOURCE_GOOGLE_MAPS,
    SOURCE_MOCK,
    SOURCE_STATIC_PK,
    Facility,
    LocationResult,
    LocationValidationError,
    UserLocation,
)
from .ranking import explain_recommendation, rank_facilities
from .services.maps_service import (
    GoogleMapsService,
    MapsMalformedResponseError,
    MapsQuotaError,
    MapsRequestDeniedError,
    MapsServiceBase,
    MapsServiceError,
    MapsTimeoutError,
    RawFacility,
    RouteInfo,
)
from .services.mock_maps_service import MockMapsService
from .services.static_hospital_service import StaticHospitalService

logger = logging.getLogger(__name__)

# Facility types searched in order. Extend to support more types later.
FACILITY_TYPES = ("hospital",)


class LocationAgent:
    """Location intelligence agent for GuardianAI.

    Parameters
    ----------
    config:
        Optional LocationAgentConfig. Defaults to environment variables.
    maps_service:
        Optional injected maps service (used by tests). When omitted,
        mock mode or the live Google Maps service is selected from config.
    """

    def __init__(
        self,
        config: LocationAgentConfig | None = None,
        maps_service: MapsServiceBase | None = None,
    ):
        self.config = config or LocationAgentConfig.from_env()

        if maps_service is not None:
            self._service = maps_service
            self._source = getattr(
                maps_service, "source", SOURCE_GOOGLE_MAPS
            )
        elif self.config.static_mode:
            self._service = StaticHospitalService()
            self._source = SOURCE_STATIC_PK
        elif self.config.mock_mode:
            self._service = MockMapsService()
            self._source = SOURCE_MOCK
        else:
            if not self.config.google_maps_api_key:
                raise ValueError(
                    "GOOGLE_MAPS_API_KEY is not set. Set it in the "
                    "environment or enable LOCATION_AGENT_MOCK=true."
                )
            self._service = GoogleMapsService(
                api_key=self.config.google_maps_api_key,
                timeout_s=self.config.request_timeout_s,
            )
            self._source = SOURCE_GOOGLE_MAPS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_best_facility(
        self,
        latitude: Any = None,
        longitude: Any = None,
        accuracy_m: Any = None,
        timestamp: str | None = None,
    ) -> LocationResult:
        """Run the full search -> route -> rank pipeline.

        Always returns a LocationResult; never raises.
        """
        try:
            location = UserLocation.create(
                latitude, longitude, accuracy_m, timestamp
            )
        except LocationValidationError as error:
            code = (
                ERROR_MISSING_COORDINATES
                if latitude is None or longitude is None
                else ERROR_INVALID_COORDINATES
            )
            logger.warning("Location input rejected: %s", error)
            return LocationResult.failure(code, str(error))

        # Coordinates are only logged at reduced precision, if at all.
        logger.debug(
            "Searching facilities near (%.2f, %.2f)",
            location.latitude,
            location.longitude,
        )

        try:
            raw_facilities = self._service.search_nearby_facilities(
                location.latitude,
                location.longitude,
                radius_m=self.config.search_radius_m,
                facility_types=FACILITY_TYPES,
            )
        except MapsTimeoutError as error:
            return self._fail(ERROR_TIMEOUT, str(error))
        except MapsQuotaError as error:
            return self._fail(ERROR_API_QUOTA, str(error))
        except MapsRequestDeniedError as error:
            return self._fail(ERROR_SEARCH_FAILED, str(error))
        except MapsMalformedResponseError as error:
            return self._fail(ERROR_MALFORMED_RESPONSE, str(error))
        except MapsServiceError as error:
            return self._fail(ERROR_SEARCH_FAILED, str(error))
        except Exception as error:
            logger.exception("Unexpected error during facility search")
            return self._fail(ERROR_UNKNOWN, f"Unexpected error: {error}")

        if not raw_facilities:
            return self._fail(
                ERROR_NO_FACILITIES,
                "No emergency facilities were found near the provided "
                "location.",
            )

        candidates = self._build_candidates(location, raw_facilities)
        ranked = rank_facilities(candidates)

        recommended = ranked[0]
        recommended.reason = explain_recommendation(
            recommended, candidate_count=len(ranked)
        )

        return LocationResult.success(
            current_location=location,
            recommended=recommended,
            alternatives=ranked[1 : self.config.max_facilities],
            source=self._source,
        )

    def handle(self, state: dict[str, Any]) -> dict[str, Any]:
        """Dict-in / dict-out interface for LangGraph nodes.

        Reads latitude, longitude, accuracy_m and timestamp from the
        shared state and returns the serialized LocationResult.
        """
        result = self.find_best_facility(
            latitude=state.get("latitude"),
            longitude=state.get("longitude"),
            accuracy_m=state.get("accuracy_m"),
            timestamp=state.get("timestamp"),
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        location: UserLocation,
        raw_facilities: list[RawFacility],
    ) -> list[Facility]:
        facilities: list[Facility] = []

        for raw in raw_facilities[: self.config.max_facilities]:
            facility = Facility(
                name=raw.name,
                latitude=raw.latitude,
                longitude=raw.longitude,
                facility_type=raw.facility_type,
                address=raw.address,
                place_id=raw.place_id,
                rating=raw.rating,
                source=raw.source,
                distance_km=round(
                    haversine_km(
                        location.latitude,
                        location.longitude,
                        raw.latitude,
                        raw.longitude,
                    ),
                    2,
                ),
                route_status="unavailable",
            )

            route = self._safe_route(location, raw)
            if route is not None:
                facility.distance_km = round(route.distance_m / 1000.0, 2)
                facility.travel_time_minutes = round(
                    route.duration_s / 60.0, 1
                )
                facility.route_status = route.status

            facilities.append(facility)

        return facilities

    def _safe_route(
        self,
        location: UserLocation,
        raw: RawFacility,
    ) -> RouteInfo | None:
        """Fetch one route; per-facility routing errors are not fatal."""
        try:
            return self._service.get_route(
                location.latitude,
                location.longitude,
                raw.latitude,
                raw.longitude,
            )
        except MapsServiceError as error:
            logger.warning(
                "Route unavailable for facility %r: %s", raw.name, error
            )
            return None
        except Exception as error:
            logger.warning(
                "Unexpected routing error for facility %r: %s",
                raw.name,
                error,
            )
            return None

    def _fail(self, error_code: str, message: str) -> LocationResult:
        # The message is logged without precise user coordinates.
        logger.warning("Location Agent error [%s]: %s", error_code, message)
        return LocationResult.failure(error_code, message)
