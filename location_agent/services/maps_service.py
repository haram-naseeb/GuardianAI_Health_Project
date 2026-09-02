"""Google Maps implementation of the maps service.

Real geographic data only comes from Google Maps Platform:
- Places API (Nearby Search) finds emergency facilities.
- Directions API provides road distance and estimated travel time.

No LLM is involved in producing facility or route data. The `googlemaps`
library is imported lazily so unit tests can run without it installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service errors
# ---------------------------------------------------------------------------

class MapsServiceError(Exception):
    """Base error for map provider failures."""


class MapsTimeoutError(MapsServiceError):
    """The provider request timed out."""


class MapsQuotaError(MapsServiceError):
    """The provider rejected the request due to quota/billing limits."""


class MapsRequestDeniedError(MapsServiceError):
    """The provider denied the request (e.g. missing key or API)."""


class MapsMalformedResponseError(MapsServiceError):
    """The provider returned a response that could not be interpreted."""


# ---------------------------------------------------------------------------
# Provider-neutral intermediate data
# ---------------------------------------------------------------------------

@dataclass
class RawFacility:
    """Facility data exactly as reported by the maps provider."""

    name: str
    latitude: float
    longitude: float
    facility_type: str
    address: str | None = None
    place_id: str | None = None
    rating: float | None = None
    source: str = "google_maps"


@dataclass
class RouteInfo:
    """Route between the user location and a facility."""

    distance_m: float
    duration_s: float
    status: str = "ok"


class MapsServiceBase:
    """Interface implemented by both Google Maps and mock services."""

    source: str = "unknown"

    def search_nearby_facilities(
        self,
        latitude: float,
        longitude: float,
        radius_m: int = 10_000,
        facility_types: Sequence[str] = ("hospital",),
    ) -> list[RawFacility]:
        raise NotImplementedError

    def get_route(
        self,
        origin_latitude: float,
        origin_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
    ) -> RouteInfo | None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Google Maps implementation
# ---------------------------------------------------------------------------

class GoogleMapsService(MapsServiceBase):
    """Talks to Google Maps Platform using the official client library."""

    source = "google_maps"

    def __init__(self, api_key: str, timeout_s: int = 10):
        if not api_key:
            raise MapsRequestDeniedError(
                "Google Maps API key is missing. Set GOOGLE_MAPS_API_KEY "
                "or enable LOCATION_AGENT_MOCK for development."
            )

        # Lazy import keeps the package importable without googlemaps.
        import googlemaps
        from googlemaps import exceptions as gmaps_exceptions

        self._exceptions = gmaps_exceptions
        self._client = googlemaps.Client(key=api_key, timeout=timeout_s)

    def search_nearby_facilities(
        self,
        latitude: float,
        longitude: float,
        radius_m: int = 10_000,
        facility_types: Sequence[str] = ("hospital",),
    ) -> list[RawFacility]:
        facilities: list[RawFacility] = []
        seen_place_ids: set[str] = set()

        for facility_type in facility_types:
            response = self._nearby_search(
                latitude, longitude, radius_m, facility_type
            )

            for place in response.get("results", []):
                facility = parse_nearby_place(place, facility_type)
                if facility is None:
                    logger.debug("Skipping incomplete place entry.")
                    continue
                if facility.place_id and facility.place_id in seen_place_ids:
                    continue
                if facility.place_id:
                    seen_place_ids.add(facility.place_id)
                facilities.append(facility)

        return facilities

    def get_route(
        self,
        origin_latitude: float,
        origin_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
    ) -> RouteInfo | None:
        try:
            routes = self._client.directions(
                origin=(origin_latitude, origin_longitude),
                destination=(destination_latitude, destination_longitude),
                mode="driving",
            )
        except self._exceptions.Timeout as error:
            raise MapsTimeoutError(
                f"Route calculation timed out: {error}"
            ) from error
        except self._exceptions.ApiError as error:
            raise self._convert_api_error(error) from error
        except Exception as error:  # network failure, bad payload, etc.
            raise MapsServiceError(
                f"Route calculation failed: {error}"
            ) from error

        return parse_directions(routes)

    # -- internal helpers ---------------------------------------------------

    def _nearby_search(
        self,
        latitude: float,
        longitude: float,
        radius_m: int,
        facility_type: str,
    ) -> dict[str, Any]:
        try:
            response = self._client.places_nearby(
                location=(latitude, longitude),
                radius=radius_m,
                type=facility_type,
            )
        except self._exceptions.Timeout as error:
            raise MapsTimeoutError(
                f"Nearby facility search timed out: {error}"
            ) from error
        except self._exceptions.ApiError as error:
            raise self._convert_api_error(error) from error
        except Exception as error:
            raise MapsServiceError(
                f"Nearby facility search failed: {error}"
            ) from error

        if not isinstance(response, dict):
            raise MapsMalformedResponseError(
                "Nearby search returned an unexpected response shape."
            )
        return response

    def _convert_api_error(self, error: Exception) -> MapsServiceError:
        status = str(getattr(error, "status", "") or error).upper()
        if "OVER_QUERY_LIMIT" in status:
            return MapsQuotaError(f"Google Maps quota exceeded: {error}")
        if "REQUEST_DENIED" in status:
            return MapsRequestDeniedError(
                f"Google Maps request denied: {error}"
            )
        return MapsServiceError(f"Google Maps API error: {error}")


# ---------------------------------------------------------------------------
# Response parsers (pure functions, unit-testable without any network)
# ---------------------------------------------------------------------------

def parse_nearby_place(
    place: Any,
    facility_type: str,
) -> RawFacility | None:
    """Convert one Nearby Search result into a RawFacility.

    Returns None when required fields are missing instead of guessing
    values that Google did not provide.
    """
    if not isinstance(place, dict):
        return None

    geometry = place.get("geometry") or {}
    location = geometry.get("location") or {}
    latitude = location.get("lat")
    longitude = location.get("lng")
    name = place.get("name")

    if name is None or latitude is None or longitude is None:
        return None

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return None

    rating = place.get("rating")
    if rating is not None:
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            rating = None

    return RawFacility(
        name=str(name),
        latitude=latitude,
        longitude=longitude,
        facility_type=facility_type,
        address=place.get("vicinity") or place.get("formatted_address"),
        place_id=place.get("place_id"),
        rating=rating,
    )


def parse_directions(routes: Any) -> RouteInfo | None:
    """Extract distance and duration from a Directions API response.

    Returns None when no usable route exists (e.g. ZERO_RESULTS).
    Raises MapsMalformedResponseError when the response shape is invalid.
    """
    if not routes:
        return None

    try:
        leg = routes[0]["legs"][0]
        distance_m = float(leg["distance"]["value"])
        duration_s = float(leg["duration"]["value"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise MapsMalformedResponseError(
            "Directions response is missing route/leg distance data."
        ) from error

    return RouteInfo(
        distance_m=distance_m,
        duration_s=duration_s,
        status="ok",
    )
