"""Deterministic mock maps service for development and tests.

Enable with LOCATION_AGENT_MOCK=true. Every facility and route is
clearly marked as test data (source="mock") so mock output can never
be mistaken for a production Google Maps response.
"""

from __future__ import annotations

from typing import Sequence

from ..geo import haversine_km
from .maps_service import MapsServiceBase, RawFacility, RouteInfo

# Offsets are relative to the user location (degrees).
# route_factor simulates traffic: a higher factor means a slower road
# route, so estimated travel time does not simply follow distance.
_MOCK_FACILITIES = [
    {
        "name": "Mock General Hospital (MOCK TEST DATA)",
        "dlat": 0.010,
        "dlng": 0.002,
        "route_factor": 2.6,
        "rating": 4.1,
    },
    {
        "name": "Mock Emergency Medical Center (MOCK TEST DATA)",
        "dlat": 0.022,
        "dlng": -0.004,
        "route_factor": 1.1,
        "rating": 4.6,
    },
    {
        "name": "Mock City Hospital (MOCK TEST DATA)",
        "dlat": 0.045,
        "dlng": 0.010,
        "route_factor": 1.3,
        "rating": None,
    },
]

_AVERAGE_SPEED_KMH = 30.0
_DEFAULT_ROUTE_FACTOR = 1.3


class MockMapsService(MapsServiceBase):
    """Offline stand-in for GoogleMapsService.

    Parameters
    ----------
    fail_with:
        Optional exception instance to raise on every call. Used by
        unit tests to simulate provider failures.
    """

    source = "mock"

    def __init__(self, fail_with: Exception | None = None):
        self._fail_with = fail_with
        # Maps destination coordinates to their simulated route factor.
        self._route_factors: dict[tuple[float, float], float] = {}

    def _maybe_fail(self) -> None:
        if self._fail_with is not None:
            raise self._fail_with

    def search_nearby_facilities(
        self,
        latitude: float,
        longitude: float,
        radius_m: int = 10_000,
        facility_types: Sequence[str] = ("hospital",),
    ) -> list[RawFacility]:
        self._maybe_fail()

        facilities: list[RawFacility] = []
        for index, spec in enumerate(_MOCK_FACILITIES):
            facility_lat = latitude + spec["dlat"]
            facility_lng = longitude + spec["dlng"]

            self._route_factors[
                (round(facility_lat, 6), round(facility_lng, 6))
            ] = spec["route_factor"]

            facilities.append(
                RawFacility(
                    name=spec["name"],
                    latitude=facility_lat,
                    longitude=facility_lng,
                    facility_type="hospital",
                    address=f"Mock Street {index + 1} (test data only)",
                    place_id=f"mock-place-{index + 1}",
                    rating=spec["rating"],
                    source=self.source,
                )
            )
        return facilities

    def get_route(
        self,
        origin_latitude: float,
        origin_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
    ) -> RouteInfo | None:
        self._maybe_fail()

        straight_km = haversine_km(
            origin_latitude,
            origin_longitude,
            destination_latitude,
            destination_longitude,
        )

        factor = self._route_factors.get(
            (round(destination_latitude, 6), round(destination_longitude, 6)),
            _DEFAULT_ROUTE_FACTOR,
        )

        road_km = straight_km * factor
        duration_s = (road_km / _AVERAGE_SPEED_KMH) * 3600.0

        return RouteInfo(
            distance_m=road_km * 1000.0,
            duration_s=duration_s,
            status="ok",
        )
