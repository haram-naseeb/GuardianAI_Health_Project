"""Tests for StaticHospitalService.

No network access or API keys required.
"""

import pytest

from location_agent.services.static_hospital_service import StaticHospitalService
from location_agent.services.maps_service import RawFacility, RouteInfo


LAHORE_LAT = 31.5204
LAHORE_LNG = 74.3587

KARACHI_LAT = 24.8607
KARACHI_LNG = 67.0011


@pytest.fixture
def service():
    return StaticHospitalService()


class TestDistanceSorting:
    def test_results_sorted_by_haversine_distance(self, service):
        results = service.search_nearby_facilities(
            LAHORE_LAT, LAHORE_LNG, radius_m=1_000_000
        )

        assert len(results) > 1
        assert all(isinstance(f, RawFacility) for f in results)

        from location_agent.geo import haversine_km

        distances = [
            haversine_km(LAHORE_LAT, LAHORE_LNG, f.latitude, f.longitude)
            for f in results
        ]
        assert distances == sorted(distances)


class TestLahoreArea:
    def test_lahore_coordinates_return_lahore_hospitals_first(self, service):
        results = service.search_nearby_facilities(
            LAHORE_LAT, LAHORE_LNG, radius_m=50_000
        )

        assert len(results) >= 3
        top_names = [f.name for f in results[:3]]
        assert any("Lahore" in name or "Services Hospital" in name or "Mayo" in name for name in top_names)

        for facility in results[:3]:
            assert facility.latitude > 31.0
            assert facility.longitude > 73.0

    def test_all_results_have_static_pk_source(self, service):
        results = service.search_nearby_facilities(
            LAHORE_LAT, LAHORE_LNG, radius_m=50_000
        )
        assert all(f.source == "static_pk" for f in results)


class TestKarachiArea:
    def test_karachi_coordinates_return_karachi_hospitals_first(self, service):
        results = service.search_nearby_facilities(
            KARACHI_LAT, KARACHI_LNG, radius_m=50_000
        )

        assert len(results) >= 2
        top_names = [f.name for f in results[:2]]
        assert any("Civil Hospital" in name or "Memon" in name for name in top_names)

        for facility in results[:2]:
            assert facility.latitude < 25.5
            assert facility.longitude < 68.0


class TestRouteEstimation:
    def test_get_route_returns_estimated_status(self, service):
        route = service.get_route(
            LAHORE_LAT, LAHORE_LNG, 31.4843, 74.2969
        )

        assert route is not None
        assert isinstance(route, RouteInfo)
        assert route.status == "estimated"
        assert route.distance_m > 0
        assert route.duration_s > 0

    def test_get_route_distance_exceeds_straight_line(self, service):
        route = service.get_route(
            LAHORE_LAT, LAHORE_LNG, 31.4843, 74.2969
        )

        from location_agent.geo import haversine_km

        straight_m = haversine_km(LAHORE_LAT, LAHORE_LNG, 31.4843, 74.2969) * 1000
        assert route.distance_m > straight_m


class TestRadiusFallback:
    def test_no_hospitals_in_tiny_radius_returns_nearest_three(self, service):
        results = service.search_nearby_facilities(
            LAHORE_LAT, LAHORE_LNG, radius_m=10
        )

        assert len(results) == 3
        assert all(f.source == "static_pk" for f in results)

    def test_zero_radius_still_returns_fallback(self, service):
        results = service.search_nearby_facilities(
            0.0, 0.0, radius_m=0
        )
        assert len(results) == 3
