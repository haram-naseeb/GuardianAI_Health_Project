"""End-to-end tests for LocationAgent using injected fake services.

No Google Maps API key or network access is required.
"""

import pytest

from location_agent.agent import LocationAgent
from location_agent.config import LocationAgentConfig
from location_agent.models import (
    ERROR_API_QUOTA,
    ERROR_INVALID_COORDINATES,
    ERROR_MALFORMED_RESPONSE,
    ERROR_MISSING_COORDINATES,
    ERROR_NO_FACILITIES,
    ERROR_SEARCH_FAILED,
    ERROR_TIMEOUT,
)
from location_agent.services.maps_service import (
    MapsMalformedResponseError,
    MapsQuotaError,
    MapsServiceBase,
    MapsServiceError,
    MapsTimeoutError,
    RawFacility,
    RouteInfo,
    parse_directions,
    parse_nearby_place,
)
from location_agent.services.mock_maps_service import MockMapsService

LAHORE_LAT = 31.5204
LAHORE_LNG = 74.3587


def _config(**overrides) -> LocationAgentConfig:
    defaults = dict(
        google_maps_api_key=None,
        mock_mode=False,
        search_radius_m=10_000,
        max_facilities=10,
        request_timeout_s=5,
    )
    defaults.update(overrides)
    return LocationAgentConfig(**defaults)


class FakeMapsService(MapsServiceBase):
    """Configurable stand-in for GoogleMapsService."""

    source = "fake"

    def __init__(
        self,
        facilities=None,
        route_error: Exception | None = None,
        search_error: Exception | None = None,
    ):
        self.facilities = facilities if facilities is not None else []
        self.route_error = route_error
        self.search_error = search_error
        self.route_calls = 0

    def search_nearby_facilities(self, latitude, longitude, radius_m=10_000, facility_types=("hospital",)):
        if self.search_error is not None:
            raise self.search_error
        return self.facilities

    def get_route(self, origin_latitude, origin_longitude, destination_latitude, destination_longitude):
        self.route_calls += 1
        if self.route_error is not None:
            raise self.route_error
        return RouteInfo(distance_m=3000.0, duration_s=420.0, status="ok")


def _raw(name, dlat, dlng, **overrides) -> RawFacility:
    defaults = dict(
        facility_type="hospital",
        address=f"{name} Address",
        place_id=f"place-{name}",
        rating=4.0,
        source="fake",
    )
    defaults.update(overrides)
    return RawFacility(
        name=name,
        latitude=LAHORE_LAT + dlat,
        longitude=LAHORE_LNG + dlng,
        **defaults,
    )


class TestSuccessfulFlow:
    def test_multiple_facilities_produce_recommendation_and_alternatives(self):
        service = FakeMapsService(
            facilities=[
                _raw("Alpha Hospital", 0.01, 0.0),
                _raw("Beta Hospital", 0.02, 0.0),
                _raw("Gamma Hospital", 0.03, 0.0),
            ]
        )
        agent = LocationAgent(config=_config(), maps_service=service)

        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)

        assert result.status == "success"
        assert result.recommended_facility is not None
        assert len(result.alternatives) == 2
        assert result.recommended_facility.reason
        assert result.source == "fake"

    def test_route_information_is_attached_to_candidates(self):
        service = FakeMapsService(facilities=[_raw("Alpha Hospital", 0.01, 0.0)])
        agent = LocationAgent(config=_config(), maps_service=service)

        payload = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG).to_dict()
        facility = payload["recommended_facility"]

        assert facility["distance_km"] == pytest.approx(3.0)
        assert facility["travel_time_minutes"] == pytest.approx(7.0)
        assert facility["route_status"] == "ok"

    def test_final_structured_response_schema(self):
        service = FakeMapsService(facilities=[_raw("Alpha Hospital", 0.01, 0.0)])
        agent = LocationAgent(config=_config(), maps_service=service)

        payload = agent.find_best_facility(
            LAHORE_LAT, LAHORE_LNG, accuracy_m=15
        ).to_dict()

        assert payload["status"] == "success"
        assert payload["current_location"] == {
            "latitude": LAHORE_LAT,
            "longitude": LAHORE_LNG,
            "accuracy_m": 15.0,
        }
        assert set(payload.keys()) == {
            "status",
            "current_location",
            "recommended_facility",
            "alternatives",
            "searched_at",
            "source",
        }

    def test_travel_time_prioritized_over_distance(self):
        # Same distance for both; only travel time differs via routes.
        near = _raw("Near Hospital", 0.005, 0.0)
        far = _raw("Far Hospital", 0.02, 0.0)

        class SplitRouteService(FakeMapsService):
            def get_route(self, o_lat, o_lng, d_lat, d_lng):
                if d_lat == near.latitude:
                    return RouteInfo(distance_m=2000, duration_s=1500)  # 25 min
                return RouteInfo(distance_m=5000, duration_s=480)  # 8 min

        agent = LocationAgent(
            config=_config(),
            maps_service=SplitRouteService(facilities=[near, far]),
        )
        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)
        assert result.recommended_facility.name == "Far Hospital"

    def test_per_facility_route_failure_is_not_fatal(self):
        service = FakeMapsService(
            facilities=[_raw("Alpha Hospital", 0.01, 0.0)],
            route_error=MapsServiceError("routing down"),
        )
        agent = LocationAgent(config=_config(), maps_service=service)

        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)

        assert result.status == "success"
        facility = result.recommended_facility
        assert facility.route_status == "unavailable"
        assert facility.travel_time_minutes is None
        assert facility.distance_km is not None  # straight-line fallback
        assert "routes unavailable" in facility.reason


class TestInputValidation:
    def test_invalid_latitude_returns_structured_error(self):
        agent = LocationAgent(config=_config(), maps_service=FakeMapsService())
        result = agent.find_best_facility(95.0, LAHORE_LNG)
        assert result.status == "error"
        assert result.error_code == ERROR_INVALID_COORDINATES
        assert result.fallback_available is True

    def test_invalid_longitude_returns_structured_error(self):
        agent = LocationAgent(config=_config(), maps_service=FakeMapsService())
        result = agent.find_best_facility(LAHORE_LAT, -200.0)
        assert result.error_code == ERROR_INVALID_COORDINATES

    def test_missing_coordinates_return_structured_error(self):
        agent = LocationAgent(config=_config(), maps_service=FakeMapsService())
        result = agent.find_best_facility(None, None)
        assert result.status == "error"
        assert result.error_code == ERROR_MISSING_COORDINATES


class TestFailureHandling:
    def test_empty_facility_results(self):
        agent = LocationAgent(config=_config(), maps_service=FakeMapsService())
        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)
        assert result.status == "error"
        assert result.error_code == ERROR_NO_FACILITIES
        assert result.fallback_available is True

    def test_generic_api_failure(self):
        service = FakeMapsService(
            search_error=MapsServiceError("provider down")
        )
        agent = LocationAgent(config=_config(), maps_service=service)
        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)
        assert result.error_code == ERROR_SEARCH_FAILED

    def test_timeout_maps_to_timeout_error(self):
        service = FakeMapsService(
            search_error=MapsTimeoutError("timed out")
        )
        agent = LocationAgent(config=_config(), maps_service=service)
        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)
        assert result.error_code == ERROR_TIMEOUT

    def test_quota_error_maps_to_quota_code(self):
        service = FakeMapsService(search_error=MapsQuotaError("quota"))
        agent = LocationAgent(config=_config(), maps_service=service)
        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)
        assert result.error_code == ERROR_API_QUOTA

    def test_malformed_response_maps_to_malformed_code(self):
        service = FakeMapsService(
            search_error=MapsMalformedResponseError("bad shape")
        )
        agent = LocationAgent(config=_config(), maps_service=service)
        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)
        assert result.error_code == ERROR_MALFORMED_RESPONSE

    def test_unexpected_exception_becomes_unknown_error(self):
        service = FakeMapsService(search_error=RuntimeError("boom"))
        agent = LocationAgent(config=_config(), maps_service=service)
        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)
        assert result.status == "error"
        assert result.error_code == "UNKNOWN_ERROR"

    def test_missing_key_without_mock_raises_at_construction(self):
        with pytest.raises(ValueError, match="GOOGLE_MAPS_API_KEY"):
            LocationAgent(config=_config())


class TestMockMode:
    def test_mock_mode_selected_from_config(self):
        agent = LocationAgent(config=_config(mock_mode=True))
        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)

        assert result.status == "success"
        assert result.source == "mock"
        assert "MOCK TEST DATA" in result.recommended_facility.name
        assert all(
            "MOCK TEST DATA" in alt.name for alt in result.alternatives
        )

    def test_mock_ranking_prefers_faster_route_over_shorter_distance(self):
        # Mock General Hospital is closer but congested (factor 2.6);
        # Mock Emergency Medical Center is farther but fast (factor 1.1).
        agent = LocationAgent(config=_config(mock_mode=True))
        result = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG)
        assert "Emergency Medical Center" in result.recommended_facility.name
        assert result.recommended_facility.travel_time_minutes is not None

    def test_mock_mode_never_reports_google_maps_source(self):
        agent = LocationAgent(config=_config(mock_mode=True))
        payload = agent.find_best_facility(LAHORE_LAT, LAHORE_LNG).to_dict()
        assert payload["source"] == "mock"
        assert payload["recommended_facility"]["source"] == "mock"


class TestResponseParsing:
    def test_parse_nearby_place_valid_entry(self):
        place = {
            "name": "Real Hospital",
            "place_id": "abc123",
            "vicinity": "123 Main Street",
            "rating": 4.3,
            "geometry": {"location": {"lat": 31.5, "lng": 74.3}},
        }
        facility = parse_nearby_place(place, "hospital")
        assert facility is not None
        assert facility.name == "Real Hospital"
        assert facility.address == "123 Main Street"
        assert facility.rating == pytest.approx(4.3)

    @pytest.mark.parametrize(
        "place",
        [
            {},  # empty
            {"name": "No Geometry"},  # missing geometry
            {"geometry": {"location": {"lat": 31.5}}},  # missing lng + name
            {"name": "Bad Coords", "geometry": {"location": {"lat": "x", "lng": "y"}}},
            "not a dict",
        ],
    )
    def test_parse_nearby_place_rejects_malformed_entries(self, place):
        assert parse_nearby_place(place, "hospital") is None

    def test_parse_directions_extracts_distance_and_duration(self):
        routes = [
            {
                "legs": [
                    {
                        "distance": {"value": 4200},
                        "duration": {"value": 600},
                    }
                ]
            }
        ]
        route = parse_directions(routes)
        assert route.distance_m == 4200.0
        assert route.duration_s == 600.0

    def test_parse_directions_empty_routes_returns_none(self):
        assert parse_directions([]) is None

    def test_parse_directions_malformed_shape_raises(self):
        with pytest.raises(MapsMalformedResponseError):
            parse_directions([{"legs": [{"distance": {}}]}])


class TestLangGraphInterface:
    def test_handle_accepts_state_dict_and_returns_dict(self):
        service = FakeMapsService(facilities=[_raw("Alpha Hospital", 0.01, 0.0)])
        agent = LocationAgent(config=_config(), maps_service=service)

        payload = agent.handle(
            {
                "latitude": LAHORE_LAT,
                "longitude": LAHORE_LNG,
                "accuracy_m": 15,
                "timestamp": "2026-09-01T14:00:00Z",
            }
        )
        assert payload["status"] == "success"
        assert payload["recommended_facility"]["name"] == "Alpha Hospital"

    def test_handle_with_missing_coordinates_returns_error_dict(self):
        agent = LocationAgent(config=_config(), maps_service=FakeMapsService())
        payload = agent.handle({})
        assert payload["status"] == "error"
        assert payload["error_code"] == ERROR_MISSING_COORDINATES
        assert payload["fallback_available"] is True
