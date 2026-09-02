"""Tests for the Location Agent data models and result schemas."""

from location_agent.models import (
    ERROR_INVALID_COORDINATES,
    ERROR_NO_FACILITIES,
    Facility,
    LocationResult,
    UserLocation,
)


def _facility(**overrides) -> Facility:
    defaults = dict(
        name="Test Hospital",
        latitude=31.53,
        longitude=74.36,
        facility_type="hospital",
        address="Test Address",
        place_id="place-1",
        rating=4.2,
        distance_km=1.2,
        travel_time_minutes=6.0,
        route_status="ok",
    )
    defaults.update(overrides)
    return Facility(**defaults)


class TestFacility:
    def test_to_dict_contains_schema_fields(self):
        payload = _facility().to_dict()
        for key in (
            "name",
            "address",
            "place_id",
            "latitude",
            "longitude",
            "distance_km",
            "travel_time_minutes",
            "facility_type",
            "reason",
        ):
            assert key in payload

    def test_missing_fields_stay_none(self):
        facility = _facility(rating=None, address=None, place_id=None)
        payload = facility.to_dict()
        assert payload["rating"] is None
        assert payload["address"] is None
        assert payload["place_id"] is None


class TestLocationResult:
    def test_success_schema_matches_coordinator_contract(self):
        location = UserLocation.create(31.5204, 74.3587, accuracy_m=15)
        result = LocationResult.success(
            current_location=location,
            recommended=_facility(),
            alternatives=[_facility(name="Other Hospital")],
            source="google_maps",
        )
        payload = result.to_dict()

        assert set(payload.keys()) == {
            "status",
            "current_location",
            "recommended_facility",
            "alternatives",
            "searched_at",
            "source",
        }
        assert payload["status"] == "success"
        assert payload["source"] == "google_maps"
        assert payload["current_location"]["latitude"] == 31.5204
        assert payload["recommended_facility"]["name"] == "Test Hospital"
        assert len(payload["alternatives"]) == 1
        assert payload["searched_at"]

    def test_error_schema_matches_coordinator_contract(self):
        result = LocationResult.failure(
            ERROR_NO_FACILITIES, "No facilities found."
        )
        payload = result.to_dict()

        assert set(payload.keys()) == {
            "status",
            "error_code",
            "message",
            "fallback_available",
        }
        assert payload["status"] == "error"
        assert payload["error_code"] == ERROR_NO_FACILITIES
        assert payload["fallback_available"] is True

    def test_error_always_keeps_fallback_available(self):
        result = LocationResult.failure(
            ERROR_INVALID_COORDINATES, "Bad coordinates."
        )
        assert result.to_dict()["fallback_available"] is True
