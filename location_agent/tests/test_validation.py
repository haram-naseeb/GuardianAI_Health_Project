"""Tests for coordinate validation and UserLocation creation."""

import math

import pytest

from location_agent.models import LocationValidationError, UserLocation, validate_coordinates


class TestValidateCoordinates:
    def test_valid_coordinates(self):
        lat, lng = validate_coordinates(31.5204, 74.3587)
        assert lat == pytest.approx(31.5204)
        assert lng == pytest.approx(74.3587)

    def test_numeric_strings_are_accepted(self):
        lat, lng = validate_coordinates("31.5204", "74.3587")
        assert lat == pytest.approx(31.5204)
        assert lng == pytest.approx(74.3587)

    @pytest.mark.parametrize(
        "lat,lng",
        [
            (90.0, 180.0),
            (-90.0, -180.0),
            (0.0, 0.0),
        ],
    )
    def test_boundary_values_are_valid(self, lat, lng):
        assert validate_coordinates(lat, lng) == (lat, lng)

    @pytest.mark.parametrize("lat", [90.0001, -90.5, 91, 180])
    def test_invalid_latitude(self, lat):
        with pytest.raises(LocationValidationError, match="Latitude"):
            validate_coordinates(lat, 0.0)

    @pytest.mark.parametrize("lng", [180.0001, -180.5, 200, -180.0001])
    def test_invalid_longitude(self, lng):
        with pytest.raises(LocationValidationError, match="Longitude"):
            validate_coordinates(0.0, lng)

    @pytest.mark.parametrize(
        "lat,lng",
        [
            (None, None),
            (31.5, None),
            (None, 74.3),
        ],
    )
    def test_missing_coordinates(self, lat, lng):
        with pytest.raises(LocationValidationError, match="Missing"):
            validate_coordinates(lat, lng)

    @pytest.mark.parametrize(
        "lat,lng",
        [
            ("abc", 74.3),
            (31.5, "xyz"),
            ([31.5], 74.3),
        ],
    )
    def test_non_numeric_coordinates(self, lat, lng):
        with pytest.raises(LocationValidationError, match="numeric"):
            validate_coordinates(lat, lng)

    def test_boolean_coordinates_rejected(self):
        with pytest.raises(LocationValidationError, match="booleans"):
            validate_coordinates(True, False)

    @pytest.mark.parametrize(
        "lat,lng",
        [
            (float("nan"), 74.3),
            (31.5, float("inf")),
        ],
    )
    def test_non_finite_coordinates(self, lat, lng):
        with pytest.raises(LocationValidationError, match="finite"):
            validate_coordinates(lat, lng)


class TestUserLocation:
    def test_create_valid_location(self):
        location = UserLocation.create(
            latitude=31.5204,
            longitude=74.3587,
            accuracy_m=15,
            timestamp="2026-09-01T14:00:00Z",
        )
        assert location.latitude == pytest.approx(31.5204)
        assert location.longitude == pytest.approx(74.3587)
        assert location.accuracy_m == 15.0
        assert location.timestamp == "2026-09-01T14:00:00Z"

    def test_optional_fields_default_to_none(self):
        location = UserLocation.create(31.5, 74.3)
        assert location.accuracy_m is None
        assert location.timestamp is None

    def test_negative_accuracy_rejected(self):
        with pytest.raises(LocationValidationError, match="accuracy_m"):
            UserLocation.create(31.5, 74.3, accuracy_m=-5)

    def test_non_numeric_accuracy_rejected(self):
        with pytest.raises(LocationValidationError, match="accuracy_m"):
            UserLocation.create(31.5, 74.3, accuracy_m="high")

    def test_to_dict_omits_missing_fields(self):
        location = UserLocation.create(31.5, 74.3)
        assert location.to_dict() == {
            "latitude": 31.5,
            "longitude": 74.3,
        }

    def test_to_dict_includes_provided_fields(self):
        location = UserLocation.create(31.5, 74.3, accuracy_m=10)
        payload = location.to_dict()
        assert payload["accuracy_m"] == 10.0
        assert "timestamp" not in payload


def test_nan_constant_is_rejected():
    with pytest.raises(LocationValidationError):
        validate_coordinates(math.nan, math.nan)
