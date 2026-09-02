"""Tests for the deterministic facility ranking logic."""

from location_agent.models import Facility
from location_agent.ranking import explain_recommendation, rank_facilities


def _facility(name, **overrides) -> Facility:
    defaults = dict(
        latitude=31.5,
        longitude=74.3,
        facility_type="hospital",
        distance_km=2.0,
        travel_time_minutes=10.0,
        route_status="ok",
    )
    defaults.update(overrides)
    return Facility(name=name, **defaults)


class TestRanking:
    def test_travel_time_beats_raw_distance(self):
        close_but_slow = _facility(
            "Close Slow", distance_km=2.0, travel_time_minutes=15.0
        )
        far_but_fast = _facility(
            "Far Fast", distance_km=5.0, travel_time_minutes=6.0
        )
        ranked = rank_facilities([close_but_slow, far_but_fast])
        assert ranked[0].name == "Far Fast"

    def test_equal_travel_time_falls_back_to_distance(self):
        near = _facility("Near", distance_km=2.0, travel_time_minutes=8.0)
        far = _facility("Far", distance_km=6.0, travel_time_minutes=8.0)
        ranked = rank_facilities([far, near])
        assert ranked[0].name == "Near"

    def test_equal_time_and_distance_falls_back_to_rating(self):
        low_rated = _facility(
            "Low Rated", rating=3.0, distance_km=2.0, travel_time_minutes=8.0
        )
        high_rated = _facility(
            "High Rated", rating=4.8, distance_km=2.0, travel_time_minutes=8.0
        )
        ranked = rank_facilities([low_rated, high_rated])
        assert ranked[0].name == "High Rated"

    def test_missing_rating_ranks_last_in_full_tie(self):
        unrated = _facility(
            "Unrated", rating=None, distance_km=2.0, travel_time_minutes=8.0
        )
        rated = _facility(
            "Rated", rating=3.5, distance_km=2.0, travel_time_minutes=8.0
        )
        ranked = rank_facilities([unrated, rated])
        assert ranked[0].name == "Rated"

    def test_emergency_relevance_beats_travel_time(self):
        clinic = _facility(
            "Fast Clinic",
            facility_type="clinic",
            travel_time_minutes=2.0,
            distance_km=0.5,
        )
        hospital = _facility(
            "Slower Hospital",
            facility_type="hospital",
            travel_time_minutes=9.0,
            distance_km=4.0,
        )
        ranked = rank_facilities([clinic, hospital])
        assert ranked[0].name == "Slower Hospital"

    def test_unknown_type_ranks_below_known_types(self):
        unknown = _facility(
            "Unknown", facility_type="helipad", travel_time_minutes=1.0
        )
        hospital = _facility("Hospital", travel_time_minutes=9.0)
        ranked = rank_facilities([unknown, hospital])
        assert ranked[0].name == "Hospital"

    def test_facilities_without_routes_rank_after_routed_ones(self):
        unrouted = _facility(
            "Unrouted",
            travel_time_minutes=None,
            route_status="unavailable",
            distance_km=0.5,
        )
        routed = _facility("Routed", travel_time_minutes=12.0, distance_km=6.0)
        ranked = rank_facilities([unrouted, routed])
        assert ranked[0].name == "Routed"

    def test_ranking_is_deterministic_for_shuffled_input(self):
        a = _facility("A", travel_time_minutes=5.0)
        b = _facility("B", travel_time_minutes=3.0)
        c = _facility("C", travel_time_minutes=9.0)
        assert [f.name for f in rank_facilities([a, b, c])] == [
            f.name for f in rank_facilities([c, a, b])
        ] == ["B", "A", "C"]


class TestExplanation:
    def test_explanation_for_routed_hospital(self):
        facility = _facility("Hospital", travel_time_minutes=5.0)
        text = explain_recommendation(facility, candidate_count=3)
        assert "emergency-capable" in text
        assert "shortest estimated travel time" in text
        assert "3 evaluated candidate(s)" in text

    def test_explanation_when_routes_unavailable(self):
        facility = _facility(
            "Hospital", travel_time_minutes=None, distance_km=2.0
        )
        text = explain_recommendation(facility, candidate_count=2)
        assert "straight-line distance" in text

    def test_explanation_makes_no_medical_claims(self):
        facility = _facility("Hospital")
        text = explain_recommendation(facility, candidate_count=1).lower()
        for forbidden in ("diagnos", "fracture", "injury", "treatment"):
            assert forbidden not in text
