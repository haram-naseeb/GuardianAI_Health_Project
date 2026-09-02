"""Deterministic facility ranking for the Location Agent.

No LLM is involved: ranking is a transparent, explainable sort.

Priority order:
1. Emergency relevance (facility type suitability)
2. A valid route exists (travel time is known)
3. Estimated travel time (lower is better)
4. Straight-line distance (lower is better)
5. Facility rating (higher is better)
6. Name (final deterministic tie-breaker)

Travel time is deliberately preferred over raw distance: a farther
hospital on a clear main road can be faster to reach than a closer one
behind congested streets.
"""

from __future__ import annotations

from .models import Facility

# Lower score = more suitable for emergencies.
# Extend this mapping when new facility types are added.
FACILITY_TYPE_SCORES: dict[str, int] = {
    "hospital": 0,
    "emergency_hospital": 0,
    "clinic": 1,
    "pharmacy": 2,
}
DEFAULT_TYPE_SCORE = 3


def type_score(facility_type: str | None) -> int:
    if not facility_type:
        return DEFAULT_TYPE_SCORE
    return FACILITY_TYPE_SCORES.get(facility_type.lower(), DEFAULT_TYPE_SCORE)


def _sort_key(facility: Facility) -> tuple:
    travel_time = (
        facility.travel_time_minutes
        if facility.travel_time_minutes is not None
        else float("inf")
    )
    distance = (
        facility.distance_km
        if facility.distance_km is not None
        else float("inf")
    )
    rating = (
        -facility.rating if facility.rating is not None else float("inf")
    )
    return (
        type_score(facility.facility_type),
        facility.travel_time_minutes is None,
        travel_time,
        distance,
        rating,
        facility.name or "",
    )


def rank_facilities(facilities: list[Facility]) -> list[Facility]:
    """Return facilities sorted from most to least suitable."""
    return sorted(facilities, key=_sort_key)


def explain_recommendation(
    facility: Facility,
    candidate_count: int,
) -> str:
    """Build a factual, non-medical explanation for the top choice."""
    emergency_capable = type_score(facility.facility_type) == 0

    if facility.travel_time_minutes is not None:
        basis = "the shortest estimated travel time"
    elif facility.distance_km is not None:
        basis = "the shortest straight-line distance (routes unavailable)"
    else:
        basis = "the best available facility information"

    capability = (
        "an emergency-capable facility"
        if emergency_capable
        else "a nearby facility"
    )

    return (
        f"Recommended because it is {capability} with {basis} "
        f"among {candidate_count} evaluated candidate(s)."
    )
