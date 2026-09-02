"""Static Pakistan hospital list for offline demos and hackathons.

Enable with LOCATION_AGENT_STATIC=true.  Uses a real hand-curated list
of 14 Pakistani emergency hospitals instead of the mock service's
obviously-fake test data.  Distances are haversine (straight-line) and
routes are estimated at a fixed city speed, so every result carries
source="static_pk" and status="estimated" — it can never be mistaken
for live Google Maps output.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..geo import haversine_km
from .maps_service import MapsServiceBase, RawFacility, RouteInfo

logger = logging.getLogger(__name__)

_AVERAGE_SPEED_KMH = 25.0
_NEAREST_FALLBACK_COUNT = 3


class StaticHospitalService(MapsServiceBase):
    """Real Pakistan hospitals, no network calls."""

    source = "static_pk"

    HOSPITALS: tuple[dict, ...] = (
        {
            "name": "Jinnah Hospital Lahore",
            "latitude": 31.4843,
            "longitude": 74.2969,
            "address": "Ahmed, Usmani Rd, Faisal Town",
            "phone": "+92 42 99231400",
            "rating": 4.0,
        },
        {
            "name": "Services Hospital",
            "latitude": 31.5410,
            "longitude": 74.3324,
            "address": "Shadman 1, Shadman, Lahore",
            "phone": "+92 42 99203402",
            "rating": 3.9,
        },
        {
            "name": "Mayo Hospital Emergency Department",
            "latitude": 31.5708,
            "longitude": 74.3165,
            "address": "Hospital Rd, Anarkali Bazaar, Lahore",
            "phone": "+92 309 7884930",
            "rating": 3.9,
        },
        {
            "name": "Civil Hospital Emergency",
            "latitude": 24.8593,
            "longitude": 67.0114,
            "address": "New Labour Colony Nanakwara, Karachi",
            "phone": "+92 21 33335656",
            "rating": 3.9,
        },
        {
            "name": "Memon Medical Institute Emergency",
            "latitude": 24.9454,
            "longitude": 67.1467,
            "address": "MMI Circulation Rd, Gulzar-e-Hijri, Karachi",
            "phone": "+92 21 34691147",
            "rating": 3.2,
        },
        {
            "name": "Shifa International Hospitals",
            "latitude": 33.6753,
            "longitude": 73.0667,
            "address": "4 Pitras Bukhari Rd, H-8/4, Islamabad",
            "phone": "+92 51 8464646",
            "rating": 3.3,
        },
        {
            "name": "PAF Hospital Islamabad",
            "latitude": 33.7092,
            "longitude": 73.0158,
            "address": "Main Margalla Rd, E-9/1, Islamabad",
            "phone": "+92 51 9564000",
            "rating": 3.5,
        },
        {
            "name": "RIC Emergency",
            "latitude": 33.6153,
            "longitude": 73.0810,
            "address": "Chaklala Cantt, Rawalpindi",
            "phone": "+92 51 9281111",
            "rating": 4.3,
        },
        {
            "name": "Rawalpindi International Hospital",
            "latitude": 33.6342,
            "longitude": 73.0761,
            "address": "A Block, 57A Iran Rd, Satellite Town, Rawalpindi",
            "phone": "+92 51 8444481",
            "rating": 4.3,
        },
        {
            "name": "Allied Hospital 2",
            "latitude": 31.4211,
            "longitude": 73.0947,
            "address": "Mall Road, Faisalabad",
            "phone": "+92 41 9200140",
            "rating": 4.1,
        },
        {
            "name": "Multan Medical Mission Hospital",
            "latitude": 30.2086,
            "longitude": 71.4846,
            "address": "Wahdat Colony, Multan",
            "phone": "+92 61 4424757",
            "rating": 4.7,
        },
        {
            "name": "Lady Reading Hospital MTI",
            "latitude": 34.0106,
            "longitude": 71.5688,
            "address": "Soekarno Rd, Pipal Mandi, Peshawar",
            "phone": "+92 91 9211430",
            "rating": 4.1,
        },
        {
            "name": "Civil Hospital Quetta",
            "latitude": 30.1939,
            "longitude": 67.0089,
            "address": "M.A Jinnah Road, Quetta",
            "phone": "+92 81 9202017",
            "rating": 3.7,
        },
        {
            "name": "Quetta Hospital",
            "latitude": 30.1923,
            "longitude": 67.0094,
            "address": "Fatah Muhammed Road, Quetta",
            "phone": "+92 81 2829915",
            "rating": 3.8,
        },
    )

    def search_nearby_facilities(
        self,
        latitude: float,
        longitude: float,
        radius_m: int = 10_000,
        facility_types: Sequence[str] = ("hospital",),
    ) -> list[RawFacility]:
        radius_km = radius_m / 1000.0

        scored: list[tuple[float, dict]] = []
        for hospital in self.HOSPITALS:
            dist_km = haversine_km(
                latitude,
                longitude,
                hospital["latitude"],
                hospital["longitude"],
            )
            scored.append((dist_km, hospital))

        scored.sort(key=lambda pair: pair[0])

        within_radius = [
            (dist, h) for dist, h in scored if dist <= radius_km
        ]

        if within_radius:
            selected = within_radius
        else:
            logger.info(
                "No static hospitals within %d m of (%.4f, %.4f); "
                "falling back to nearest %d.",
                radius_m,
                latitude,
                longitude,
                _NEAREST_FALLBACK_COUNT,
            )
            selected = scored[:_NEAREST_FALLBACK_COUNT]

        facilities: list[RawFacility] = []
        for dist_km, hospital in selected:
            facilities.append(
                RawFacility(
                    name=hospital["name"],
                    latitude=hospital["latitude"],
                    longitude=hospital["longitude"],
                    facility_type="hospital",
                    address=hospital["address"],
                    place_id=None,
                    rating=hospital["rating"],
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
        straight_km = haversine_km(
            origin_latitude,
            origin_longitude,
            destination_latitude,
            destination_longitude,
        )

        road_km = straight_km * 1.3
        duration_s = (road_km / _AVERAGE_SPEED_KMH) * 3600.0

        return RouteInfo(
            distance_m=road_km * 1000.0,
            duration_s=duration_s,
            status="estimated",
        )
