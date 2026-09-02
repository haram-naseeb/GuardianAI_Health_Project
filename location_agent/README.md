# GuardianAI — Location Agent

Location intelligence for **GuardianAI**, the AI-assisted emergency
coordination and first-response support system.

The Location Agent finds nearby emergency facilities, calculates road
distance **and** estimated travel time to each one, ranks them with
transparent deterministic logic, and returns a structured result the
future Coordinator (LangGraph) can consume directly.

> Safety principle: this agent **enhances** emergency coordination. It
> never gates, delays, or replaces the emergency call path (1122), and
> it never invents facilities, addresses, coordinates, or travel times.

---

## What it does

Given a user location:

```json
{
  "latitude": 31.5204,
  "longitude": 74.3587,
  "accuracy_m": 15,
  "timestamp": "2026-09-01T14:00:00Z"
}
```

it returns the best emergency facility plus ranked alternatives:

```json
{
  "status": "success",
  "current_location": { "latitude": 31.5204, "longitude": 74.3587, "accuracy_m": 15.0 },
  "recommended_facility": {
    "name": "...",
    "address": "...",
    "place_id": "...",
    "latitude": 0.0,
    "longitude": 0.0,
    "distance_km": 2.72,
    "travel_time_minutes": 5.4,
    "facility_type": "hospital",
    "route_status": "ok",
    "rating": 4.6,
    "source": "google_maps",
    "reason": "Recommended because it is an emergency-capable facility with the shortest estimated travel time among 3 evaluated candidate(s)."
  },
  "alternatives": [ ],
  "searched_at": "2026-09-01T09:00:00+00:00",
  "source": "google_maps"
}
```

On failure it returns a structured error and never raises:

```json
{
  "status": "error",
  "error_code": "NO_FACILITIES_FOUND",
  "message": "No emergency facilities were found near the provided location.",
  "fallback_available": true
}
```

`fallback_available` is always `true`: emergency calling never depends
on this agent.

## Architecture

```
location_agent/
├── __init__.py          # Public package API
├── __main__.py          # CLI demo: python -m location_agent
├── agent.py             # LocationAgent: search -> route -> rank pipeline
├── config.py            # Environment-variable configuration
├── geo.py               # Haversine helper
├── models.py            # UserLocation, Facility, LocationResult, validation
├── ranking.py           # Deterministic ranking + explanation
├── services/
│   ├── maps_service.py       # Google Maps implementation + parsers
│   └── mock_maps_service.py  # Offline mock for dev/tests
├── tests/               # pytest suite (no API key needed)
├── examples/            # Future LangGraph Coordinator example
├── .env.example
└── requirements.txt
```

Pipeline:

```
User location
  -> validation (lat/lng ranges, missing input)
  -> facility search (Places Nearby Search, type=hospital)
  -> route + ETA per candidate (Directions API)
  -> deterministic ranking
  -> structured LocationResult
```

### Ranking logic (no LLM involved)

Priority order, all deterministic:

1. Emergency relevance of the facility type (`hospital` first)
2. A valid route exists (known travel time)
3. Estimated travel time — preferred over raw distance
4. Straight-line distance
5. Facility rating
6. Name (final tie-breaker for reproducibility)

The recommendation always carries a plain-language `reason`. Ranking is
isolated in `ranking.py` so it can be tuned without touching the agent.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env      # then edit .env
```

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GOOGLE_MAPS_API_KEY` | for live mode | — | Google Maps Platform key |
| `LOCATION_AGENT_MOCK` | no | `false` | `true` = clearly marked fake data, no API calls |
| `LOCATION_SEARCH_RADIUS_M` | no | `10000` | Nearby-search radius in meters |
| `LOCATION_MAX_FACILITIES` | no | `10` | Max candidates routed/ranked |
| `LOCATION_REQUEST_TIMEOUT_S` | no | `10` | Per-request timeout |

## Google Maps API setup

1. Create a project in [Google Cloud Console](https://console.cloud.google.com).
2. Enable **Places API** (Nearby Search) and **Directions API**.
3. Create an API key and restrict it to those two APIs.
4. Put the key in `.env` as `GOOGLE_MAPS_API_KEY` — never in code.

Note: the service layer uses the official `googlemaps` client (Places
Nearby Search + Directions). Migrating later to Places (New) / Routes
API only changes `services/maps_service.py`.

## Mock mode

```bash
LOCATION_AGENT_MOCK=true python -m location_agent --latitude 31.5204 --longitude 74.3587
```

Mock mode returns deterministic facilities marked
`(MOCK TEST DATA)` with `source: "mock"`. Mock output is never mixed
with production responses. Use it for demos and development so you do
not burn Google Maps quota.

## How to run

```bash
# CLI (mock mode)
LOCATION_AGENT_MOCK=true python -m location_agent --latitude 31.5204 --longitude 74.3587 --accuracy-m 15

# From Python
python
>>> from location_agent import LocationAgent
>>> agent = LocationAgent()                      # reads .env / environment
>>> result = agent.find_best_facility(31.5204, 74.3587, accuracy_m=15)
>>> result.to_dict()

# Future-Coordinator interface (dict in / dict out)
>>> agent.handle({"latitude": 31.5204, "longitude": 74.3587})
```

## How to test

Tests run fully offline — no Google Maps key, no network:

```bash
python -m pytest location_agent/tests -v
```

Covered: valid/invalid/missing coordinates, empty results, multiple
facilities, ranking by travel time, equal-time tie-breaks, API failure,
timeout, quota, malformed responses, per-facility route failure, mock
mode, and the final structured response schema.

## Error handling

| Error code | Meaning |
|---|---|
| `MISSING_COORDINATES` | latitude and/or longitude not provided |
| `INVALID_COORDINATES` | out of range, non-numeric, or non-finite input |
| `FACILITY_SEARCH_FAILED` | Places search failed or was denied |
| `NO_FACILITIES_FOUND` | search succeeded but returned nothing |
| `API_QUOTA_EXCEEDED` | Google Maps quota/billing limit hit |
| `REQUEST_TIMEOUT` | provider request timed out |
| `MALFORMED_RESPONSE` | provider response could not be interpreted |
| `UNKNOWN_ERROR` | anything else (logged, still structured) |

Rules the agent follows:

- Never fabricate a hospital, address, coordinate, or route. Fields the
  provider does not supply stay `null`.
- A single failed route does not fail the search; that facility is kept
  with `route_status: "unavailable"` and straight-line distance.
- The whole search failing returns a structured error — never an
  exception into the Coordinator.

## LangGraph integration (future)

The agent exposes a dict-in/dict-out `handle()` method designed to be
wrapped as a graph node:

```python
from location_agent import LocationAgent

def location_node(state: dict) -> dict:
    state["location_result"] = LocationAgent().handle(state)
    return state

# graph.add_node("location", location_node)
```

See `examples/coordinator_integration_example.py` for the full sketch:

```
Coordinator -> Location Agent -> Maps tools -> Facility ranking
            -> Structured LocationResult -> Coordinator
```

## Security & privacy notes

- API keys come only from environment variables (`.env`); `.env` is
  never committed and keys never appear in code, logs, or tests.
- User coordinates are not persisted by this agent; they are used
  in-memory for the single search and appear in logs only at reduced
  precision (`DEBUG` level).
- No LLM generates geographic data. All facility and route facts come
  from Google Maps (or are clearly labeled mock data).
- This agent provides location assistance only — no medical advice, no
  diagnosis — and never blocks the 1122 emergency call path.
