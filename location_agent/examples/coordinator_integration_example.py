"""Example: how the future GuardianAI Coordinator could call the
Location Agent as a LangGraph node.

Conceptual flow:

    Coordinator -> Location Agent -> Maps tools -> Facility ranking
                -> Structured LocationResult -> Coordinator

This file is illustrative only; the Coordinator itself is not part of
the Location Agent package.
"""

from __future__ import annotations

import json

from location_agent import LocationAgent


def location_node(state: dict) -> dict:
    """LangGraph-style node.

    Reads the user location from shared state, runs the Location Agent,
    and writes the structured result back into state under
    "location_result". The node never raises: on any failure it returns
    a structured error dict with fallback_available=True, so the
    emergency call path is never blocked.
    """
    agent = LocationAgent()
    state["location_result"] = agent.handle(state)
    return state


# A future Coordinator would wire the node roughly like this:
#
#   from langgraph.graph import StateGraph
#
#   graph = StateGraph(dict)
#   graph.add_node("location", location_node)
#   graph.add_edge("triage", "location")
#   graph.add_edge("location", "report")


if __name__ == "__main__":
    # Simulated coordinator state (e.g. after a First Aid / Emergency
    # button press). Run with LOCATION_AGENT_MOCK=true to avoid using
    # Google Maps quota:
    #
    #   LOCATION_AGENT_MOCK=true python -m location_agent.examples.coordinator_integration_example
    state = {
        "latitude": 31.5204,
        "longitude": 74.3587,
        "accuracy_m": 15,
        "timestamp": "2026-09-01T14:00:00Z",
    }

    state = location_node(state)
    print(json.dumps(state["location_result"], indent=2))
