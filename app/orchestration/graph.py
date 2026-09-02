"""
GuardianAI Coordinator — LangGraph StateGraph definition.

Contains the graph topology, node functions, conditional routing,
and merge logic for the GuardianAI multi-agent coordinator.

The graph orchestrates Vision, Medical, Knowledge, and Location
agents based on the classified case type, with graceful degradation
when individual agents fail.
"""

from __future__ import annotations

import logging
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# KEYWORD SETS FOR RULE-BASED FALLBACK CLASSIFICATION
# ---------------------------------------------------------

ROAD_ACCIDENT_KEYWORDS = [
    "accident", "crash", "collision", "hit by", "crashed",
    "car accident", "road accident", "bike accident",
    "vehicle", "truck", "motorcycle",
    "hadsa", "takkar", "sadma",
]

MINOR_INJURY_KEYWORDS = [
    "cut", "scrape", "bruise", "minor burn", "small wound",
    "fell", "tripped", "sprain", "small cut",
    "chot", "kharash", "mamooli", "zakhm",
]


# ---------------------------------------------------------
# COORDINATOR STATE SCHEMA
# ---------------------------------------------------------

class CoordinatorState(TypedDict, total=False):
    """State schema for the coordinator graph.

    Inputs are provided by the caller. Classifier and agent results
    are populated as the graph executes. The errors list uses
    operator.add to accumulate errors across multiple nodes.
    """

    # Inputs
    image_input: Any
    user_description: str
    symptoms: dict[str, Any] | None
    latitude: float | None
    longitude: float | None
    accuracy_m: float | None
    timestamp: str | None

    # Classifier outputs
    case_type: str
    classifier_source: str

    # Agent results
    vision_result: dict[str, Any] | None
    medical_result: dict[str, Any] | None
    knowledge_result: dict[str, Any] | None
    location_result: dict[str, Any] | None

    # Error tracking — accumulates via operator.add
    errors: Annotated[list[str], add]

    # Merge output
    status: str


# ---------------------------------------------------------
# SHARED ERROR-NORMALIZING WRAPPER
# ---------------------------------------------------------

def _safe_agent_node(agent_name: str, call_fn):
    """Shared error-normalizing wrapper for Vision/Medical/Knowledge nodes.

    All three raising agents (Vision, Medical, Knowledge) use this
    wrapper so that exceptions are caught, logged, and recorded in
    the errors list rather than crashing the graph.

    Location's node does not use this wrapper because LocationAgent
    never raises — it returns a structured error dict with
    status="error" when something goes wrong.
    """

    def node(state: CoordinatorState) -> dict:
        try:
            result = call_fn(state)
            return {f"{agent_name}_result": result}
        except Exception as exc:
            logger.warning("%s agent failed: %s", agent_name, exc)
            return {
                f"{agent_name}_result": None,
                "errors": [
                    f"{agent_name}: {type(exc).__name__}: {exc}"
                ],
            }

    node.__name__ = f"{agent_name}_node"
    return node


# ---------------------------------------------------------
# KEYWORD FALLBACK CLASSIFIER
# ---------------------------------------------------------

def _classify_by_keywords(text: str) -> str:
    """Rule-based fallback classifier using keyword matching.

    Used only when the Gemini classifier fails or is unavailable.
    Matches against English and common Urdu transliteration keywords
    to determine the most likely case type.
    """

    lower = text.lower()

    road_score = sum(
        1 for kw in ROAD_ACCIDENT_KEYWORDS if kw in lower
    )
    minor_score = sum(
        1 for kw in MINOR_INJURY_KEYWORDS if kw in lower
    )

    if road_score > minor_score and road_score > 0:
        return "road_accident"
    if minor_score > 0:
        return "minor_injury"
    return "unclear"


# ---------------------------------------------------------
# GRAPH BUILDER
# ---------------------------------------------------------

def build_graph(
    *,
    vision_agent: Any,
    medical_agent: Any,
    knowledge_agent: Any,
    location_agent: Any,
    classifier_client: Any = None,
    classifier_model: str = "gemini-3.5-flash",
) -> Any:
    """Build and compile the coordinator StateGraph.

    All agents are injected so tests can provide fakes.
    classifier_client is the genai.Client used for classification
    (typically the same client used by one of the agents).

    Returns a compiled LangGraph that can be invoked with
    graph.invoke(initial_state).
    """

    # -----------------------------------------------------
    # CLASSIFIER NODE
    # -----------------------------------------------------

    def classify_node(state: CoordinatorState) -> dict:
        user_description = state.get("user_description", "")
        image_input = state.get("image_input")

        # When user_description is empty and no image is provided,
        # default to "unclear" — there is no information to triage
        # on, and assuming a non-emergency could be dangerous.
        if not user_description and image_input is None:
            return {
                "case_type": "unclear",
                "classifier_source": "no_input_default",
            }

        if classifier_client is not None:
            try:
                prompt = (
                    "You are an emergency classifier for "
                    "GuardianAI.\n\n"
                    "Classify this emergency report into exactly "
                    "one category:\n"
                    "- road_accident: vehicle collision, road "
                    "crash, person hit by vehicle\n"
                    "- minor_injury: small cuts, scrapes, minor "
                    "burns, sprains, minor falls\n"
                    "- unclear: cannot determine the type of "
                    "emergency\n\n"
                    "Reply with ONLY the category name."
                )

                if image_input is not None:
                    from pathlib import Path
                    from PIL import Image as PILImage

                    if isinstance(image_input, str):
                        image = PILImage.open(Path(image_input))
                    else:
                        image = image_input

                    contents = [
                        prompt + f"\n\nUSER DESCRIPTION:\n"
                        f"{user_description}",
                        image,
                    ]
                else:
                    contents = (
                        prompt + f"\n\nUSER DESCRIPTION:\n"
                        f"{user_description}"
                    )

                response = classifier_client.models.generate_content(
                    model=classifier_model,
                    contents=contents,
                )

                text = response.text.strip().lower()

                for case in ("road_accident", "minor_injury", "unclear"):
                    if case in text:
                        return {
                            "case_type": case,
                            "classifier_source": "gemini",
                        }

                raise ValueError(
                    f"Unrecognized classification: {text}"
                )

            except Exception as exc:
                logger.warning(
                    "Gemini classifier failed, using keyword "
                    "fallback: %s",
                    exc,
                )

                case = (
                    _classify_by_keywords(user_description)
                    if user_description
                    else "unclear"
                )
                return {
                    "case_type": case,
                    "classifier_source": "keyword_fallback",
                }

        case = (
            _classify_by_keywords(user_description)
            if user_description
            else "unclear"
        )
        return {
            "case_type": case,
            "classifier_source": "keyword_fallback",
        }

    # -----------------------------------------------------
    # AGENT CALL FUNCTIONS
    # -----------------------------------------------------

    def _call_vision(state: CoordinatorState) -> dict:
        image_input = state.get("image_input")
        user_description = state.get("user_description", "")
        context = user_description if user_description else None

        if image_input is None:
            # No image provided — return a sentinel indicating
            # vision had nothing to analyze
            return {
                "vision_unavailable": True,
                "observations": [],
            }

        return vision_agent.analyze_image(
            image_input=image_input,
            context=context,
        )

    def _call_medical(state: CoordinatorState) -> dict:
        vision = state.get("vision_result")

        # When vision failed or was unavailable, pass an explicit
        # sentinel so Medical can distinguish "vision found nothing
        # concerning" from "vision never ran or failed".
        # MedicalAgent.assess only checks isinstance(vision_result,
        # dict) and json.dumps it into the prompt — any dict is
        # accepted.
        if vision is None:
            vision = {"vision_unavailable": True}

        return medical_agent.assess(
            vision_result=vision,
            user_description=state.get("user_description", ""),
            symptoms=state.get("symptoms"),
        )

    def _call_knowledge(state: CoordinatorState) -> dict:
        medical = state.get("medical_result")
        query = (
            state.get("user_description", "")
            or "general emergency first aid"
        )

        return knowledge_agent.generate_guidance(
            query=query,
            medical_context=medical,
        )

    # -----------------------------------------------------
    # LOCATION NODE (no wrapper needed)
    # -----------------------------------------------------

    def location_node(state: CoordinatorState) -> dict:
        # LocationAgent.handle never raises — it returns a
        # structured error dict with status="error" when something
        # goes wrong.
        return {"location_result": location_agent.handle(state)}

    # -----------------------------------------------------
    # MERGE NODE
    # -----------------------------------------------------

    def merge_node(state: CoordinatorState) -> dict:
        errors = state.get("errors", [])
        case_type = state.get("case_type", "unclear")
        vision = state.get("vision_result")
        medical = state.get("medical_result")
        knowledge = state.get("knowledge_result")
        location = state.get("location_result")

        # Determine which agents were expected to run based on
        # case_type
        expected = ["vision", "medical", "knowledge"]
        if case_type == "road_accident":
            expected.append("location")

        results_map = {
            "vision": vision,
            "medical": medical,
            "knowledge": knowledge,
            "location": location,
        }

        succeeded = 0
        for name in expected:
            result = results_map.get(name)
            if result is None:
                continue
            if (
                isinstance(result, dict)
                and result.get("vision_unavailable") is True
            ):
                continue
            succeeded += 1

        if succeeded == 0:
            status = "failed"
        elif errors:
            status = "partial"
        else:
            status = "complete"

        return {"status": status}

    # -----------------------------------------------------
    # BUILD WRAPPED NODES
    # -----------------------------------------------------

    vision_node = _safe_agent_node("vision", _call_vision)
    medical_node = _safe_agent_node("medical", _call_medical)
    knowledge_node = _safe_agent_node("knowledge", _call_knowledge)

    # -----------------------------------------------------
    # BUILD THE GRAPH
    # -----------------------------------------------------

    graph = StateGraph(CoordinatorState)

    graph.add_node("classify", classify_node)
    graph.add_node("vision", vision_node)
    graph.add_node("medical", medical_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("location", location_node)
    graph.add_node("merge", merge_node)

    graph.add_edge(START, "classify")

    # Conditional fan-out from classifier:
    # - road_accident → vision + location (parallel)
    # - minor_injury / unclear → vision only
    # Location runs in parallel with vision for road_accident.
    # Medical waits for vision; knowledge waits for medical.
    def route_after_classify(state: CoordinatorState) -> list[str]:
        case_type = state.get("case_type", "unclear")
        if case_type == "road_accident":
            return ["vision", "location"]
        return ["vision"]

    graph.add_conditional_edges("classify", route_after_classify)

    graph.add_edge("vision", "medical")
    graph.add_edge("medical", "knowledge")
    graph.add_edge("knowledge", "merge")
    graph.add_edge("location", "merge")
    graph.add_edge("merge", END)

    return graph.compile()
