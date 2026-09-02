import pytest
from PIL import Image

from app.agents import coordinator_agent as coordinator_module
from app.agents.coordinator_agent import CoordinatorAgent, CoordinatorResult
from app.orchestration.graph import CoordinatorState, _classify_by_keywords, build_graph


DUMMY_IMAGE = Image.new("RGB", (10, 10), color="red")


VISION_RESULT = {
    "scene_type": "road_accident",
    "person_detected": True,
    "person_position": "lying down",
    "visible_bleeding": True,
    "visible_wound": True,
    "possible_burn": False,
    "visible_swelling": False,
    "visible_deformity": False,
    "possible_visible_fracture": False,
    "fire_detected": False,
    "smoke_detected": False,
    "vehicle_damage": True,
    "hazards": ["traffic"],
    "observations": ["Person appears injured", "Visible bleeding"],
}

MEDICAL_RESULT = {
    "severity": "CRITICAL",
    "emergency": True,
    "first_aid_required": True,
    "reason": ["Visible bleeding after road accident."],
}

KNOWLEDGE_RESULT = {
    "guidance_available": True,
    "answer": "Apply direct pressure to the bleed.",
    "steps": ["Apply direct pressure", "Help person lie down", "Access EMS"],
    "cautions": ["Shock is likely to develop; monitor the person."],
    "sources": ["IFRC Guidelines (page 257)"],
}

LOCATION_RESULT = {
    "status": "success",
    "current_location": {"latitude": 24.8607, "longitude": 67.0011},
    "recommended_facility": {
        "name": "Jinnah Hospital Karachi",
        "distance_km": 2.5,
        "travel_time_minutes": 8.0,
    },
    "alternatives": [],
    "searched_at": "2026-01-15T10:30:00Z",
    "source": "static_pk",
}


# ---------------------------------------------------------
# FAKE AGENT CLASSES
# ---------------------------------------------------------

class FakeVisionAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.call_count = 0
        self.last_image = None
        self.last_context = None

    def analyze_image(self, image_input, context=None):
        self.call_count += 1
        self.last_image = image_input
        self.last_context = context
        if self.error:
            raise self.error
        return self.result


class FakeMedicalAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.call_count = 0
        self.last_vision = None
        self.last_user_description = None

    def assess(self, vision_result, user_description, symptoms=None):
        self.call_count += 1
        self.last_vision = vision_result
        self.last_user_description = user_description
        if self.error:
            raise self.error
        return self.result


class FakeKnowledgeAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.call_count = 0
        self.last_query = None
        self.last_medical_context = None

    def generate_guidance(self, query, medical_context=None, top_k=None):
        self.call_count += 1
        self.last_query = query
        self.last_medical_context = medical_context
        if self.error:
            raise self.error
        return self.result


class FakeLocationAgent:
    def __init__(self, result=None):
        self.result = result
        self.call_count = 0
        self.last_state = None

    def handle(self, state):
        self.call_count += 1
        self.last_state = state
        return self.result


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeClassifierModels:
    def __init__(self, response_text):
        self.response_text = response_text

    def generate_content(self, model, contents):
        return FakeResponse(self.response_text)


class FakeClassifierClient:
    def __init__(self, response_text="road_accident"):
        self.models = FakeClassifierModels(response_text)


class BrokenClassifierClient:
    class Models:
        def generate_content(self, model, contents):
            raise RuntimeError("Gemini is down")

    models = Models()


# ---------------------------------------------------------
# TEST HELPERS
# ---------------------------------------------------------

def make_graph(
    classifier_response="road_accident",
    vision_result=None,
    vision_error=None,
    medical_result=None,
    medical_error=None,
    knowledge_result=None,
    knowledge_error=None,
    location_result=None,
    classifier_client=None,
):
    if classifier_client is None:
        classifier_client = FakeClassifierClient(classifier_response)

    vision = FakeVisionAgent(result=vision_result, error=vision_error)
    medical = FakeMedicalAgent(result=medical_result, error=medical_error)
    knowledge = FakeKnowledgeAgent(
        result=knowledge_result, error=knowledge_error
    )
    location = FakeLocationAgent(result=location_result)

    graph = build_graph(
        vision_agent=vision,
        medical_agent=medical,
        knowledge_agent=knowledge,
        location_agent=location,
        classifier_client=classifier_client,
    )

    return graph, vision, medical, knowledge, location


def run_case(graph, **kwargs):
    defaults = {
        "user_description": (
            "My brother had a bike accident and his leg is bleeding."
        ),
        "image_input": None,
        "symptoms": None,
        "latitude": None,
        "longitude": None,
        "accuracy_m": None,
        "timestamp": None,
        "errors": [],
    }
    defaults.update(kwargs)
    return graph.invoke(defaults)


# ---------------------------------------------------------
# ROUTING TESTS
# ---------------------------------------------------------

def test_road_accident_dispatches_all_four_agents():
    graph, vision, medical, knowledge, location = make_graph(
        classifier_response="road_accident",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
        location_result=LOCATION_RESULT,
    )

    state = run_case(graph, image_input=DUMMY_IMAGE)

    assert state["case_type"] == "road_accident"
    assert vision.call_count == 1
    assert medical.call_count == 1
    assert knowledge.call_count == 1
    assert location.call_count == 1
    assert state["status"] == "complete"


def test_minor_injury_dispatches_vision_medical_knowledge():
    graph, vision, medical, knowledge, location = make_graph(
        classifier_response="minor_injury",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
        location_result=LOCATION_RESULT,
    )

    state = run_case(graph, image_input=DUMMY_IMAGE)

    assert state["case_type"] == "minor_injury"
    assert vision.call_count == 1
    assert medical.call_count == 1
    assert knowledge.call_count == 1
    assert location.call_count == 0
    assert state["status"] == "complete"


def test_unclear_dispatches_vision_medical_knowledge():
    graph, vision, medical, knowledge, location = make_graph(
        classifier_response="unclear",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
        location_result=LOCATION_RESULT,
    )

    state = run_case(graph, image_input=DUMMY_IMAGE)

    assert state["case_type"] == "unclear"
    assert vision.call_count == 1
    assert medical.call_count == 1
    assert knowledge.call_count == 1
    assert location.call_count == 0
    assert state["status"] == "complete"


def test_location_not_dispatched_for_minor_injury():
    graph, _, _, _, location = make_graph(
        classifier_response="minor_injury",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
    )

    run_case(graph)

    assert location.call_count == 0


# ---------------------------------------------------------
# ERROR HANDLING TESTS
# ---------------------------------------------------------

def test_vision_failure_records_error_and_continues():
    graph, _, medical, knowledge, _ = make_graph(
        classifier_response="minor_injury",
        vision_error=ValueError("bad image"),
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
    )

    state = run_case(graph, image_input=DUMMY_IMAGE)

    assert state["vision_result"] is None
    assert any("vision" in e for e in state["errors"])
    assert state["status"] == "partial"
    assert medical.call_count == 1
    assert medical.last_vision == {"vision_unavailable": True}


def test_medical_failure_records_error_and_continues():
    graph, _, _, knowledge, _ = make_graph(
        classifier_response="minor_injury",
        vision_result=VISION_RESULT,
        medical_error=ValueError("assessment failed"),
        knowledge_result=KNOWLEDGE_RESULT,
    )

    state = run_case(graph)

    assert state["medical_result"] is None
    assert any("medical" in e for e in state["errors"])
    assert state["status"] == "partial"
    assert knowledge.call_count == 1


def test_knowledge_failure_records_error():
    graph, _, _, _, _ = make_graph(
        classifier_response="minor_injury",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_error=ValueError("retrieval failed"),
    )

    state = run_case(graph)

    assert state["knowledge_result"] is None
    assert any("knowledge" in e for e in state["errors"])
    assert state["status"] == "partial"


def test_all_agents_fail_returns_failed_status():
    graph, _, _, _, _ = make_graph(
        classifier_response="minor_injury",
        vision_error=ValueError("v"),
        medical_error=ValueError("m"),
        knowledge_error=ValueError("k"),
    )

    state = run_case(graph, image_input=DUMMY_IMAGE)

    assert state["status"] == "failed"
    assert len(state["errors"]) == 3


def test_partial_failure_returns_partial_status():
    graph, _, _, _, _ = make_graph(
        classifier_response="minor_injury",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_error=ValueError("k failed"),
    )

    state = run_case(graph)

    assert state["status"] == "partial"


# ---------------------------------------------------------
# MERGE TESTS
# ---------------------------------------------------------

def test_merge_all_succeed_complete_status():
    graph, _, _, _, _ = make_graph(
        classifier_response="road_accident",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
        location_result=LOCATION_RESULT,
    )

    state = run_case(
        graph,
        image_input=DUMMY_IMAGE,
        latitude=24.8607,
        longitude=67.0011,
    )

    assert state["status"] == "complete"
    assert len(state["errors"]) == 0


def test_merge_status_failed_when_no_results():
    graph, _, _, _, _ = make_graph(
        classifier_response="minor_injury",
        vision_error=RuntimeError("v"),
        medical_error=RuntimeError("m"),
        knowledge_error=RuntimeError("k"),
    )

    state = run_case(graph, image_input=DUMMY_IMAGE)

    assert state["status"] == "failed"


# ---------------------------------------------------------
# CLASSIFIER TESTS
# ---------------------------------------------------------

def test_classifier_uses_gemini():
    graph, _, _, _, _ = make_graph(
        classifier_response="road_accident",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
    )

    state = run_case(graph)

    assert state["classifier_source"] == "gemini"
    assert state["case_type"] == "road_accident"


def test_classifier_keyword_fallback_on_gemini_error():
    graph, _, _, _, _ = make_graph(
        classifier_client=BrokenClassifierClient(),
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
    )

    state = run_case(
        graph, user_description="My car had a crash on the road"
    )

    assert state["classifier_source"] == "keyword_fallback"
    assert state["case_type"] == "road_accident"


def test_classify_by_keywords_road_accident():
    assert _classify_by_keywords("car accident on highway") == "road_accident"
    assert _classify_by_keywords("bike crash near school") == "road_accident"
    assert _classify_by_keywords("gaari ka hadsa hua") == "road_accident"


def test_classify_by_keywords_minor_injury():
    assert (
        _classify_by_keywords("I have a small cut on my finger")
        == "minor_injury"
    )
    assert _classify_by_keywords("mamooli kharash hai") == "minor_injury"


def test_classify_by_keywords_unclear():
    assert _classify_by_keywords("help me please") == "unclear"
    assert _classify_by_keywords("") == "unclear"


def test_classifier_no_input_defaults_to_unclear():
    graph, _, _, _, _ = make_graph(
        classifier_response="road_accident",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
    )

    state = run_case(graph, user_description="", image_input=None)

    assert state["case_type"] == "unclear"
    assert state["classifier_source"] == "no_input_default"


def test_image_only_submission_uses_classifier():
    """Image provided but no text description — classifier still runs
    via Gemini with the image, rather than short-circuiting to
    no_input_default."""
    graph, vision, _, _, _ = make_graph(
        classifier_response="road_accident",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
    )

    state = run_case(
        graph,
        user_description="",
        image_input=DUMMY_IMAGE,
    )

    assert state["classifier_source"] == "gemini"
    assert state["case_type"] == "road_accident"
    assert vision.call_count == 1


# ---------------------------------------------------------
# SENTINEL TESTS
# ---------------------------------------------------------

def test_no_image_returns_vision_unavailable_sentinel():
    graph, vision, _, _, _ = make_graph(
        classifier_response="minor_injury",
        vision_result=VISION_RESULT,
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
    )

    state = run_case(
        graph, image_input=None, user_description="I have a small cut"
    )

    assert state["vision_result"] == {
        "vision_unavailable": True,
        "observations": [],
    }


def test_vision_failure_passes_sentinel_to_medical():
    graph, _, medical, _, _ = make_graph(
        classifier_response="minor_injury",
        vision_error=RuntimeError("vision crashed"),
        medical_result=MEDICAL_RESULT,
        knowledge_result=KNOWLEDGE_RESULT,
    )

    state = run_case(graph, image_input=DUMMY_IMAGE)

    assert medical.last_vision == {"vision_unavailable": True}


# ---------------------------------------------------------
# COORDINATOR AGENT CLASS TESTS
# ---------------------------------------------------------

def test_handle_case_returns_coordinator_result(monkeypatch):
    class FakeVisionForInit:
        def __init__(self, api_key=None, model=None):
            pass

        def analyze_image(self, image_input, context=None):
            return VISION_RESULT

    class FakeMedicalForInit:
        def __init__(self, api_key=None, model=None):
            pass

        def assess(self, vision_result, user_description, symptoms=None):
            return MEDICAL_RESULT

    class FakeKnowledgeForInit:
        def __init__(self, gemini_api_key=None, pinecone_api_key=None, **kw):
            pass

        def generate_guidance(self, query, medical_context=None, top_k=None):
            return KNOWLEDGE_RESULT

    class FakeLocationForInit:
        def __init__(self, **kwargs):
            pass

        def handle(self, state):
            return LOCATION_RESULT

    class FakeClientForInit:
        def __init__(self, api_key=None):
            self.models = FakeClassifierModels("road_accident")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")

    monkeypatch.setattr(coordinator_module.genai, "Client", FakeClientForInit)
    monkeypatch.setattr(coordinator_module, "VisionAgent", FakeVisionForInit)
    monkeypatch.setattr(coordinator_module, "MedicalAgent", FakeMedicalForInit)
    monkeypatch.setattr(
        coordinator_module, "KnowledgeAgent", FakeKnowledgeForInit
    )
    monkeypatch.setattr(
        coordinator_module, "LocationAgent", FakeLocationForInit
    )

    agent = CoordinatorAgent()
    result = agent.handle_case(
        user_description="bike accident, bleeding heavily"
    )

    assert isinstance(result, CoordinatorResult)
    assert result.status == "complete"
    assert result.case_type == "road_accident"


def test_missing_gemini_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        CoordinatorAgent()


def test_missing_pinecone_key_raises_value_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        CoordinatorAgent()
