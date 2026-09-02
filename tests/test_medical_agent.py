import json

import pytest

from app.agents import medical_agent
from app.agents.medical_agent import MedicalAgent


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
    "observations": [
        "Person appears injured",
        "Visible bleeding",
    ],
}

USER_DESCRIPTION = "My brother had a bike accident and his leg is bleeding heavily."

SYMPTOMS = {
    "conscious": True,
    "breathing": True,
    "severe_pain": True,
}

LOW_VISION_RESULT = {
    "scene_type": "minor_fall",
    "person_detected": True,
    "person_position": "standing",
    "visible_bleeding": False,
    "visible_wound": False,
    "possible_burn": False,
    "visible_swelling": False,
    "visible_deformity": False,
    "possible_visible_fracture": False,
    "fire_detected": False,
    "smoke_detected": False,
    "vehicle_damage": False,
    "hazards": [],
    "observations": ["Person appears unhurt"],
}

LOW_USER_DESCRIPTION = (
    "I fell off my bike but I'm fine, just a small scrape on my elbow."
)

LOW_SYMPTOMS = {
    "conscious": True,
    "breathing": True,
    "severe_pain": False,
}


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeModels:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def generate_content(self, model: str, contents: str) -> FakeResponse:
        return FakeResponse(self.response_text)


def gemini_payload(
    severity: str = "CRITICAL",
    emergency: bool = True,
    first_aid_required: bool = True,
) -> str:
    return json.dumps(
        {
            "severity": severity,
            "emergency": emergency,
            "first_aid_required": first_aid_required,
            "reason": ["Visible bleeding after a road accident."],
        }
    )


def make_agent(monkeypatch, response_text: str) -> MedicalAgent:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

    class FakeClient:
        def __init__(self, api_key: str):
            self.models = FakeModels(response_text)

    monkeypatch.setattr(medical_agent.genai, "Client", FakeClient)
    return MedicalAgent()


def test_assess_returns_valid_triage_result(monkeypatch):
    agent = make_agent(monkeypatch, gemini_payload())

    result = agent.assess(
        vision_result=VISION_RESULT,
        user_description=USER_DESCRIPTION,
        symptoms=SYMPTOMS,
    )

    assert result["severity"] == "CRITICAL"
    assert result["emergency"] is True
    assert result["first_aid_required"] is True


def test_assess_low_severity_minor_injury(monkeypatch):
    agent = make_agent(
        monkeypatch,
        gemini_payload(
            severity="LOW",
            emergency=False,
            first_aid_required=False,
        ),
    )

    result = agent.assess(
        vision_result=LOW_VISION_RESULT,
        user_description=LOW_USER_DESCRIPTION,
        symptoms=LOW_SYMPTOMS,
    )

    assert result["severity"] == "LOW"
    assert result["emergency"] is False
    assert result["first_aid_required"] is False


def test_assess_parses_json_wrapped_in_code_fences(monkeypatch):
    fenced_response = f"```json\n{gemini_payload()}\n```"
    agent = make_agent(monkeypatch, fenced_response)

    result = agent.assess(
        vision_result=VISION_RESULT,
        user_description=USER_DESCRIPTION,
        symptoms=SYMPTOMS,
    )

    assert result["severity"] == "CRITICAL"
    assert result["emergency"] is True
    assert result["first_aid_required"] is True


def test_assess_invalid_gemini_response_raises_value_error(monkeypatch):
    agent = make_agent(monkeypatch, "This is not a JSON response.")

    with pytest.raises(ValueError, match="valid JSON"):
        agent.assess(
            vision_result=VISION_RESULT,
            user_description=USER_DESCRIPTION,
            symptoms=SYMPTOMS,
        )


def test_missing_api_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        MedicalAgent()


def test_assess_rejects_non_dict_vision_result(monkeypatch):
    agent = make_agent(monkeypatch, gemini_payload())

    with pytest.raises(TypeError, match="vision_result"):
        agent.assess(
            vision_result="not a dict",
            user_description=USER_DESCRIPTION,
        )


def test_assess_rejects_non_string_user_description(monkeypatch):
    agent = make_agent(monkeypatch, gemini_payload())

    with pytest.raises(TypeError, match="user_description"):
        agent.assess(
            vision_result=VISION_RESULT,
            user_description=123,
        )
