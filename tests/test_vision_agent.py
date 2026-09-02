import json

import pytest
from PIL import Image

from app.agents import vision_agent
from app.agents.vision_agent import VisionAgent


INJURY_PAYLOAD = {
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

CLEAR_PAYLOAD = {
    "scene_type": "other",
    "person_detected": False,
    "person_position": "unclear",
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
    "observations": [],
}


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeModels:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def generate_content(self, model: str, contents: list) -> FakeResponse:
        return FakeResponse(self.response_text)


def make_vision_agent(monkeypatch, response_text: str) -> VisionAgent:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

    class FakeClient:
        def __init__(self, api_key: str):
            self.models = FakeModels(response_text)

    monkeypatch.setattr(vision_agent.genai, "Client", FakeClient)
    return VisionAgent()


def make_image() -> Image.Image:
    return Image.new("RGB", (4, 4))


def test_analyze_image_returns_injury_indicators(monkeypatch):
    agent = make_vision_agent(monkeypatch, json.dumps(INJURY_PAYLOAD))

    result = agent.analyze_image(make_image())

    assert result["scene_type"] == "road_accident"
    assert result["person_detected"] is True
    assert result["visible_bleeding"] is True
    assert result["visible_wound"] is True
    assert result["hazards"] == ["traffic"]
    assert result["observations"] == [
        "Person appears injured",
        "Visible bleeding",
    ]


def test_analyze_image_no_concerning_indicators(monkeypatch):
    agent = make_vision_agent(monkeypatch, json.dumps(CLEAR_PAYLOAD))

    result = agent.analyze_image(make_image())

    assert result["scene_type"] == "other"
    assert result["person_detected"] is False
    assert result["visible_bleeding"] is False
    assert result["visible_wound"] is False
    assert result["hazards"] == []
    assert result["observations"] == []


def test_analyze_image_accepts_file_path(monkeypatch, tmp_path):
    image_path = tmp_path / "emergency.jpg"
    make_image().save(image_path)

    agent = make_vision_agent(monkeypatch, json.dumps(INJURY_PAYLOAD))

    result = agent.analyze_image(str(image_path))

    assert result["scene_type"] == "road_accident"
    assert result["visible_bleeding"] is True


def test_analyze_image_parses_json_wrapped_in_code_fences(monkeypatch):
    fenced_response = f"```json\n{json.dumps(INJURY_PAYLOAD)}\n```"
    agent = make_vision_agent(monkeypatch, fenced_response)

    result = agent.analyze_image(make_image())

    assert result["scene_type"] == "road_accident"
    assert result["visible_bleeding"] is True
    assert result["person_detected"] is True


def test_analyze_image_invalid_gemini_response_raises_value_error(monkeypatch):
    agent = make_vision_agent(monkeypatch, "This is not a JSON response.")

    with pytest.raises(ValueError, match="valid JSON"):
        agent.analyze_image(make_image())


def test_missing_api_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        VisionAgent()


def test_analyze_image_rejects_invalid_image_input_type(monkeypatch):
    agent = make_vision_agent(monkeypatch, json.dumps(CLEAR_PAYLOAD))

    with pytest.raises(TypeError, match="image_input"):
        agent.analyze_image(123)


def test_analyze_image_includes_context_in_prompt(monkeypatch):
    captured = {}

    class RecordingModels:
        def generate_content(self, model: str, contents: list) -> FakeResponse:
            captured["model"] = model
            captured["contents"] = contents
            return FakeResponse(json.dumps(CLEAR_PAYLOAD))

    class RecordingClient:
        def __init__(self, api_key: str):
            self.models = RecordingModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setattr(vision_agent.genai, "Client", RecordingClient)

    agent = VisionAgent()
    context = "Photo taken after a bike accident on a busy road."
    agent.analyze_image(make_image(), context=context)

    prompt = captured["contents"][0]
    assert context in prompt
    assert "ADDITIONAL CONTEXT FROM USER" in prompt
