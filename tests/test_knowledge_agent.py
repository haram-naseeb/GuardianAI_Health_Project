import json

import pytest

from app.agents import knowledge_agent
from app.agents.knowledge_agent import KnowledgeAgent


RELEVANT_MATCHES = [
    {
        "id": "chunk-100",
        "score": 0.628,
        "metadata": {
            "text": (
                "Action to stem the flow of blood should be taken as soon as "
                "possible. First aid steps: 1. Apply direct pressure to the "
                "bleed. 2. Help the person to lie down. 3. Access EMS."
            ),
            "source": (
                "IFRC International First Aid, Resuscitation and "
                "Education Guidelines 2025.pdf"
            ),
            "page": 257,
        },
    },
    {
        "id": "chunk-101",
        "score": 0.619,
        "metadata": {
            "text": (
                "If the wound is bleeding heavily, apply pressure to the "
                "wound to stop the bleeding."
            ),
            "source": (
                "IFRC International First Aid, Resuscitation and "
                "Education Guidelines 2025.pdf"
            ),
            "page": 275,
        },
    },
]

IRRELEVANT_MATCHES = [
    {
        "id": "chunk-900",
        "score": 0.21,
        "metadata": {
            "text": "Copies of all or part of this study may be made for non-commercial use.",
            "source": "first Aid book.pdf",
            "page": 2,
        },
    },
]

GUIDANCE_PAYLOAD = {
    "guidance_available": True,
    "answer": "Apply direct pressure to the bleed and help the person to lie down.",
    "steps": [
        "Apply direct pressure to the bleed.",
        "Help the person to lie down.",
        "Access EMS.",
    ],
    "cautions": [
        "Shock is likely to develop; monitor the person.",
    ],
    "sources": [
        "IFRC International First Aid, Resuscitation and Education Guidelines 2025.pdf (page 257)",
    ],
}

NO_GUIDANCE_PAYLOAD = {
    "guidance_available": False,
    "answer": "I don't have specific guidance on this.",
    "steps": [],
    "cautions": [],
    "sources": [],
}

MEDICAL_CONTEXT = {
    "severity": "CRITICAL",
    "emergency": True,
    "first_aid_required": True,
    "reason": ["Severe bleeding reported"],
}


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeModels:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_contents = None

    def generate_content(self, model: str, contents) -> FakeResponse:
        self.last_contents = contents
        return FakeResponse(self.response_text)


class FakeEmbedding:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeEmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.last_encoded = None

    def encode(self, text, **kwargs):
        self.last_encoded = text
        return FakeEmbedding([0.1, 0.2, 0.3, 0.4])


def make_fakes(monkeypatch, response_text: str, matches, index_names=("guardianai",)):
    fakes = {}

    class FakeClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            fakes["gemini_api_key"] = api_key
            self.models = FakeModels(response_text)

    class FakeIndex:
        def __init__(self):
            self.last_vector = None
            self.last_top_k = None

        def query(self, vector, top_k, include_metadata):
            self.last_vector = vector
            self.last_top_k = top_k
            return {"matches": matches}

    class FakePinecone:
        def __init__(self, api_key: str):
            self.api_key = api_key
            fakes["pinecone_api_key"] = api_key
            self.index = FakeIndex()

        def list_indexes(self):
            return [{"name": name} for name in index_names]

        def Index(self, name: str):
            fakes["index_name"] = name
            return self.index

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "guardianai")

    monkeypatch.setattr(knowledge_agent.genai, "Client", FakeClient)
    monkeypatch.setattr(knowledge_agent, "Pinecone", FakePinecone)
    monkeypatch.setattr(knowledge_agent, "SentenceTransformer", FakeEmbeddingModel)

    return fakes


def make_agent(monkeypatch, response_text: str, matches=RELEVANT_MATCHES, index_names=("guardianai",)):
    fakes = make_fakes(monkeypatch, response_text, matches, index_names)
    agent = KnowledgeAgent()
    return agent, fakes


def test_retrieve_returns_matches_with_metadata(monkeypatch):
    agent, fakes = make_agent(monkeypatch, json.dumps(GUIDANCE_PAYLOAD))

    results = agent.retrieve("What should I do if someone has severe bleeding?")

    assert len(results) == 2

    assert results[0]["score"] == 0.628
    assert results[0]["source"] == (
        "IFRC International First Aid, Resuscitation and "
        "Education Guidelines 2025.pdf"
    )
    assert results[0]["page"] == 257
    assert "Apply direct pressure" in results[0]["text"]

    assert results[1]["page"] == 275


def test_retrieve_uses_matching_embedding_model_and_query(monkeypatch):
    agent, fakes = make_agent(monkeypatch, json.dumps(GUIDANCE_PAYLOAD))

    agent.retrieve("How do I help someone who is choking?", top_k=3)

    assert agent.embedding_model.model_name == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    assert agent.embedding_model.last_encoded == (
        "How do I help someone who is choking?"
    )

    assert fakes and agent.index.last_vector == [0.1, 0.2, 0.3, 0.4]
    assert agent.index.last_top_k == 3


def test_generate_guidance_with_relevant_context(monkeypatch):
    agent, fakes = make_agent(monkeypatch, json.dumps(GUIDANCE_PAYLOAD))

    result = agent.generate_guidance(
        "What should I do if someone has severe bleeding?"
    )

    assert result["guidance_available"] is True
    assert result["answer"] == (
        "Apply direct pressure to the bleed and help the person to lie down."
    )
    assert result["steps"] == [
        "Apply direct pressure to the bleed.",
        "Help the person to lie down.",
        "Access EMS.",
    ]
    assert result["cautions"] == [
        "Shock is likely to develop; monitor the person."
    ]
    assert "page 257" in result["sources"][0]


def test_generate_guidance_prompt_contains_only_retrieved_context(monkeypatch):
    agent, fakes = make_agent(monkeypatch, json.dumps(GUIDANCE_PAYLOAD))

    agent.generate_guidance(
        "What should I do if someone has severe bleeding?",
        medical_context=MEDICAL_CONTEXT,
    )

    prompt = agent.client.models.last_contents

    assert "What should I do if someone has severe bleeding?" in prompt
    assert "Apply direct pressure to the bleed." in prompt
    assert "IFRC International First Aid" in prompt
    assert "page 257" in prompt
    assert "MEDICAL AGENT CONTEXT" in prompt
    assert "CRITICAL" in prompt


def test_generate_guidance_without_medical_context_omits_section(monkeypatch):
    agent, fakes = make_agent(monkeypatch, json.dumps(GUIDANCE_PAYLOAD))

    agent.generate_guidance("What first aid is appropriate for a burn?")

    prompt = agent.client.models.last_contents

    assert "MEDICAL AGENT CONTEXT" not in prompt


def test_generate_guidance_no_good_matches(monkeypatch):
    agent, fakes = make_agent(
        monkeypatch,
        json.dumps(NO_GUIDANCE_PAYLOAD),
        matches=IRRELEVANT_MATCHES,
    )

    result = agent.generate_guidance(
        "How do I tie a climbing knot?"
    )

    assert result["guidance_available"] is False
    assert result["answer"] == "I don't have specific guidance on this."
    assert result["steps"] == []
    assert result["sources"] == []


def test_generate_guidance_with_empty_retrieval_results(monkeypatch):
    agent, fakes = make_agent(
        monkeypatch,
        json.dumps(NO_GUIDANCE_PAYLOAD),
        matches=[],
    )

    result = agent.generate_guidance("What should I do if someone is choking?")

    assert result["guidance_available"] is False

    prompt = agent.client.models.last_contents
    assert "(no context was retrieved)" in prompt


def test_generate_guidance_parses_json_wrapped_in_code_fences(monkeypatch):
    fenced = f"```json\n{json.dumps(GUIDANCE_PAYLOAD)}\n```"
    agent, fakes = make_agent(monkeypatch, fenced)

    result = agent.generate_guidance(
        "What should I do if someone has severe bleeding?"
    )

    assert result["guidance_available"] is True
    assert result["steps"] == GUIDANCE_PAYLOAD["steps"]


def test_generate_guidance_invalid_gemini_response_raises_value_error(monkeypatch):
    agent, fakes = make_agent(monkeypatch, "This is not a JSON response.")

    with pytest.raises(ValueError, match="valid JSON"):
        agent.generate_guidance("How do I treat a burn?")


def test_missing_gemini_api_key_raises_value_error(monkeypatch):
    make_fakes(
        monkeypatch,
        json.dumps(GUIDANCE_PAYLOAD),
        RELEVANT_MATCHES,
    )
    monkeypatch.delenv("GEMINI_API_KEY")

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        KnowledgeAgent()


def test_missing_pinecone_api_key_raises_value_error(monkeypatch):
    make_fakes(
        monkeypatch,
        json.dumps(GUIDANCE_PAYLOAD),
        RELEVANT_MATCHES,
    )
    monkeypatch.delenv("PINECONE_API_KEY")

    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        KnowledgeAgent()


def test_missing_index_raises_value_error(monkeypatch):
    make_fakes(
        monkeypatch,
        json.dumps(GUIDANCE_PAYLOAD),
        matches=RELEVANT_MATCHES,
        index_names=("some-other-index",),
    )

    with pytest.raises(ValueError, match="was not found"):
        KnowledgeAgent()


def test_retrieve_rejects_non_string_query(monkeypatch):
    agent, fakes = make_agent(monkeypatch, json.dumps(GUIDANCE_PAYLOAD))

    with pytest.raises(TypeError, match="query"):
        agent.retrieve(123)


def test_generate_guidance_rejects_non_dict_medical_context(monkeypatch):
    agent, fakes = make_agent(monkeypatch, json.dumps(GUIDANCE_PAYLOAD))

    with pytest.raises(TypeError, match="medical_context"):
        agent.generate_guidance("How do I treat a burn?", medical_context="CRITICAL")
