"""
GuardianAI Coordinator Agent

Responsibilities:
- Orchestrate Vision, Medical, Knowledge, and Location agents
  through a LangGraph StateGraph.
- Classify emergency cases and route to appropriate agents.
- Merge agent outputs into a unified CoordinatorResult.
- Gracefully degrade when individual agents fail.

This agent does NOT diagnose medical conditions.
It coordinates the multi-agent pipeline for emergency response.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from PIL import Image
from pydantic import BaseModel

from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.medical_agent import MedicalAgent
from app.agents.vision_agent import VisionAgent
from app.orchestration.graph import build_graph
from location_agent.agent import LocationAgent


load_dotenv()


# ---------------------------------------------------------
# COORDINATOR RESULT MODEL
# ---------------------------------------------------------

class CoordinatorResult(BaseModel):
    """Unified result from the coordinator pipeline.

    Contains per-agent sub-results (each optional), a top-level
    status field, and accumulated errors. Designed to serialize
    directly into a FastAPI response model.
    """

    status: str
    case_type: str
    classifier_source: str
    vision_result: dict[str, Any] | None = None
    medical_result: dict[str, Any] | None = None
    knowledge_result: dict[str, Any] | None = None
    location_result: dict[str, Any] | None = None
    errors: list[str] = []


# ---------------------------------------------------------
# COORDINATOR AGENT CLASS
# ---------------------------------------------------------

class CoordinatorAgent:
    """
    GuardianAI Coordinator Agent.

    Orchestrates the multi-agent pipeline for emergency response.

    Inputs:
        - User description of the emergency
        - Optional image (file path or PIL Image)
        - Optional symptoms
        - Optional GPS coordinates (for location agent)

    Output:
        CoordinatorResult with merged agent outputs.
    """

    def __init__(
        self,
        gemini_api_key: str | None = None,
        pinecone_api_key: str | None = None,
    ):
        self.gemini_api_key = (
            gemini_api_key or os.getenv("GEMINI_API_KEY")
        )

        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Set it as an environment variable."
            )

        self.pinecone_api_key = (
            pinecone_api_key or os.getenv("PINECONE_API_KEY")
        )

        if not self.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY was not found. "
                "Set it as an environment variable."
            )

        classifier_client = genai.Client(
            api_key=self.gemini_api_key
        )

        vision = VisionAgent(api_key=self.gemini_api_key)
        medical = MedicalAgent(api_key=self.gemini_api_key)
        knowledge = KnowledgeAgent(
            gemini_api_key=self.gemini_api_key,
            pinecone_api_key=self.pinecone_api_key,
        )
        location = LocationAgent()

        self._graph = build_graph(
            vision_agent=vision,
            medical_agent=medical,
            knowledge_agent=knowledge,
            location_agent=location,
            classifier_client=classifier_client,
        )

    # ---------------------------------------------------------
    # MAIN ENTRYPOINT
    # ---------------------------------------------------------

    def handle_case(
        self,
        user_description: str = "",
        image_input: str | Image.Image | None = None,
        symptoms: dict[str, Any] | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        accuracy_m: float | None = None,
        timestamp: str | None = None,
    ) -> CoordinatorResult:
        """
        Process an emergency case through the multi-agent pipeline.

        Returns a CoordinatorResult with merged agent outputs.
        """

        initial_state = {
            "user_description": user_description,
            "image_input": image_input,
            "symptoms": symptoms,
            "latitude": latitude,
            "longitude": longitude,
            "accuracy_m": accuracy_m,
            "timestamp": timestamp,
            "errors": [],
        }

        final_state = self._graph.invoke(initial_state)

        return CoordinatorResult(
            status=final_state.get("status", "failed"),
            case_type=final_state.get("case_type", "unclear"),
            classifier_source=final_state.get(
                "classifier_source", "unknown"
            ),
            vision_result=final_state.get("vision_result"),
            medical_result=final_state.get("medical_result"),
            knowledge_result=final_state.get("knowledge_result"),
            location_result=final_state.get("location_result"),
            errors=final_state.get("errors", []),
        )

    # ---------------------------------------------------------
    # DISPLAY RESULT
    # ---------------------------------------------------------

    def display_result(
        self,
        result: CoordinatorResult,
    ) -> None:

        print("\n")
        print("=" * 70)
        print("        GUARDIANAI — COORDINATOR RESULT")
        print("=" * 70)

        print(f"Status: {result.status}")
        print(f"Case Type: {result.case_type}")
        print(f"Classifier Source: {result.classifier_source}")

        if result.vision_result:
            print("\n[Vision Agent]")
            print(
                f"  Scene: "
                f"{result.vision_result.get('scene_type', 'N/A')}"
            )
            print(
                f"  Visible Bleeding: "
                f"{result.vision_result.get('visible_bleeding', 'N/A')}"
            )

        if result.medical_result:
            print("\n[Medical Agent]")
            print(
                f"  Severity: "
                f"{result.medical_result.get('severity', 'N/A')}"
            )
            print(
                f"  Emergency: "
                f"{result.medical_result.get('emergency', 'N/A')}"
            )

        if result.knowledge_result:
            print("\n[Knowledge Agent]")
            guidance_avail = result.knowledge_result.get(
                "guidance_available", False
            )
            print(f"  Guidance Available: {guidance_avail}")
            if guidance_avail:
                answer = result.knowledge_result.get("answer", "")
                print(f"  Answer: {answer}")

        if result.location_result:
            print("\n[Location Agent]")
            loc_status = result.location_result.get("status", "N/A")
            print(f"  Status: {loc_status}")
            if loc_status == "success":
                facility = result.location_result.get(
                    "recommended_facility", {}
                )
                print(f"  Recommended: {facility.get('name', 'N/A')}")

        if result.errors:
            print("\n[Errors]")
            for error in result.errors:
                print(f"  • {error}")

        print("=" * 70)
