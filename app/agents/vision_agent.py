"""
GuardianAI Vision Agent

Responsibilities:
- Analyze emergency images
- Identify visually observable features
- Detect visible hazards
- Return structured JSON

This agent does NOT diagnose medical conditions.
Its output is intended as input for the Medical Agent.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from PIL import Image


load_dotenv()


class VisionAgent:
    """
    GuardianAI Vision Agent.

    Inputs:
        - Image file path or PIL Image object
        - Optional user context

    Output:
        Structured, visually observable findings.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Set it as an environment variable."
            )

        self.model = model

        self.client = genai.Client(
            api_key=self.api_key
        )

    # ---------------------------------------------------------
    # VISION AGENT INSTRUCTIONS
    # ---------------------------------------------------------

    def _build_prompt(
        self,
        context: str | None = None,
    ) -> str:

        prompt = """
You are GuardianAI's Vision Agent.

Your responsibility is to analyze an emergency image
and identify ONLY visually observable information.

IMPORTANT SAFETY RULES:

1. Do NOT diagnose medical conditions.
2. Do NOT claim internal injuries.
3. Do NOT claim that a person definitely has a disease
   or medical condition.
4. Do NOT invent information.
5. Only report information reasonably supported by
   the image.
6. If something is unclear, mark it as false or uncertain.
7. Your output will be passed to a Medical Triage Agent.
8. The Medical Triage Agent will make the emergency
   severity assessment.

Analyze the following visual categories:

SCENE:
- road accident
- vehicle accident
- fire
- indoor emergency
- outdoor emergency
- other

PERSON:
- person detected
- approximate visible position
- lying down
- standing
- sitting
- unclear

VISIBLE INJURY INDICATORS:
- visible bleeding
- visible wound
- possible burn appearance
- visible swelling
- visible deformity
- possible visible fracture appearance

ENVIRONMENTAL HAZARDS:
- fire
- smoke
- traffic
- damaged vehicle
- dangerous objects
- other hazards

Return ONLY valid JSON.

Use exactly this structure:

{
    "scene_type": "",
    "person_detected": false,
    "person_position": "",
    "visible_bleeding": false,
    "visible_wound": false,
    "possible_burn": false,
    "visible_swelling": false,
    "visible_deformity": false,
    "possible_visible_fracture": false,
    "fire_detected": false,
    "smoke_detected": false,
    "vehicle_damage": false,
    "hazards": [],
    "observations": []
}
"""

        if context:
            prompt += f"""

ADDITIONAL CONTEXT FROM USER:

{context}
"""

        return prompt

    # ---------------------------------------------------------
    # IMAGE LOADING
    # ---------------------------------------------------------

    def _load_image(
        self,
        image_input: str | Image.Image,
    ) -> Image.Image:

        if isinstance(image_input, Image.Image):
            return image_input

        if isinstance(image_input, str):
            image_path = Path(image_input)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_input}"
                )

            return Image.open(image_path)

        raise TypeError(
            "image_input must be a file path "
            "or PIL Image object."
        )

    # ---------------------------------------------------------
    # JSON PARSER
    # ---------------------------------------------------------

    def _parse_json(self, response_text: str) -> dict[str, Any]:

        text = response_text.strip()

        # Remove Markdown JSON code fences if Gemini adds them.
        text = re.sub(
            r"```json",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"```",
            "",
            text,
        )

        text = text.strip()

        try:
            return json.loads(text)

        except json.JSONDecodeError as error:

            raise ValueError(
                "Vision Agent did not return valid JSON.\n\n"
                f"Raw response:\n{text}"
            ) from error

    # ---------------------------------------------------------
    # OUTPUT VALIDATION
    # ---------------------------------------------------------

    def _validate_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:

        required_fields = [
            "scene_type",
            "person_detected",
            "person_position",
            "visible_bleeding",
            "visible_wound",
            "possible_burn",
            "visible_swelling",
            "visible_deformity",
            "possible_visible_fracture",
            "fire_detected",
            "smoke_detected",
            "vehicle_damage",
            "hazards",
            "observations",
        ]

        for field in required_fields:
            if field not in result:
                result[field] = None

        return result

    # ---------------------------------------------------------
    # MAIN IMAGE ANALYSIS
    # ---------------------------------------------------------

    def analyze_image(
        self,
        image_input: str | Image.Image,
        context: str | None = None,
    ) -> dict[str, Any]:

        image = self._load_image(image_input)

        prompt = self._build_prompt(context=context)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                image,
            ],
        )

        result = self._parse_json(response.text)

        return self._validate_result(result)

    # ---------------------------------------------------------
    # DISPLAY RESULT
    # ---------------------------------------------------------

    def display_result(
        self,
        result: dict[str, Any],
    ) -> None:

        print("\n")
        print("=" * 70)
        print("        GUARDIANAI — VISION AGENT")
        print("=" * 70)

        print(
            f"Scene Type: "
            f"{result.get('scene_type')}"
        )

        print(
            f"Person Detected: "
            f"{result.get('person_detected')}"
        )

        print(
            f"Person Position: "
            f"{result.get('person_position')}"
        )

        print(
            f"Visible Bleeding: "
            f"{result.get('visible_bleeding')}"
        )

        print(
            f"Visible Wound: "
            f"{result.get('visible_wound')}"
        )

        print(
            f"Possible Burn: "
            f"{result.get('possible_burn')}"
        )

        print(
            f"Visible Swelling: "
            f"{result.get('visible_swelling')}"
        )

        print(
            f"Visible Deformity: "
            f"{result.get('visible_deformity')}"
        )

        print(
            f"Possible Visible Fracture: "
            f"{result.get('possible_visible_fracture')}"
        )

        print(
            f"Fire Detected: "
            f"{result.get('fire_detected')}"
        )

        print(
            f"Smoke Detected: "
            f"{result.get('smoke_detected')}"
        )

        print(
            f"Vehicle Damage: "
            f"{result.get('vehicle_damage')}"
        )

        print("\nHazards:")

        for hazard in result.get("hazards", []):
            print(f"  • {hazard}")

        print("\nObservations:")

        for observation in result.get("observations", []):
            print(f"  • {observation}")

        print("=" * 70)
