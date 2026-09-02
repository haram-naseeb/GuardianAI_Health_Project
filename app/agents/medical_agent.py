"""
GuardianAI Medical Agent

Responsibilities:
- Assess emergency urgency using user information and optional Vision findings.
- Provide cautious severity assessment.
- Identify whether emergency assistance may be required.
- Identify whether first-aid guidance may be required.

This agent does NOT diagnose medical conditions.
It is intentionally standalone so it can be integrated with
the rest of GuardianAI later.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai


load_dotenv()


class MedicalAgent:
    """
    GuardianAI Medical Assessment Agent.

    Inputs:
        - Vision Agent structured result
        - User description
        - Optional symptoms

    Output:
        Structured medical urgency assessment.

    This agent does NOT provide a definitive diagnosis.
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
    # MEDICAL AGENT INSTRUCTIONS
    # ---------------------------------------------------------

    def _build_prompt(
        self,
        vision_result: dict[str, Any],
        user_description: str,
        symptoms: dict[str, Any] | None = None,
    ) -> str:

        symptoms = symptoms or {}

        return f"""
You are the Medical Assessment Agent of GuardianAI.

Your task is to assess the urgency of an emergency situation
using the information provided by the user and the structured
findings from the Vision Agent.

IMPORTANT SAFETY RULES:

1. You are NOT a doctor.
2. Do NOT provide a definitive medical diagnosis.
3. Do NOT claim that the patient definitely has a disease,
   fracture, internal injury, or other medical condition.
4. Use cautious language such as:
   "possible injury", "potentially serious", or
   "requires professional evaluation".
5. Do NOT invent symptoms or observations.
6. Use only the information provided.
7. If information is missing, treat it as unknown.
8. If there are potentially life-threatening signs,
   classify the situation as CRITICAL.
9. This is an emergency triage/support system, not a
   replacement for professional medical care.

SEVERITY LEVELS:

LOW:
No obvious serious emergency indicators are present.

MODERATE:
A concerning injury or condition may be present and
medical evaluation may be appropriate, but there is no
clear immediate life-threatening indicator in the
available information.

CRITICAL:
Potentially life-threatening indicators are present,
such as:
- unconsciousness
- no normal breathing
- severe bleeding
- serious trauma
- severe breathing difficulty
- major fire/emergency danger
- other clearly serious emergency indicators

VISION AGENT FINDINGS:

{json.dumps(vision_result, indent=2)}

USER DESCRIPTION:

{user_description}

OPTIONAL SYMPTOMS:

{json.dumps(symptoms, indent=2)}

TASK:

1. Assess emergency severity.
2. Decide whether emergency assistance may be required.
3. Decide whether first-aid guidance may be required.
4. Give the main reasons supporting the assessment.
5. Use cautious, non-diagnostic language.

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
    "severity": "LOW",
    "emergency": false,
    "first_aid_required": true,
    "reason": []
}}
"""

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
                "Medical Agent did not return valid JSON.\n\n"
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
            "severity",
            "emergency",
            "first_aid_required",
            "reason",
        ]

        for field in required_fields:
            if field not in result:
                result[field] = None

        # Keep severity within the project's defined levels.
        valid_severities = {
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        }

        if result["severity"] not in valid_severities:
            result["severity"] = "MODERATE"

        # Make sure reason is always a list.
        if not isinstance(result["reason"], list):
            result["reason"] = [str(result["reason"])]

        return result

    # ---------------------------------------------------------
    # MAIN MEDICAL ASSESSMENT
    # ---------------------------------------------------------

    def assess(
        self,
        vision_result: dict[str, Any],
        user_description: str,
        symptoms: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        if not isinstance(vision_result, dict):
            raise TypeError(
                "vision_result must be a dictionary."
            )

        if not isinstance(user_description, str):
            raise TypeError(
                "user_description must be a string."
            )

        prompt = self._build_prompt(
            vision_result=vision_result,
            user_description=user_description,
            symptoms=symptoms,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
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
        print("=" * 60)
        print("        GUARDIANAI - MEDICAL AGENT")
        print("=" * 60)

        print(
            f"Severity: "
            f"{result.get('severity')}"
        )

        print(
            f"Emergency: "
            f"{result.get('emergency')}"
        )

        print(
            f"First Aid Required: "
            f"{result.get('first_aid_required')}"
        )

        print("\nReasons:")

        for reason in result.get("reason", []):
            print(f"  - {reason}")

        print("=" * 60)
