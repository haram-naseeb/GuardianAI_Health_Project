"""
GuardianAI Knowledge Agent

Responsibilities:
- Retrieve relevant first-aid knowledge chunks from the
  GuardianAI Pinecone knowledge base.
- Generate grounded first-aid guidance with Gemini using
  ONLY the retrieved context.
- Include the retrieved sources (document, page) with the
  guidance.

This agent does NOT diagnose medical conditions and does
NOT invent guidance that is absent from the retrieved
context. It supplements, but does not replace, professional
medical care or emergency services.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_NAME = "guardianai"
DEFAULT_TOP_K = 5


class KnowledgeAgent:
    """
    GuardianAI Knowledge Agent.

    Inputs:
        - User question
        - Optional Medical Agent context

    Output:
        Structured, source-grounded first-aid guidance.
    """

    def __init__(
        self,
        gemini_api_key: str | None = None,
        pinecone_api_key: str | None = None,
        index_name: str | None = None,
        model: str = "gemini-3.5-flash",
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.gemini_api_key = (
            gemini_api_key
            or os.getenv("GEMINI_API_KEY")
        )

        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Set it as an environment variable."
            )

        self.pinecone_api_key = (
            pinecone_api_key
            or os.getenv("PINECONE_API_KEY")
        )

        if not self.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY was not found. "
                "Set it as an environment variable."
            )

        self.index_name = (
            index_name
            or os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
        ).lower()

        self.model = model
        self.top_k = top_k

        self.client = genai.Client(
            api_key=self.gemini_api_key
        )

        self.pinecone = Pinecone(
            api_key=self.pinecone_api_key
        )

        existing_indexes = [
            item["name"]
            for item in self.pinecone.list_indexes()
        ]

        if self.index_name not in existing_indexes:
            raise ValueError(
                f"Pinecone index '{self.index_name}' was not found. "
                "Run scripts/ingest_documents.py first."
            )

        self.index = self.pinecone.Index(self.index_name)

        # Must be the exact same model used during ingestion;
        # otherwise query embeddings live in a different
        # vector space and retrieval quality breaks.
        self.embedding_model = SentenceTransformer(
            embedding_model_name
        )

    # ---------------------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        effective_top_k = top_k or self.top_k

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.index.query(
            vector=query_embedding,
            top_k=effective_top_k,
            include_metadata=True,
        )

        matches = []

        for match in results["matches"]:

            metadata = match.get("metadata") or {}

            matches.append(
                {
                    "score": match.get("score"),
                    "text": metadata.get("text"),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                }
            )

        return matches

    # ---------------------------------------------------------
    # KNOWLEDGE AGENT INSTRUCTIONS
    # ---------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        matches: list[dict[str, Any]],
        medical_context: dict[str, Any] | None = None,
    ) -> str:

        context_blocks = []

        for i, match in enumerate(matches, start=1):
            context_blocks.append(
                f"[{i}] Source: {match['source']} "
                f"(page {match['page']})\n"
                f"{match['text']}"
            )

        if context_blocks:
            retrieved_context = "\n\n".join(context_blocks)
        else:
            retrieved_context = "(no context was retrieved)"

        medical_context_section = ""

        if medical_context:
            medical_context_section = (
                "MEDICAL AGENT CONTEXT (optional severity "
                f"assessment):\n\n"
                f"{json.dumps(medical_context, indent=2)}\n\n"
            )

        return f"""
You are the Knowledge Agent of GuardianAI.

Your task is to provide first-aid guidance for the user's
question using ONLY the retrieved context from the
GuardianAI first-aid knowledge base.

IMPORTANT SAFETY RULES:

1. Use ONLY the information in the RETRIEVED CONTEXT below.
2. Do NOT invent first-aid steps that are not present in
   the retrieved context.
3. If the retrieved context does not cover the question,
   set guidance_available to false and answer with
   "I don't have specific guidance on this."
4. Use cautious language such as "may help" or
   "the guidelines recommend".
5. Do NOT make definitive medical claims or diagnoses.
6. Encourage accessing emergency medical services (EMS)
   when the situation may be serious.
7. This guidance supplements but does not replace
   professional medical care or emergency services.

USER QUESTION:

{query}

{medical_context_section}RETRIEVED CONTEXT:

{retrieved_context}

TASK:

1. Decide whether the retrieved context covers the question.
2. If it does, summarize the first-aid steps from the
   context in clear, simple language.
3. List important cautions or warnings mentioned in the
   context.
4. List the sources used (document name and page).

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
    "guidance_available": true,
    "answer": "",
    "steps": [],
    "cautions": [],
    "sources": []
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
                "Knowledge Agent did not return valid JSON.\n\n"
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
            "guidance_available",
            "answer",
            "steps",
            "cautions",
            "sources",
        ]

        for field in required_fields:
            if field not in result:
                result[field] = None

        if result["guidance_available"] is not None and not isinstance(
            result["guidance_available"], bool
        ):
            result["guidance_available"] = bool(
                result["guidance_available"]
            )

        for field in ("steps", "cautions", "sources"):
            if result[field] is not None and not isinstance(
                result[field], list
            ):
                result[field] = [str(result[field])]

        return result

    # ---------------------------------------------------------
    # MAIN GUIDANCE GENERATION
    # ---------------------------------------------------------

    def generate_guidance(
        self,
        query: str,
        medical_context: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if medical_context is not None and not isinstance(
            medical_context, dict
        ):
            raise TypeError(
                "medical_context must be a dictionary."
            )

        matches = self.retrieve(query, top_k=top_k)

        prompt = self._build_prompt(
            query=query,
            matches=matches,
            medical_context=medical_context,
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
        print("        GUARDIANAI - KNOWLEDGE AGENT")
        print("=" * 60)

        print(
            f"Guidance Available: "
            f"{result.get('guidance_available')}"
        )

        print(f"\nAnswer:\n  {result.get('answer')}")

        print("\nSteps:")

        for step in result.get("steps") or []:
            print(f"  - {step}")

        print("\nCautions:")

        for caution in result.get("cautions") or []:
            print(f"  - {caution}")

        print("\nSources:")

        for source in result.get("sources") or []:
            print(f"  - {source}")

        print("=" * 60)
