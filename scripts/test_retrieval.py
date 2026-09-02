"""GuardianAI Knowledge Base Retrieval Test

Embeds a query with the same model used at ingestion time and prints the
top 5 matches from the GuardianAI Pinecone index.

Usage:
    python scripts/test_retrieval.py
    python scripts/test_retrieval.py "What should I do if someone has severe bleeding?"
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_QUERIES = [
    "What should I do if someone has severe bleeding?",
    "What first aid is appropriate for a burn?",
    "What should I do if someone is choking?",
    "What should I do if someone is unconscious?",
]

load_dotenv(PROJECT_ROOT / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "guardianai-firstaid").lower()


def search_pinecone(index, embedding_model, query: str, top_k: int = 5):
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    return results


def main():
    if not PINECONE_API_KEY:
        sys.exit("PINECONE_API_KEY was not found. Add it to your .env file.")

    queries = sys.argv[1:] or DEFAULT_QUERIES

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [item["name"] for item in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        sys.exit(
            f"Index '{INDEX_NAME}' was not found in your Pinecone project. "
            "Run scripts/ingest_documents.py first."
        )

    index = pc.Index(INDEX_NAME)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Querying index: {INDEX_NAME}")

    for query in queries:
        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("=" * 80)

        results = search_pinecone(index, embedding_model, query)

        for match in results["matches"]:
            print("-" * 80)
            print("Score:", match["score"])
            print("Source:", match["metadata"]["source"])
            print("Page:", match["metadata"]["page"])
            print("Text:")
            print(match["metadata"]["text"][:1000])


if __name__ == "__main__":
    main()
