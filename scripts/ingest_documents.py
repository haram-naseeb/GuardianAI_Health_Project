"""GuardianAI Knowledge Base Ingestion

Reads every PDF in data/first_aid_sources/, extracts and cleans per-page
text, chunks it, embeds the chunks with sentence-transformers, and upserts
them into the GuardianAI Pinecone index.

Pipeline (mirrors the RAG prototype notebook):
    extract -> clean -> chunk (800/150) -> embed (all-MiniLM-L6-v2, 384-dim)
    -> upsert to Pinecone serverless (aws/us-east-1, cosine) in batches of 100
"""

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "first_aid_sources"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
UPSERT_BATCH_SIZE = 100
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

load_dotenv(PROJECT_ROOT / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "guardianai-firstaid").lower()


def extract_pdf_pages(pdf_path: Path) -> list:
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def load_documents() -> list:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        sys.exit(f"No PDF files found in {DATA_DIR}")

    documents = []
    chunk_id = 0
    total_pages = 0

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        pages = extract_pdf_pages(pdf_file)
        total_pages += len(pages)
        print(f"  Pages with extractable text: {len(pages)}")

        for page in pages:
            cleaned = clean_text(page["text"])

            for chunk in chunk_text(cleaned):
                documents.append(
                    {
                        "id": f"chunk-{chunk_id}",
                        "text": chunk,
                        "source": pdf_file.name,
                        "page": page["page"],
                    }
                )
                chunk_id += 1

    print(f"\nTotal pages extracted: {total_pages}")
    print(f"Total chunks: {len(documents)}")

    return documents


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [item["name"] for item in pc.list_indexes()]

    if INDEX_NAME in existing_indexes:
        print(f"Using existing Pinecone index: {INDEX_NAME}")
        return pc.Index(INDEX_NAME)

    print(
        f"Creating Pinecone index: {INDEX_NAME} "
        f"(dimension={EMBEDDING_DIMENSION}, metric=cosine, aws/us-east-1)"
    )

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    index = pc.Index(INDEX_NAME)
    wait_until_ready(index)
    return index


def wait_until_ready(index, timeout_seconds: int = 180) -> None:
    print("Waiting for index to be ready", end="")

    deadline = time.time() + timeout_seconds

    while True:
        try:
            index.describe_index_stats()
            print(" ready.")
            return
        except Exception:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Index '{INDEX_NAME}' was not ready after {timeout_seconds} seconds."
                )
            print(".", end="", flush=True)
            time.sleep(5)


def confirm_upsert_if_nonempty(index) -> None:
    stats = index.describe_index_stats()
    existing_count = stats["total_vector_count"]

    if existing_count == 0:
        return

    print(f"\nWARNING: index '{INDEX_NAME}' already contains {existing_count} vectors.")

    print(
        "Re-running ingestion will upsert chunk-0, chunk-1, ... from scratch. With the "
        "SAME documents this overwrites the old vectors cleanly, but with CHANGED or "
        "REMOVED documents the numbering shifts and stale vectors may remain. To start "
        "completely fresh, delete the index in the Pinecone console and re-run."
    )

    try:
        answer = input("Continue anyway? [y/N]: ").strip().lower()
    except EOFError:
        answer = "n"

    if answer not in ("y", "yes"):
        sys.exit("Aborted. No vectors were uploaded.")


def embed_documents(documents: list):
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [doc["text"] for doc in documents]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )

    print(f"Embedding shape: {embeddings.shape}")

    return embeddings


def upsert_documents(index, documents: list, embeddings) -> None:
    vectors = [
        {
            "id": doc["id"],
            "values": embeddings[i].tolist(),
            "metadata": {
                "text": doc["text"],
                "source": doc["source"],
                "page": doc["page"],
            },
        }
        for i, doc in enumerate(documents)
    ]

    print(f"\nUploading {len(vectors)} vectors in batches of {UPSERT_BATCH_SIZE}...")

    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i : i + UPSERT_BATCH_SIZE]

        index.upsert(vectors=batch)

        print(f"Uploaded {i} - {i + len(batch)}")

    print("All vectors uploaded.")


def main():
    if not PINECONE_API_KEY:
        sys.exit("PINECONE_API_KEY was not found. Add it to your .env file.")

    print("=" * 70)
    print("GUARDIANAI — KNOWLEDGE BASE INGESTION")
    print("=" * 70)

    documents = load_documents()

    index = get_pinecone_index()

    confirm_upsert_if_nonempty(index)

    embeddings = embed_documents(documents)

    upsert_documents(index, documents, embeddings)

    final_count = index.describe_index_stats()["total_vector_count"]
    print(f"\nFinal vector count in '{INDEX_NAME}': {final_count}")


if __name__ == "__main__":
    main()
