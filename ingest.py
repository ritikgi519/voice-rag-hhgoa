import os

# Set local cache and disable symlink warnings on Windows
os.environ["HF_HOME"] = os.path.abspath("./hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs("./hf_cache", exist_ok=True)

import uuid
from datasets import load_dataset
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION_NAME = "msmarco_xi"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
DB_PATH = "./qdrant_db"


def chunk_sliding_window(text: str, chunk_size: int = 40, overlap: int = 10):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def run_ingestion(record_limit: int = 350):
    print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Connecting to local vector store at {DB_PATH}...")
    client = QdrantClient(path=DB_PATH)

    # Clean collection setup
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    print("Streaming 'default' split from ai4bharat/MSMARCO-XI...")
    # Using 'default' builder config
    dataset = load_dataset(
        "ai4bharat/MSMARCO-XI", "default", split="validation", streaming=True
    )

    points = []
    processed_queries = 0

    for item in dataset:
        if processed_queries >= record_limit:
            break

        q_id = item.get("query_id", processed_queries)
        q_type = item.get("query_type", "GENERAL")
        passages_data = item.get("passages", {})

        # Extract passages (fallback to English if translated is absent)
        trans_passages = passages_data.get(
            "Translated_passages", []
        ) or passages_data.get("English_passages", [])
        is_selected_flags = passages_data.get("is_selected", [])

        if not trans_passages and "passage_text" in item:
            trans_passages = [item["passage_text"]]

        for idx, passage in enumerate(trans_passages):
            if not passage or len(passage.strip()) < 15:
                continue

            is_gold = (
                bool(is_selected_flags[idx])
                if idx < len(is_selected_flags)
                else False
            )
            child_chunks = chunk_sliding_window(
                passage, chunk_size=35, overlap=10
            )

            for c_idx, child in enumerate(child_chunks):
                vec = embedder.encode(
                    f"passage: {child}", normalize_embeddings=True
                ).tolist()

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={
                            "child_chunk": child,
                            "parent_passage": passage,
                            "query_id": q_id,
                            "query_type": q_type,
                            "is_selected": is_gold,
                        },
                    )
                )

        processed_queries += 1
        if processed_queries % 50 == 0:
            print(f"Processed {processed_queries}/{record_limit} queries...")

    print(f"Upserting {len(points)} vectors into Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Ingestion complete successfully!")


if __name__ == "__main__":
    run_ingestion(record_limit=300)