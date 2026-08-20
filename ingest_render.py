import os
import time
import requests
from dotenv import load_dotenv
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/intfloat/multilingual-e5-small"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=15,
    )
    if response.status_code == 200:
        res = response.json()
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            return res[0]
        return res
    raise RuntimeError(f"HF Inference API Error ({response.status_code}): {response.text}")


def run_ingest():
    db_path = "./qdrant_db"
    os.makedirs(db_path, exist_ok=True)

    client = QdrantClient(path=db_path)
    collection_name = "msmarco_xi"

    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return

    print("Creating Qdrant collection...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    print("Fetching dataset sample from Hugging Face...")
    dataset = load_dataset("castorini/msmarco-hindi-100k", split="train", streaming=True)

    points = []
    # Index 100 sample documents via API
    for idx, item in enumerate(dataset.take(100)):
        passage = item.get("passage", "")
        if not passage:
            continue
        try:
            vector = get_embedding(f"passage: {passage}")
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "parent_passage": passage,
                        "text": passage,
                        "id": str(item.get("id", idx)),
                    },
                )
            )
            time.sleep(0.05)  # slight pause for API rate limits
        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")

    client.upsert(collection_name=collection_name, points=points)
    print(f"Ingested {len(points)} records into Render Qdrant instance successfully.")


if __name__ == "__main__":
    run_ingest()