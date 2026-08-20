import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

def run_ingest():
    db_path = "./qdrant_db"
    os.makedirs(db_path, exist_ok=True)
    
    client = QdrantClient(path=db_path)
    collection_name = "msmarco_hindi"
    
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return

    print("Loading embedding model on CPU...")
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    print("Loading dataset sample...")
    dataset = load_dataset("castorini/msmarco-hindi-100k", split="train", streaming=True)
    
    points = []
    # Index 300 passages to keep memory under 300MB
    for idx, item in enumerate(dataset.take(300)):
        passage = item.get("passage", "")
        if not passage:
            continue
        text_to_embed = f"passage: {passage}"
        vector = model.encode(text_to_embed, normalize_embeddings=True).tolist()
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={"text": passage, "id": item.get("id", str(idx))}
            )
        )
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"Ingested {len(points)} records into Render Qdrant instance successfully.")

if __name__ == "__main__":
    run_ingest()
    