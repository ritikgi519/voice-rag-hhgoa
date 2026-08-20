import os
import time
import requests
from dotenv import load_dotenv
from groq import Groq
from guardrails import GuardrailEngine
from pydantic import BaseModel
from qdrant_client import QdrantClient

load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/intfloat/multilingual-e5-small"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}


def get_embedding(text: str) -> list[float]:
    """Generates normalized embeddings via HF Inference API without local PyTorch."""
    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=10,
    )
    if response.status_code == 200:
        res = response.json()
        # Handle single vs batch response shapes
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            return res[0]
        return res
    raise RuntimeError(f"HF Inference Error ({response.status_code}): {response.text}")


class RAGResult(BaseModel):
    query: str
    answer: str
    grounded: bool
    retrieval_score: float
    total_latency_ms: float
    retrieval_latency_ms: float
    inference_latency_ms: float


class VoiceRAGHarness:

    def __init__(self, db_path: str = "./qdrant_db"):
        self.qdrant = QdrantClient(path=db_path)
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.collection = "msmarco_xi"
        self.guardrails = GuardrailEngine()

    def execute(self, query: str) -> RAGResult:
        t_start = time.perf_counter()

        # 1. Input Guardrail
        is_safe, msg = self.guardrails.validate_input(query)
        if not is_safe:
            return RAGResult(
                query=query,
                answer=f"Declined: {msg}",
                grounded=False,
                retrieval_score=0.0,
                total_latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
                retrieval_latency_ms=0.0,
                inference_latency_ms=0.0,
            )

        # 2. Vector Retrieval (API Embedding)
        t_ret_start = time.perf_counter()
        try:
            q_vec = get_embedding(f"query: {query}")
        except Exception as e:
            return RAGResult(
                query=query,
                answer=f"Embedding generation error: {str(e)}",
                grounded=False,
                retrieval_score=0.0,
                total_latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
                retrieval_latency_ms=0.0,
                inference_latency_ms=0.0,
            )

        if hasattr(self.qdrant, "query_points"):
            search_res = self.qdrant.query_points(
                collection_name=self.collection, query=q_vec, limit=2
            )
            hits = search_res.points
        else:
            hits = self.qdrant.search(
                collection_name=self.collection, query_vector=q_vec, limit=2
            )

        t_ret_end = time.perf_counter()
        ret_ms = round((t_ret_end - t_ret_start) * 1000, 2)

        top_score = hits[0].score if hits else 0.0

        # 3. Context Similarity Gating
        if not hits or not self.guardrails.check_retrieval_confidence(
            top_score, threshold=0.68
        ):
            return RAGResult(
                query=query,
                answer="I do not have sufficient information in the dataset to answer this question accurately.",
                grounded=False,
                retrieval_score=round(top_score, 4),
                total_latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
                retrieval_latency_ms=ret_ms,
                inference_latency_ms=0.0,
            )

        # Parent Passage Extraction
        parent_passages = list({h.payload.get("parent_passage", h.payload.get("text", "")) for h in hits})
        context_block = "\n---\n".join(parent_passages)

        # 4. Low-Latency LLM Inference via Groq
        t_inf_start = time.perf_counter()
        try:
            completion = self.groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict, low-latency QA system. Answer concisely in at most 2 sentences "
                            "using ONLY the provided context. If unsure or ungrounded, state that information is insufficient.\n\n"
                            f"Context:\n{context_block}"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=60,
                temperature=0.0,
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            answer = f"Inference error: {str(e)}"

        t_inf_end = time.perf_counter()
        inf_ms = round((t_inf_end - t_inf_start) * 1000, 2)

        # 5. Output Groundedness Check
        grounded = self.guardrails.verify_groundedness(answer, context_block)
        if not grounded:
            answer = "I do not have sufficient verifiable information in the dataset to answer this."

        total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return RAGResult(
            query=query,
            answer=answer,
            grounded=grounded,
            retrieval_score=round(top_score, 4),
            total_latency_ms=total_ms,
            retrieval_latency_ms=ret_ms,
            inference_latency_ms=inf_ms,
        )