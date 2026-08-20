import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from harness import VoiceRAGHarness
import requests

load_dotenv()

app = FastAPI(title="HH Goa 2026 - Voice RAG Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

harness = VoiceRAGHarness()
SARVAM_KEY = os.getenv("SARVAM_API_KEY", "")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/voice-query")
async def voice_query_endpoint(file: UploadFile = File(...)):
    """Receives voice audio, transcribes with Sarvam AI, and executes RAG harness."""
    if not SARVAM_KEY:
        raise HTTPException(
            status_code=500, detail="SARVAM_API_KEY is not configured in .env"
        )

    audio_bytes = await file.read()

    # Sarvam Speech-to-Text API Call
    headers = {"api-subscription-key": SARVAM_KEY}
    files = {
        "file": (
            file.filename or "audio.wav",
            audio_bytes,
            file.content_type or "audio/wav",
        )
    }
    payload = {
        "model": "saaras:v3",
        "language_code": "unknown",  # Auto-detects Indic/English
        "mode": "transcribe",
    }

    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers=headers,
            files=files,
            data=payload,
            timeout=8,
        )
        resp_json = resp.json()
        transcript = resp_json.get("transcript", "").strip()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Sarvam STT failed: {str(e)}"
        )

    if not transcript:
        return {
            "transcript": "",
            "rag": {
                "query": "",
                "answer": "Could not recognize audio. Please try again.",
                "grounded": False,
                "retrieval_score": 0.0,
                "total_latency_ms": 0.0,
                "retrieval_latency_ms": 0.0,
                "inference_latency_ms": 0.0,
            },
        }

    rag_result = harness.execute(transcript)
    return {
        "transcript": transcript,
        "rag": rag_result.model_dump(),
    }


@app.post("/api/text-query")
async def text_query_endpoint(query: str):
    """Direct text query endpoint for benchmarking."""
    return harness.execute(query)


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)