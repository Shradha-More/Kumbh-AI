"""
Nashik Kumbh Mela 2027 — Multilingual AI Assistant API
FastAPI backend — voice in, voice out, multilingual RAG, emergency, places.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1

Endpoints:
    POST /api/v1/query            — text query
    POST /api/v1/query/stream     — SSE streaming
    GET  /api/v1/query/history/{session_id}
    POST /api/v1/voice/input      — voice → voice (full pipeline)
    POST /api/v1/voice/tts        — text → speech
    POST /api/v1/voice/stt        — speech → text
    WS   /ws/voice                — streaming voice WebSocket
    WS   /ws/chat                 — streaming text WebSocket
    POST /api/v1/emergency        — emergency response (hardcoded, fast)
    GET  /api/v1/emergency/contacts
    GET  /api/v1/emergency/nearest
    GET  /api/v1/places           — tourist places list
    GET  /api/v1/places/{id}      — place detail
    GET  /api/v1/places/nearby    — places near coordinates
    POST /api/v1/places/recommend — itinerary planner
    GET  /api/v1/health
"""

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load .env before any service reads os.environ

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import query, voice, emergency, places, feedback, auth, sessions
from api.models.database import db
from api.services.rag import get_rag
from api.services.llm import get_llm
from api.services.asr import get_asr
from api.services.tts import get_tts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models on startup. Heavy tasks run in background to avoid HF timeout."""
    import threading
    log.info("=== Nashik Kumbh Mela 2027 AI Backend starting ===")
    log.info("Initializing database...")
    await db.init_db()

    # Load RAG (fast — just opens ChromaDB)
    log.info("Loading RAG / ChromaDB...")
    rag = get_rag()
    doc_count = rag.doc_count()
    log.info(f"ChromaDB has {doc_count} documents")

    # Load TTS (instant)
    log.info("Loading TTS...")
    get_tts()

    log.info("=== Server ready (basic). Loading heavy models in background... ===")

    # Heavy loading in background thread (LLM download + Whisper + ChromaDB build)
    def background_init():
        import subprocess, sys

        # Build ChromaDB if empty
        if doc_count == 0:
            log.info("[BG] ChromaDB empty — building from knowledge base...")
            try:
                subprocess.run(
                    [sys.executable, "vectordb/ingest_chunks.py"],
                    cwd=str(Path(__file__).parent.parent),
                    timeout=3600,
                )
                get_rag(force_reload=True)
                log.info("[BG] ChromaDB built successfully")
            except Exception as e:
                log.error(f"[BG] ChromaDB build failed: {e}")

        # Load LLM (may download 1.9GB GGUF on first run)
        log.info("[BG] Loading LLM...")
        try:
            get_llm()
            log.info("[BG] LLM ready")
        except Exception as e:
            log.error(f"[BG] LLM failed: {e}")

        # Load Whisper ASR
        log.info("[BG] Loading Whisper ASR...")
        try:
            get_asr()
            log.info("[BG] Whisper ready")
        except Exception as e:
            log.error(f"[BG] Whisper failed: {e}")

        log.info("[BG] === All models loaded. Fully operational. ===")

    threading.Thread(target=background_init, daemon=True).start()
    log.info("=== All models loaded. API ready. ===")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Nashik Kumbh Mela 2027 — AI Multilingual Assistant",
    description="Voice-first multilingual AI assistant for Nashik Kumbh Mela 2027. "
                "Supports Hindi, Marathi, Gujarati, Tamil, Telugu, Kannada, Malayalam, English.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React Native app on all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(query.router,     prefix="/api/v1")
app.include_router(voice.router,     prefix="/api/v1")
app.include_router(emergency.router, prefix="/api/v1")
app.include_router(places.router,    prefix="/api/v1")
app.include_router(feedback.router,  prefix="/api/v1")
app.include_router(auth.router,      prefix="/api/v1")
app.include_router(sessions.router,  prefix="/api/v1")

# ── WebSocket chat (text streaming) ───────────────────────────────────────────
@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """
    WebSocket streaming chat.
    Client sends: {"query": "...", "language": "hi", "session_id": "..."}
    Server sends tokens: {"type": "token", "text": "..."} then {"type": "done"}
    """
    import json as _json
    await websocket.accept()
    rag = get_rag()
    llm = get_llm()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = _json.loads(raw)
            query_text = msg.get("query", "")
            language = msg.get("language", "en")

            docs = rag.retrieve(query_text, language=language, top_k=5)
            context_texts = [d["text"] for d in docs]

            async for token in llm.generate_stream(query_text, context_texts, language):
                await websocket.send_json({"type": "token", "text": token})

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WS error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/api/v1/health")
async def health():
    llm = get_llm()
    rag = get_rag()
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = round(torch.cuda.memory_allocated() / 1e9, 2)

    return {
        "status": "ok",
        "model": llm.model_name,
        "backend": llm.backend,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "gpu_memory_used_gb": gpu_mem,
        "total_documents": rag.doc_count(),
        "version": "1.0.0",
    }


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse({"detail": "Not found"}, status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    log.error(f"500 error: {exc}")
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


STATIC_DIR = Path(__file__).parent.parent / "static"

@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {
        "name": "Nashik Kumbh Mela 2027 AI Assistant",
        "version": "1.0.0",
        "languages": ["en", "hi", "mr", "gu", "ta", "te", "kn", "ml"],
        "docs": "/docs",
    }

# Serve static assets (CSS/JS/images if added later)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
