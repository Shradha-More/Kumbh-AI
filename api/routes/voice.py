"""Voice routes — full voice pipeline, STT-only, TTS-only, WebSocket streaming."""

import base64
import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from api.models.schemas import STTResponse, TTSRequest, VoiceInputResponse
from api.services.asr import get_asr, ASRService
from api.services.tts import get_tts, TTSService
from api.services.rag import get_rag, RAGService
from api.services.llm import get_llm, LLMService
from api.routes.query import detect_language

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/input", response_model=VoiceInputResponse)
async def voice_input(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    asr: ASRService = Depends(get_asr),
    tts: TTSService = Depends(get_tts),
    rag: RAGService = Depends(get_rag),
    llm: LLMService = Depends(get_llm),
):
    """Full voice pipeline: audio → text → LLM → audio response."""
    t0 = time.time()

    # 1. Read audio
    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1] if audio.filename else "m4a"

    # 2. Transcribe
    hint = language if language and language != "auto" else None
    asr_result = asr.transcribe(audio_bytes, hint_language=hint, audio_format=ext)
    transcript = asr_result["text"]

    if not transcript.strip():
        return VoiceInputResponse(
            audio_base64="",
            transcript="",
            response_text="I couldn't hear anything clearly. Please try again.",
            language=language or "en",
            duration_ms=int((time.time() - t0) * 1000),
        )

    # 3. Detect language:
    #    - User explicitly chose a language → trust that
    #    - Auto mode → use script detection on transcript (Devanagari, Tamil, etc.)
    #      with Whisper's guess as a fallback for Latin-script languages
    if language and language != "auto":
        detected_lang = language
    else:
        script_lang = detect_language(transcript, hint=None)
        whisper_lang = asr_result["language"]
        whisper_conf = asr_result.get("confidence", 0)
        # Script detection is reliable for non-Latin scripts
        # For Latin script (English), use Whisper if confident, else default to English
        if script_lang != "en":
            detected_lang = script_lang
        elif whisper_conf > 0.7:
            detected_lang = whisper_lang
        else:
            detected_lang = script_lang

    # 4. RAG retrieval — also search English as fallback for better context
    docs = rag.retrieve(transcript, language=detected_lang, top_k=3)
    if len(docs) < 2 and detected_lang != "en":
        docs += rag.retrieve(transcript, language="en", top_k=2)
    context_texts = [d["text"] for d in docs]

    # 5. LLM generation
    response_text = llm.generate(
        query=transcript,
        context=context_texts,
        language=detected_lang,
    )

    # 6. Detect language of response for correct TTS voice
    response_lang = detect_language(response_text, hint=detected_lang)

    # 7. TTS
    audio_out = tts.synthesize(response_text, response_lang)
    audio_b64 = base64.b64encode(audio_out).decode("utf-8") if audio_out else ""

    return VoiceInputResponse(
        audio_base64=audio_b64,
        transcript=transcript,
        response_text=response_text,
        language=response_lang,
        duration_ms=int((time.time() - t0) * 1000),
    )


@router.post("/tts")
async def text_to_speech(
    req: TTSRequest,
    tts: TTSService = Depends(get_tts),
):
    """Convert text to speech — returns MP3 audio directly."""
    audio_bytes = tts.synthesize(req.text, req.language)
    return Response(content=audio_bytes, media_type="audio/mpeg")


@router.post("/stt", response_model=STTResponse)
async def speech_to_text(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    asr: ASRService = Depends(get_asr),
):
    """Convert speech to text — no LLM, just transcription."""
    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1] if audio.filename else "m4a"
    result = asr.transcribe(audio_bytes, hint_language=language, audio_format=ext)
    return STTResponse(
        transcript=result["text"],
        language=result["language"],
        confidence=result["confidence"],
    )


@router.websocket("/ws/voice")
async def voice_websocket(
    websocket: WebSocket,
    asr: ASRService = Depends(get_asr),
    tts: TTSService = Depends(get_tts),
    rag: RAGService = Depends(get_rag),
    llm: LLMService = Depends(get_llm),
):
    """
    WebSocket for real-time voice interaction.

    Client sends: JSON {"type": "audio", "data": "<base64 audio>", "language": "hi", "format": "m4a"}
    Server sends:
      {"type": "transcript", "text": "...", "language": "..."}
      {"type": "token", "text": "..."}   (streaming LLM tokens)
      {"type": "audio", "data": "<base64 mp3>"}
      {"type": "done"}
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "audio":
                continue

            audio_b64 = msg.get("data", "")
            language_hint = msg.get("language")
            audio_format = msg.get("format", "m4a")

            # Decode audio
            audio_bytes = base64.b64decode(audio_b64)

            # Transcribe
            hint = language_hint if language_hint and language_hint != "auto" else None
            asr_result = asr.transcribe(
                audio_bytes, hint_language=hint, audio_format=audio_format
            )
            transcript = asr_result["text"]

            # Detect language
            if language_hint and language_hint != "auto":
                lang = language_hint
            else:
                script_lang = detect_language(transcript, hint=None)
                whisper_lang = asr_result["language"]
                whisper_conf = asr_result.get("confidence", 0)
                if script_lang != "en":
                    lang = script_lang
                elif whisper_conf > 0.7:
                    lang = whisper_lang
                else:
                    lang = script_lang

            await websocket.send_json({"type": "transcript", "text": transcript, "language": lang})

            if not transcript.strip():
                await websocket.send_json({"type": "done"})
                continue

            # RAG — also search English for better context
            docs = rag.retrieve(transcript, language=lang, top_k=3)
            if len(docs) < 2 and lang != "en":
                docs += rag.retrieve(transcript, language="en", top_k=2)
            context_texts = [d["text"] for d in docs]

            # Stream LLM tokens
            full_response = []
            async for token in llm.generate_stream(transcript, context_texts, lang):
                full_response.append(token)
                await websocket.send_json({"type": "token", "text": token})

            response_text = "".join(full_response)

            # TTS — detect response language for correct voice
            response_lang = detect_language(response_text, hint=lang)
            audio_out = tts.synthesize(response_text, response_lang)
            if audio_out:
                audio_b64_out = base64.b64encode(audio_out).decode("utf-8")
                await websocket.send_json({"type": "audio", "data": audio_b64_out})

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
