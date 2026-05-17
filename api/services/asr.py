"""ASR service — Groq Whisper API (fast, accurate) with local faster-whisper fallback."""

import io
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_ASR_MODEL = "whisper-large-v3"
GROQ_ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class ASRService:
    def __init__(self, model_size: str = WHISPER_MODEL, device: str = "auto"):
        self._local_model = None

        # Try Groq first — no local model needed
        if GROQ_API_KEY:
            log.info("ASR backend: Groq Whisper Large v3 (cloud, fast)")
        else:
            log.info("No GROQ_API_KEY — loading local Whisper...")
            self._load_local(model_size, device)

    def _load_local(self, model_size: str, device: str):
        """Load local faster-whisper as fallback."""
        try:
            log.info(f"Loading Whisper {model_size}...")
            from faster_whisper import WhisperModel
            import torch

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            self._local_model = WhisperModel(model_size, device=device, compute_type=compute)
            log.info("Whisper ready")
        except Exception as e:
            log.warning(f"Could not load local Whisper: {e}")

    def _convert_to_wav(self, audio_bytes: bytes, src_format: str = "m4a") -> bytes:
        """Convert any audio format to 16kHz mono WAV using ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix=f".{src_format}", delete=False) as inp:
            inp.write(audio_bytes)
            inp_path = inp.name

        out_path = inp_path.replace(f".{src_format}", ".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", inp_path, "-ar", "16000", "-ac", "1", out_path],
                capture_output=True, check=True,
            )
            with open(out_path, "rb") as f:
                return f.read()
        finally:
            Path(inp_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

    def transcribe(
        self,
        audio_bytes: bytes,
        hint_language: Optional[str] = None,
        audio_format: str = "m4a",
    ) -> dict:
        """Transcribe audio bytes. Returns {text, language, confidence, segments}."""

        # Try Groq cloud ASR first (Whisper Large v3 — much better quality)
        if GROQ_API_KEY:
            result = self._groq_transcribe(audio_bytes, hint_language, audio_format)
            if result:
                return result
            log.warning("Groq ASR failed, falling back to local Whisper")

        # Fallback: local faster-whisper
        if self._local_model is None:
            self._load_local(WHISPER_MODEL, "auto")

        if self._local_model:
            return self._local_transcribe(audio_bytes, hint_language, audio_format)

        return {"text": "", "language": hint_language or "en", "confidence": 0, "segments": []}

    def _groq_transcribe(
        self, audio_bytes: bytes, hint_language: Optional[str], audio_format: str
    ) -> Optional[dict]:
        """Transcribe via Groq Whisper API (large-v3, fast and accurate)."""
        try:
            # Convert to wav for consistent format
            wav_bytes = self._convert_to_wav(audio_bytes, audio_format)

            # Build multipart form
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            data = {
                "model": GROQ_ASR_MODEL,
                "response_format": "verbose_json",
                "temperature": 0.0,
            }
            if hint_language and hint_language != "auto":
                data["language"] = hint_language

            resp = httpx.post(
                GROQ_ASR_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files=files,
                data=data,
                timeout=30.0,
            )
            resp.raise_for_status()
            result = resp.json()

            text = result.get("text", "").strip()
            language = result.get("language", hint_language or "en")

            # Map full language names to ISO codes + fix common misdetections
            lang_map = {
                "hindi": "hi", "marathi": "mr", "gujarati": "gu",
                "tamil": "ta", "telugu": "te", "kannada": "kn",
                "malayalam": "ml", "english": "en",
                "urdu": "hi",      # Urdu/Hindi are same spoken language
                "nepali": "hi",    # Often confused with Hindi
                "sindhi": "hi",    # Often confused with Hindi
                "punjabi": "hi",   # Fallback to Hindi
                "bengali": "hi",   # Fallback to Hindi for Kumbh context
            }
            if language in lang_map:
                language = lang_map[language]
            # Also handle ISO codes
            iso_fix = {"ur": "hi", "ne": "hi", "sd": "hi", "pa": "hi", "bn": "hi"}
            language = iso_fix.get(language, language)

            segments = [
                {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
                for s in result.get("segments", [])
            ]

            log.info(f"Groq ASR: lang={language}, text={text[:80]}...")
            return {
                "text": text,
                "language": language,
                "confidence": 0.95,  # Large v3 is highly reliable
                "segments": segments,
            }
        except Exception as e:
            log.warning(f"Groq ASR error: {e}")
            return None

    def _local_transcribe(
        self, audio_bytes: bytes, hint_language: Optional[str], audio_format: str
    ) -> dict:
        """Transcribe using local faster-whisper."""
        wav_bytes = self._convert_to_wav(audio_bytes, audio_format)

        VALID_WHISPER_CODES = {
            "hi", "mr", "gu", "ta", "te", "kn", "ml", "en",
            "bn", "pa", "ur", "sa", "ne", "si",
        }
        whisper_lang = None
        if hint_language and hint_language in VALID_WHISPER_CODES:
            whisper_lang = hint_language

        import numpy as np
        import io as _io
        import wave

        with wave.open(_io.BytesIO(wav_bytes)) as wf:
            frames = wf.readframes(wf.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = self._local_model.transcribe(
            audio_np,
            language=whisper_lang,
            beam_size=5,
            vad_filter=True,
        )

        segments_list = list(segments)
        text = " ".join(s.text.strip() for s in segments_list).strip()
        confidence = float(info.language_probability) if hasattr(info, "language_probability") else 0.9
        detected_lang = info.language if hasattr(info, "language") else (hint_language or "hi")

        return {
            "text": text,
            "language": detected_lang,
            "confidence": round(confidence, 3),
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments_list
            ],
        }

    def transcribe_file(self, file_path: str) -> dict:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        ext = Path(file_path).suffix.lstrip(".")
        return self.transcribe(audio_bytes, audio_format=ext)


_asr_instance: Optional[ASRService] = None


def get_asr() -> ASRService:
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASRService()
    return _asr_instance
