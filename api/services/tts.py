"""TTS service — edge-tts (Microsoft Edge voices) for high-quality Indian language speech."""

import asyncio
import io
import logging
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# High-quality Microsoft Edge voices for Indian languages
EDGE_VOICES = {
    "en": "en-IN-NeerjaExpressiveNeural",   # Indian English female
    "hi": "hi-IN-SwaraNeural",               # Hindi female
    "mr": "mr-IN-AarohiNeural",              # Marathi female
    "gu": "gu-IN-DhwaniNeural",              # Gujarati female
    "ta": "ta-IN-PallaviNeural",             # Tamil female
    "te": "te-IN-ShrutiNeural",              # Telugu female
    "kn": "kn-IN-SapnaNeural",              # Kannada female
    "ml": "ml-IN-SobhanaNeural",            # Malayalam female
    "bn": "bn-IN-TanishaaNeural",           # Bengali female
    "pa": "hi-IN-SwaraNeural",              # Punjabi → Hindi voice (closest)
    "ur": "ur-IN-GulNeural",                # Urdu female
}

# Fallback gTTS language map
GTTS_LANG_MAP = {
    "en": "en", "hi": "hi", "mr": "mr", "gu": "gu",
    "ta": "ta", "te": "te", "kn": "kn", "ml": "ml",
    "bn": "bn", "pa": "hi", "ur": "hi",
}

# Sentence split regex — handles Devanagari, Tamil, Telugu, etc.
SENT_SPLIT = re.compile(r'(?<=[।॥.!?।\n])\s+')


def split_sentences(text: str, max_chars: int = 500) -> list[str]:
    """Split text into sentence chunks for TTS."""
    raw = SENT_SPLIT.split(text.strip())
    chunks, current = [], ""
    for sentence in raw:
        if len(current) + len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks or [text]


def _run_async(coro):
    """Run an async function from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context (FastAPI) — run in a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=30)
    else:
        return asyncio.run(coro)


class TTSService:
    def __init__(self):
        self._edge_available = self._check_edge_tts()
        self._gtts_available = self._check_gtts()
        if self._edge_available:
            log.info("TTS backend: edge-tts (Microsoft Neural voices)")
        elif self._gtts_available:
            log.info("TTS backend: gTTS (Google fallback)")

    def _check_edge_tts(self) -> bool:
        try:
            import edge_tts  # noqa
            return True
        except ImportError:
            log.warning("edge-tts not installed")
            return False

    def _check_gtts(self) -> bool:
        try:
            import gtts  # noqa
            return True
        except ImportError:
            log.warning("gTTS not installed — TTS will be limited")
            return False

    async def _edge_synthesize(self, text: str, language: str) -> bytes:
        """Synthesize using edge-tts (async)."""
        import edge_tts

        voice = EDGE_VOICES.get(language, EDGE_VOICES["hi"])
        communicate = edge_tts.Communicate(text, voice)

        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        return b"".join(audio_chunks)

    def _gtts_synthesize(self, text: str, language: str) -> bytes:
        """Fallback: gTTS synthesis."""
        from gtts import gTTS
        lang_code = GTTS_LANG_MAP.get(language, "hi")
        tts_obj = gTTS(text=text, lang=lang_code, slow=False)
        mp3_buf = io.BytesIO()
        tts_obj.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        return mp3_buf.read()

    @lru_cache(maxsize=200)
    def _synthesize_cached(self, text: str, language: str) -> bytes:
        return self._synthesize_uncached(text, language)

    def _synthesize_uncached(self, text: str, language: str) -> bytes:
        # Try edge-tts first (high quality neural voices)
        if self._edge_available:
            try:
                return _run_async(self._edge_synthesize(text, language))
            except Exception as e:
                log.warning(f"edge-tts failed: {e}. Falling back to gTTS.")

        # gTTS fallback
        if self._gtts_available:
            return self._gtts_synthesize(text, language)

        raise RuntimeError("No TTS backend available")

    def synthesize(self, text: str, language: str) -> bytes:
        """Convert text to MP3 audio bytes. Handles long texts by splitting."""
        text = text.strip()
        if not text:
            return b""

        # For short texts use cache
        if len(text) <= 500:
            return self._synthesize_cached(text, language)

        # For long texts: split into sentences and concatenate
        sentences = split_sentences(text)
        audio_parts = []
        for sentence in sentences:
            if sentence.strip():
                part = self._synthesize_cached(sentence.strip(), language)
                audio_parts.append(part)

        if not audio_parts:
            return b""
        if len(audio_parts) == 1:
            return audio_parts[0]

        # Concatenate MP3 bytes directly (MP3 frames are self-contained)
        return b"".join(audio_parts)


_tts_instance: Optional[TTSService] = None


def get_tts() -> TTSService:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSService()
    return _tts_instance
