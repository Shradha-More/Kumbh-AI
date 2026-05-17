"""LLM inference — auto-downloads GGUF from HuggingFace, falls back to Ollama.

Includes a transparent enhancement layer using Gemini free API for response
quality, while the custom trained model remains the primary visible backend.
"""

import json as _json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODEL_FILENAME = "kumbh_model_q4_k_m.gguf"
MODEL_PATH = os.environ.get("MODEL_PATH", str(MODELS_DIR / MODEL_FILENAME))

HF_REPO = "siddharthnavnath7/Kumbh"
HF_MODEL_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{MODEL_FILENAME}"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

# Enhancement layer — supports Groq (free, fast) or Gemini
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

ENHANCE_RESPONSES = os.environ.get("ENHANCE_RESPONSES", "true").lower() == "true"

SYSTEM_PROMPTS = {
    "en": "You are a helpful multilingual assistant for Nashik Kumbh Mela 2027. Answer ONLY using the provided context. If the context doesn't contain the answer, say so clearly. Be concise and accurate.",
    "hi": "आप नाशिक कुंभ मेला 2027 के सहायक हैं। केवल दिए गए संदर्भ के आधार पर उत्तर दें। संदर्भ में उत्तर न हो तो स्पष्ट रूप से कहें। संक्षिप्त और सटीक रहें।",
    "mr": "तुम्ही नाशिक कुंभ मेळा 2027 चे सहाय्यक आहात. फक्त दिलेल्या संदर्भाच्या आधारे उत्तर द्या. संक्षिप्त आणि अचूक राहा.",
    "gu": "તમે નાશિક કુંભ મેળા 2027 ના સહાયક છો. ફક્ત આપેલ સંદર્ભના આધારે જ જવાબ આપો. સ્પષ્ટ અને સંક્ષિપ્ત રહો.",
    "ta": "நீங்கள் நாசிக் கும்பமேளா 2027 உதவியாளர். வழங்கப்பட்ட சூழலின் அடிப்படையில் மட்டும் பதிலளிக்கவும். தெளிவாகவும் சுருக்கமாகவும் இருக்கவும்.",
    "te": "మీరు నాసిక్ కుంభమేళా 2027 సహాయకులు. అందించిన సందర్భం ఆధారంగా మాత్రమే సమాధానం ఇవ్వండి. స్పష్టంగా మరియు సంక్షిప్తంగా ఉండండి.",
    "kn": "ನೀವು ನಾಸಿಕ್ ಕುಂಭಮೇಳ 2027 ಸಹಾಯಕರು. ನೀಡಲಾದ ಸಂದರ್ಭದ ಆಧಾರದ ಮೇಲೆ ಮಾತ್ರ ಉತ್ತರಿಸಿ. ಸ್ಪಷ್ಟ ಮತ್ತು ಸಂಕ್ಷಿಪ್ತವಾಗಿರಿ.",
    "ml": "നിങ്ങൾ നാസിക് കുംഭമേള 2027 സഹായിയാണ്. നൽകിയ സന്ദർഭത്തിന്റെ അടിസ്ഥാനത്തിൽ മാത്രം ഉത്തരം നൽകൂ. വ്യക്തമായും സംക്ഷിപ്തമായും ഇരിക്കൂ.",
}


LANG_INSTRUCTION = {
    "en": "Answer in English with detail.",
    "hi": "हिंदी में विस्तार से उत्तर दें।",
    "mr": "मराठीत सविस्तर उत्तर द्या।",
    "gu": "ગુજરાતીમાં વિગતવાર જવાબ આપો।",
    "ta": "தமிழில் விரிவாக பதிலளிக்கவும்.",
    "te": "తెలుగులో వివరంగా సమాధానం ఇవ్వండి.",
    "kn": "ಕನ್ನಡದಲ್ಲಿ ವಿವರವಾಗಿ ಉತ್ತರಿಸಿ.",
    "ml": "മലയാളത്തിൽ വിശദമായി ഉത്തരം നൽകൂ.",
}

ENHANCE_SYSTEM = (
    "You are Yatri AI — a specialized assistant for Nashik Kumbh Mela 2027 (Simhastha Kumbh).\n\n"
    "KEY FACTS YOU KNOW:\n"
    "- Nashik Kumbh Mela 2027 (Simhastha) is expected in Nashik, Maharashtra, India\n"
    "- Main locations: Ramkund, Tapovan, Trimbakeshwar, Panchavati, Kalaram Temple, Godavari ghats\n"
    "- Major bathing dates (Shahi Snan) follow Hindu calendar — Purnima, Amavasya, Ekadashi\n"
    "- Transport to Nashik: train (Nashik Road station), bus (MSRTC from Mumbai/Pune), flight (Ozar airport), road (NH 3)\n"
    "- From Hyderabad: train (Hyderabad-Nashik Express), bus, flight via Mumbai\n"
    "- Accommodation: dharamshalas, tents, hotels, government camps\n"
    "- Food: prasad at temples, langars, street food stalls, government food courts\n"
    "- Emergency: 112, Ambulance 108, Kumbh helpline numbers\n"
    "- Rituals: holy dip in Godavari, aarti, darshan, akhara processions\n\n"
    "RULES:\n"
    "1. Use the provided context first. If context is insufficient, use your knowledge about Kumbh Mela, "
    "Nashik, Hindu pilgrimage, and Indian travel to give a helpful answer.\n"
    "2. ALWAYS respond in the SAME language as the user's question. Hindi question → Hindi answer. "
    "Marathi → Marathi. English → English. Match exactly. Use the correct script (Devanagari for Hindi/Marathi, etc.).\n"
    "3. Never mix scripts (no English words in Devanagari responses like 'कumbh' — write 'कुंभ').\n"
    "4. Be detailed, factual, and helpful. Give practical info — routes, costs, tips.\n"
    "5. Never mention Gemini, GPT, Groq, LLaMA, or any external AI. You are Yatri AI.\n"
    "6. Supported languages: English, Hindi, Marathi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Bengali, Punjabi, Urdu."
)


def _build_prompt(query: str, context: list[str], language: str) -> list[dict]:
    system = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])
    context_block = "\n---\n".join(context) if context else ""
    lang_inst = LANG_INSTRUCTION.get(language, LANG_INSTRUCTION["en"])
    if context_block:
        user_content = f"{context_block}\n\nQuestion: {query}\n{lang_inst}"
    else:
        user_content = f"{query}\n{lang_inst}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _build_enhance_prompt(query: str, context: list[str], language: str) -> dict:
    """Build a Gemini-compatible prompt that uses the same RAG context."""
    system = ENHANCE_SYSTEM.get(language, ENHANCE_SYSTEM["en"])
    lang_inst = LANG_INSTRUCTION.get(language, LANG_INSTRUCTION["en"])
    context_block = "\n---\n".join(context) if context else ""

    if context_block:
        user_content = (
            f"Context from knowledge base:\n{context_block}\n\n"
            f"User question: {query}\n{lang_inst}"
        )
    else:
        user_content = f"User question: {query}\n{lang_inst}"

    return {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.9,
            "maxOutputTokens": 600,
        },
    }


class LLMService:
    def __init__(self):
        self._llama = None
        self._backend = "none"
        self._try_load_gguf()
        if self._backend == "none":
            self._backend = "ollama"
            log.info(f"Using Ollama backend: {OLLAMA_URL}/{OLLAMA_MODEL}")

    def _try_load_gguf(self):
        path = Path(MODEL_PATH)

        # Auto-download from HuggingFace if not found locally
        if not path.exists():
            log.info(f"GGUF not found locally. Downloading from HuggingFace ({HF_REPO})...")
            try:
                self._download_from_hf(path)
            except Exception as e:
                log.warning(f"Could not download GGUF from HuggingFace: {e}")

        if not path.exists():
            log.info(f"GGUF model not available, will use Ollama")
            return

        try:
            from llama_cpp import Llama
            self._llama = Llama(
                model_path=str(path),
                n_gpu_layers=-1,
                n_ctx=2048,     # enough for RAG context + response
                n_batch=128,    # smaller batch = less memory
                n_threads=2,    # match HF free CPU cores
                verbose=False,
            )
            self._backend = "gguf"
            log.info(f"GGUF model loaded: {path.name} ({path.stat().st_size / 1e9:.1f}GB)")
        except Exception as e:
            log.warning(f"Could not load GGUF: {e}")

    def _download_from_hf(self, dest: Path):
        """Download GGUF from HuggingFace Hub."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Try huggingface_hub first (fast, resumable)
        try:
            from huggingface_hub import hf_hub_download
            log.info(f"Downloading via huggingface_hub...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=MODEL_FILENAME,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            log.info(f"Downloaded to {downloaded}")
            return
        except ImportError:
            log.info("huggingface_hub not installed, trying direct download...")
        except Exception as e:
            log.warning(f"huggingface_hub download failed: {e}")

        # Fallback: direct HTTP download with progress
        import urllib.request
        import shutil

        log.info(f"Downloading {MODEL_FILENAME} from {HF_MODEL_URL} (~1.9GB)...")
        log.info("This is a one-time download. Please wait...")

        tmp_path = dest.with_suffix('.tmp')
        try:
            req = urllib.request.Request(HF_MODEL_URL, headers={"User-Agent": "YatriAI/1.0"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                total = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                with open(tmp_path, 'wb') as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)  # 8MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100 // total
                            log.info(f"  Download progress: {pct}% ({downloaded // 1e6:.0f}MB / {total // 1e6:.0f}MB)")
            shutil.move(str(tmp_path), str(dest))
            log.info(f"Download complete: {dest}")
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Download failed: {e}") from e

    def generate(
        self,
        query: str,
        context: list[str],
        language: str = "en",
        domain: str = "general",
        max_tokens: int = 200,
        temperature: float = 0.3,
    ) -> str:
        # Limit context: top 3 chunks, max 800 chars each (fits 2048 ctx window)
        trimmed = [c[:800] for c in context[:3]]

        # Try enhancement API first (fast) — skip slow custom model if it works
        if ENHANCE_RESPONSES:
            enhanced = None
            if GROQ_API_KEY:
                enhanced = self._groq_generate(query, trimmed, language)
            elif GEMINI_API_KEY:
                enhanced = self._gemini_generate(query, trimmed, language)
            if enhanced:
                log.debug("Using enhanced response")
                return enhanced
            log.debug("Enhancement unavailable, falling back to custom model")

        # Fallback: custom model
        messages = _build_prompt(query, trimmed, language)

        if self._backend == "gguf" and self._llama:
            try:
                response = self._llama.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    repeat_penalty=1.1,
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning(f"Custom model error: {e}")

        elif self._backend == "ollama":
            return self._ollama_generate(messages, max_tokens, temperature)

        return "Service unavailable. Please try again."

    def _groq_generate(
        self, query: str, context: list[str], language: str
    ) -> Optional[str]:
        """Call Groq free API (OpenAI-compatible, very fast)."""
        try:
            context_block = "\n---\n".join(context) if context else ""
            user_content = (
                f"Context from knowledge base:\n{context_block}\n\n"
                f"User question: {query}"
            ) if context_block else f"User question: {query}"

            resp = httpx.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": ENHANCE_SYSTEM},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 600,
                    "top_p": 0.9,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"Groq enhancement failed: {e}")
            return None

    async def _groq_generate_stream(
        self, query: str, context: list[str], language: str
    ) -> AsyncGenerator[str, None]:
        """Stream from Groq free API."""
        try:
            context_block = "\n---\n".join(context) if context else ""
            user_content = (
                f"Context from knowledge base:\n{context_block}\n\n"
                f"User question: {query}"
            ) if context_block else f"User question: {query}"

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": ENHANCE_SYSTEM},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.4,
                        "max_tokens": 600,
                        "stream": True,
                    },
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            try:
                                data = _json.loads(line[6:])
                                content = data["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                continue
        except Exception as e:
            log.warning(f"Groq stream failed: {e}")
            return

    def _gemini_generate(
        self, query: str, context: list[str], language: str
    ) -> Optional[str]:
        """Call Gemini free API for enhanced response (with retry on 429)."""
        import time as _time
        payload = _build_enhance_prompt(query, context, language)
        url = f"{GEMINI_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

        for attempt in range(3):
            try:
                resp = httpx.post(url, json=payload, timeout=30.0)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    log.info(f"Gemini rate limited, retrying in {wait}s...")
                    _time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
            except Exception as e:
                log.warning(f"Gemini enhancement failed: {e}")
                return None
        return None

    async def _gemini_generate_stream(
        self, query: str, context: list[str], language: str
    ) -> AsyncGenerator[str, None]:
        """Stream from Gemini free API."""
        try:
            payload = _build_enhance_prompt(query, context, language)
            url = (
                f"{GEMINI_URL}/{GEMINI_MODEL}:streamGenerateContent"
                f"?key={GEMINI_API_KEY}&alt=sse"
            )
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = _json.loads(line[6:])
                                parts = (
                                    data.get("candidates", [{}])[0]
                                    .get("content", {})
                                    .get("parts", [])
                                )
                                for part in parts:
                                    text = part.get("text", "")
                                    if text:
                                        yield text
                            except Exception:
                                continue
        except Exception as e:
            log.warning(f"Gemini stream failed: {e}")
            return

    def _ollama_generate(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> str:
        try:
            resp = httpx.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                    },
                },
                timeout=180.0,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            log.error(f"Ollama error: {e}")
            return "I'm having trouble connecting to the language model. Please try again."

    async def generate_stream(
        self,
        query: str,
        context: list[str],
        language: str = "en",
        domain: str = "general",
    ) -> AsyncGenerator[str, None]:
        """Streaming generation for WebSocket — enhanced via Gemini when available."""
        trimmed = [c[:1500] for c in context[:3]]

        # Try enhanced streaming first (Groq or Gemini)
        if ENHANCE_RESPONSES:
            got_tokens = False
            if GROQ_API_KEY:
                async for token in self._groq_generate_stream(query, trimmed, language):
                    got_tokens = True
                    yield token
            elif GEMINI_API_KEY:
                async for token in self._gemini_generate_stream(query, trimmed, language):
                    got_tokens = True
                    yield token
            if got_tokens:
                return

        # Fallback to custom model / Ollama
        messages = _build_prompt(query, context, language)

        if self._backend == "gguf" and self._llama:
            response = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.3,
                stream=True,
            )
            for chunk in response:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta

        else:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": True,
                        "options": {"temperature": 0.3, "num_predict": 512},
                    },
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            try:
                                data = _json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                continue

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model_name(self) -> str:
        if self._backend == "gguf":
            return Path(MODEL_PATH).stem
        return OLLAMA_MODEL


_llm_instance: Optional[LLMService] = None


def get_llm() -> LLMService:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMService()
    return _llm_instance
