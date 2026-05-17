"""Text query route with session history and streaming."""

import time
import uuid
from collections import defaultdict, deque
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langdetect import detect

from api.models.schemas import QueryRequest, QueryResponse, SourceDoc
from api.services.llm import get_llm, LLMService
from api.services.rag import get_rag, RAGService

router = APIRouter(prefix="/query", tags=["query"])

# In-memory session history: session_id → deque of (role, text) pairs
_sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))


def detect_language(text: str, hint: Optional[str] = None) -> str:
    if hint and hint not in ("auto", "en"):
        return hint

    # Quick script-based detection (more reliable than langdetect for Indian languages)
    import unicodedata
    script_counts = {}
    for ch in text:
        try:
            name = unicodedata.name(ch, '')
            if 'DEVANAGARI' in name: script_counts['devanagari'] = script_counts.get('devanagari', 0) + 1
            elif 'GUJARATI' in name: script_counts['gu'] = script_counts.get('gu', 0) + 1
            elif 'TAMIL' in name: script_counts['ta'] = script_counts.get('ta', 0) + 1
            elif 'TELUGU' in name: script_counts['te'] = script_counts.get('te', 0) + 1
            elif 'KANNADA' in name: script_counts['kn'] = script_counts.get('kn', 0) + 1
            elif 'MALAYALAM' in name: script_counts['ml'] = script_counts.get('ml', 0) + 1
            elif 'BENGALI' in name: script_counts['bn'] = script_counts.get('bn', 0) + 1
            elif 'GURMUKHI' in name: script_counts['pa'] = script_counts.get('pa', 0) + 1
            elif 'ARABIC' in name: script_counts['ur'] = script_counts.get('ur', 0) + 1
        except: pass

    # If Devanagari script detected — could be Hindi or Marathi
    if script_counts.get('devanagari', 0) > 2:
        marathi_markers = ['आहे', 'नाही', 'कसे', 'काय', 'मध्ये', 'पासून', 'आणि', 'तुम्ही', 'त्या', 'हे']
        if any(m in text for m in marathi_markers):
            return 'mr'
        return 'hi'
    if script_counts.get('bn', 0) > 2: return 'bn'
    if script_counts.get('pa', 0) > 2: return 'pa'
    if script_counts.get('ur', 0) > 2: return 'ur'
    if script_counts.get('gu', 0) > 2: return 'gu'
    if script_counts.get('ta', 0) > 2: return 'ta'
    if script_counts.get('te', 0) > 2: return 'te'
    if script_counts.get('kn', 0) > 2: return 'kn'
    if script_counts.get('ml', 0) > 2: return 'ml'

    # Fallback to langdetect for romanized text
    if hint and hint != "auto":
        return hint
    try:
        lang = detect(text)
        mapping = {
            "hi": "hi", "mr": "mr", "gu": "gu", "ta": "ta", "te": "te",
            "kn": "kn", "ml": "ml", "en": "en", "bn": "bn", "pa": "pa",
            "ur": "hi",  # Urdu → Hindi for spoken
        }
        return mapping.get(lang, "en")
    except:
        return "en"


EMERGENCY_KEYWORDS = {
    "medical": ["doctor", "hospital", "ambulance", "injury", "hurt", "sick", "bleeding", "fever", "heart attack",
                "डॉक्टर", "अस्पताल", "एम्बुलेंस", "चोट", "बीमार", "दर्द", "बुखार", "खून", "बेहोश", "तबीयत",
                "दुखापत", "आजारी", "रुग्णवाहिका", "हॉस्पिटल"],
    "missing": ["missing", "lost", "child lost", "खोया", "गुम", "बच्चा खो", "लापता", "हरवले"],
    "fire": ["fire", "burning", "आग", "जल रहा", "आग लागली"],
    "drowning": ["drowning", "river", "डूब", "नदी", "बुडत"],
}

EMERGENCY_RESPONSES = {
    "medical": {
        "en": "EMERGENCY MEDICAL HELP:\n- Ambulance: 108\n- Kumbh Helpline: 1800-120-2027\n- Nearest hospitals: Civil Hospital (Trimbak Road), Bytco Hospital (College Road), Wockhardt Hospital (Mumbai Naka)\n- First aid centers at all major ghats during Kumbh\n- Carry basic medicines, stay hydrated",
        "hi": "आपातकालीन चिकित्सा सहायता:\n- एम्बुलेंस: 108\n- कुंभ हेल्पलाइन: 1800-120-2027\n- नजदीकी अस्पताल: सिविल हॉस्पिटल (त्रिंबक रोड), बायटको हॉस्पिटल (कॉलेज रोड), वॉकहार्ट हॉस्पिटल (मुंबई नाका)\n- कुंभ के दौरान सभी प्रमुख घाटों पर प्राथमिक चिकित्सा केंद्र\n- बुनियादी दवाइयाँ रखें, पानी पीते रहें",
        "mr": "आणीबाणी वैद्यकीय मदत:\n- रुग्णवाहिका: 108\n- कुंभ हेल्पलाइन: 1800-120-2027\n- जवळचे हॉस्पिटल: सिव्हिल हॉस्पिटल (त्रिंबक रोड), बायटको हॉस्पिटल\n- कुंभात सर्व प्रमुख घाटांवर प्रथमोपचार केंद्रे",
    },
    "missing": {
        "en": "MISSING PERSON:\n- Kumbh Missing Persons Helpline: 1800-222-2027\n- Police: 100\n- Register at nearest Lost & Found center\n- Announce at ghat PA systems\n- Share photo with police control room",
        "hi": "लापता व्यक्ति:\n- कुंभ लापता व्यक्ति हेल्पलाइन: 1800-222-2027\n- पुलिस: 100\n- नजदीकी खोया-पाया केंद्र में रजिस्टर करें\n- घाट के PA सिस्टम पर घोषणा करवाएं",
        "mr": "बेपत्ता व्यक्ती:\n- कुंभ बेपत्ता हेल्पलाइन: 1800-222-2027\n- पोलीस: 100\n- जवळच्या हरवले-सापडले केंद्रात नोंदणी करा",
    },
    "fire": {
        "en": "FIRE EMERGENCY:\n- Fire Brigade: 101\n- Police: 100\n- Emergency: 112\n- Evacuate immediately, stay low, cover mouth",
        "hi": "अग्नि आपातकाल:\n- अग्निशमन: 101\n- पुलिस: 100\n- आपातकालीन: 112\n- तुरंत बाहर निकलें, नीचे रहें, मुंह ढकें",
        "mr": "अग्निशमन आणीबाणी:\n- अग्निशमन: 101\n- पोलीस: 100\n- आणीबाणी: 112",
    },
    "drowning": {
        "en": "DROWNING/RIVER EMERGENCY:\n- Emergency: 112\n- Ambulance: 108\n- Do NOT jump in. Throw rope or float. Call for lifeguards at ghat.",
        "hi": "डूबने की आपातकालीन:\n- आपातकालीन: 112\n- एम्बुलेंस: 108\n- अंदर न कूदें। रस्सी या तैरने वाली चीज फेंकें। घाट पर लाइफगार्ड को बुलाएं।",
        "mr": "बुडण्याची आणीबाणी:\n- आणीबाणी: 112\n- रुग्णवाहिका: 108\n- आत उडी मारू नका. दोरी किंवा तरंगणारी वस्तू फेका.",
    },
}


@router.post("", response_model=QueryResponse)
async def text_query(
    req: QueryRequest,
    rag: RAGService = Depends(get_rag),
    llm: LLMService = Depends(get_llm),
):
    session_id = req.session_id or str(uuid.uuid4())
    language = detect_language(req.query, req.language)
    query_lower = req.query.lower()

    # Emergency bypass — instant response without LLM
    for etype, keywords in EMERGENCY_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            responses = EMERGENCY_RESPONSES.get(etype, {})
            response_text = responses.get(language, responses.get("en", "Call 112 for emergency"))
            return QueryResponse(
                response=response_text,
                language=language,
                sources=[],
                session_id=session_id,
            )

    # Retrieve context — always include English docs for better coverage
    docs = rag.retrieve(req.query, language=language, domain=req.domain, top_k=5)
    if language != "en":
        en_docs = rag.retrieve(req.query, language="en", domain=req.domain, top_k=3)
        # Deduplicate by first 80 chars
        seen = {d["text"][:80] for d in docs}
        docs += [d for d in en_docs if d["text"][:80] not in seen]
    context_texts = [d["text"] for d in docs]

    # Add conversation history to context if session exists
    history = _sessions[session_id]
    if history:
        history_text = "\n".join(f"{role}: {text}" for role, text in history)
        context_texts = [f"Previous conversation:\n{history_text}"] + context_texts

    domain = docs[0]["domain"] if docs else "general"

    # Generate
    response_text = llm.generate(
        query=req.query,
        context=context_texts,
        language=language,
        domain=domain,
    )

    # Detect actual response language (Groq auto-matches user language)
    response_lang = detect_language(response_text, hint=language)

    # Update session history
    _sessions[session_id].append(("User", req.query))
    _sessions[session_id].append(("Assistant", response_text))

    return QueryResponse(
        response=response_text,
        language=response_lang,
        sources=[
            SourceDoc(text=d["text"][:200], domain=d["domain"], source=d["source"], score=d["score"])
            for d in docs[:3]
        ],
        domain=domain,
        confidence=docs[0]["score"] if docs else 0.0,
        session_id=session_id,
    )


@router.post("/stream")
async def text_query_stream(
    req: QueryRequest,
    rag: RAGService = Depends(get_rag),
    llm: LLMService = Depends(get_llm),
):
    """Server-Sent Events streaming response."""
    session_id = req.session_id or str(uuid.uuid4())
    language = detect_language(req.query, req.language)
    docs = rag.retrieve(req.query, language=language, domain=req.domain, top_k=5)
    context_texts = [d["text"] for d in docs]

    async def event_stream():
        full_response = []
        async for token in llm.generate_stream(req.query, context_texts, language):
            full_response.append(token)
            yield f"data: {token}\n\n"
        _sessions[session_id].append(("User", req.query))
        _sessions[session_id].append(("Assistant", "".join(full_response)))
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    history = list(_sessions.get(session_id, []))
    return {"session_id": session_id, "turns": [{"role": r, "text": t} for r, t in history]}
