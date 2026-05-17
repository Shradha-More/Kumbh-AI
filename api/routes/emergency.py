"""Emergency route — hardcoded responses, bypasses LLM entirely for speed."""

from fastapi import APIRouter, Depends

from api.models.schemas import EmergencyContact, EmergencyRequest, EmergencyResponse, NearestFacility
from api.services.rag import get_rag, RAGService

router = APIRouter(prefix="/emergency", tags=["emergency"])

FALLBACK_RESPONSE = {
    "en": "EMERGENCY: Call 108 (Ambulance) | 100 (Police) | 101 (Fire)\nKumbh Helpline: 1800-120-2027\nMissing Persons: 1800-222-2027",
    "hi": "आपातकाल: 108 (एम्बुलेंस) | 100 (पुलिस) | 101 (अग्नि)\nकुंभ हेल्पलाइन: 1800-120-2027",
    "mr": "आणीबाणी: 108 (रुग्णवाहिका) | 100 (पोलीस) | 101 (अग्निशमन)\nकुंभ हेल्पलाइन: 1800-120-2027",
    "gu": "કટોકટી: 108 | 100 | 101\nકુંભ: 1800-120-2027",
    "ta": "அவசரநிலை: 108 | 100 | 101\nகும்ப: 1800-120-2027",
    "te": "అత్యవసరం: 108 | 100 | 101\nకుంభ: 1800-120-2027",
    "kn": "ತುರ್ತು: 108 | 100 | 101\nಕುಂಭ: 1800-120-2027",
    "ml": "അടിയന്തരം: 108 | 100 | 101\nകുംഭ: 1800-120-2027",
}


@router.post("", response_model=EmergencyResponse)
async def emergency_help(
    req: EmergencyRequest,
    rag: RAGService = Depends(get_rag),
):
    """Instant emergency response — no LLM, hardcoded data only."""
    language = req.language if req.language != "auto" else "en"

    # Lookup hardcoded scenario
    result = rag.retrieve_emergency(req.query, language)
    helplines = rag.get_all_helplines()

    if result:
        response_text = result["response"]
        scenario_type = result["type"]
    else:
        response_text = FALLBACK_RESPONSE.get(language, FALLBACK_RESPONSE["en"])
        scenario_type = "general"

    # Build contacts list
    contacts = [
        EmergencyContact(name="Ambulance", number=helplines.get("ambulance", {}).get("number", "108")),
        EmergencyContact(name="Police", number=helplines.get("police", {}).get("number", "100")),
        EmergencyContact(name="Fire", number=helplines.get("fire", {}).get("number", "101")),
        EmergencyContact(name="Kumbh Helpline", number=helplines.get("kumbh_mela_helpline", "1800-120-2027")),
        EmergencyContact(name="Missing Persons", number=helplines.get("missing_persons", "1800-222-2027")),
        EmergencyContact(name="Women Helpline", number=helplines.get("women_helpline", "1091")),
    ]

    # Nearest facility if location provided
    nearest = None
    if req.location:
        facility_type = "hospital" if scenario_type in ("medical",) else "police"
        raw = rag.nearest_facility(req.location.lat, req.location.lon, facility_type)
        if raw:
            coords = raw.get("coordinates")
            nearest = NearestFacility(
                name=raw.get("name", ""),
                distance_m=raw.get("distance_m", 0),
                phone=raw.get("phone", ""),
                address=raw.get("address", ""),
                coordinates=coords,
            )

    return EmergencyResponse(
        type=scenario_type,
        response=response_text,
        contacts=contacts,
        nearest_facility=nearest,
    )


@router.get("/contacts")
async def get_contacts(
    language: str = "en",
    rag: RAGService = Depends(get_rag),
):
    helplines = rag.get_all_helplines()
    hospitals = rag.get_hospitals()
    police = rag.get_police_stations()
    return {
        "helplines": helplines,
        "hospitals": hospitals[:5],
        "police_stations": police[:5],
    }


@router.get("/nearest")
async def nearest_facility(
    lat: float,
    lon: float,
    type: str = "hospital",
    rag: RAGService = Depends(get_rag),
):
    result = rag.nearest_facility(lat, lon, type)
    if not result:
        return {"error": "No facility found nearby"}
    return result
