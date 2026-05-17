"""Nashik places route — tourist places, nearby search, itinerary planning."""

import json
import math
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.models.schemas import Coordinates, ItineraryRequest, ItineraryResponse, ItineraryDay, Place

router = APIRouter(prefix="/places", tags=["places"])

ROOT = Path(__file__).parent.parent.parent

PLACES_FILES = [
    ROOT / "data" / "nashik_places.json",
    ROOT / "data" / "nashik_complete_places.json",
    ROOT / "static" / "places_geo.json",
]

_places_cache: Optional[list[dict]] = None


def _load_places() -> list[dict]:
    global _places_cache
    if _places_cache is not None:
        return _places_cache

    all_places = []
    seen_names = set()

    for path in PLACES_FILES:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            items = data.get("places", data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                continue
            for item in items:
                name = item.get("name", item.get("name_en", ""))
                if name and name not in seen_names:
                    seen_names.add(name)
                    all_places.append(item)
        except Exception:
            continue

    _places_cache = all_places
    return _places_cache


def _localize_place(place: dict, language: str) -> Place:
    """Extract the right language fields from a place dict."""
    lang = language if language != "auto" else "en"
    fallback = "en"

    name = place.get(f"name_{lang}") or place.get(f"name_{fallback}") or place.get("name", "")
    description = (
        place.get(f"description_{lang}")
        or place.get(f"description_{fallback}")
        or place.get("description_en", "")
    )
    how_to_reach = (
        place.get(f"how_to_reach_{lang}")
        or place.get(f"how_to_reach_{fallback}")
        or place.get("how_to_reach_en", "")
    )

    coords = place.get("coordinates")
    return Place(
        id=place.get("id", ""),
        name=name,
        category=place.get("category", "general"),
        subcategory=place.get("subcategory"),
        description=description,
        coordinates=Coordinates(**coords) if coords else None,
        timings=place.get("timings"),
        entry_fee=place.get("entry_fee"),
        how_to_reach=how_to_reach,
        tips=place.get("tips_en", ""),
        distance_from_nashik=place.get("distance_from_nashik"),
        crowd_level_kumbh=place.get("crowd_level_kumbh"),
    )


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


@router.get("", response_model=list[Place])
async def list_places(category: Optional[str] = None, language: str = "en"):
    places = _load_places()
    if category:
        places = [p for p in places if p.get("category") == category
                  or p.get("subcategory") == category]
    return [_localize_place(p, language) for p in places]


@router.get("/nearby", response_model=list[Place])
async def nearby_places(
    lat: float,
    lon: float,
    radius: str = "5km",
    category: Optional[str] = None,
    language: str = "en",
):
    radius_m = float(radius.lower().replace("km", "").strip()) * 1000
    places = _load_places()

    results = []
    for p in places:
        coords = p.get("coordinates")
        if not coords:
            continue
        dist = _haversine(lat, lon, coords["lat"], coords["lon"])
        if dist <= radius_m:
            if category and p.get("category") != category:
                continue
            results.append((dist, p))

    results.sort(key=lambda x: x[0])
    return [_localize_place(p, language) for _, p in results[:20]]


@router.get("/{place_id}", response_model=Place)
async def get_place(place_id: str, language: str = "en"):
    places = _load_places()
    for p in places:
        if p.get("id") == place_id:
            return _localize_place(p, language)
    raise HTTPException(status_code=404, detail=f"Place '{place_id}' not found")


@router.post("/recommend", response_model=ItineraryResponse)
async def recommend_itinerary(req: ItineraryRequest):
    places = _load_places()
    lang = req.language if req.language != "auto" else "en"

    # Filter and score by interests
    interest_keywords = {
        "temple": ["temple", "mandir", "pilgrimage", "religious", "jyotirlinga"],
        "nature": ["waterfall", "hill", "trek", "nature", "forest"],
        "heritage": ["fort", "cave", "history", "heritage", "buddhist"],
        "food": ["food", "restaurant", "market"],
        "wine": ["wine", "vineyard", "winery"],
    }

    scored = []
    for p in places:
        score = 0
        cat = (p.get("category", "") + " " + p.get("subcategory", "")).lower()
        for interest in req.interests:
            for kw in interest_keywords.get(interest.lower(), [interest.lower()]):
                if kw in cat:
                    score += 2
        # Always include core Kumbh places
        if p.get("id") in ("ramkund", "kalaram_temple", "trimbakeshwar"):
            score += 5
        scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    selected = [p for _, p in scored if _[0] > 0][:req.days_available * 4]

    # Distribute across days
    days = []
    chunk = max(1, len(selected) // req.days_available)
    for day_num in range(1, req.days_available + 1):
        day_places = selected[(day_num-1)*chunk : day_num*chunk]
        place_names = [
            (p.get(f"name_{lang}") or p.get("name_en") or p.get("name", ""))
            for p in day_places
        ]
        days.append(ItineraryDay(
            day=day_num,
            places=place_names,
            description=f"Day {day_num}: Visit {', '.join(place_names[:3])}",
        ))

    return ItineraryResponse(
        days=days,
        total_places=len(selected),
        language=lang,
    )
