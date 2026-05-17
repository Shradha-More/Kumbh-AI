"""
OpenStreetMap / Overpass API spider for Nashik Kumbh Mela 2027.
Queries OSM for all relevant places in the Nashik region.

Dependencies:
    pip install overpy

Usage:
    python osm_places.py
    python osm_places.py --categories temples ghats hospitals
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import overpy

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "raw" / "osm"

# Nashik bounding box: S, W, N, E
BBOX = (19.8, 73.6, 20.2, 74.0)

CATEGORIES: dict[str, list[str]] = {
    "temples": [
        'node["amenity"="place_of_worship"]["religion"="hindu"]',
        'way["amenity"="place_of_worship"]["religion"="hindu"]',
        'node["historic"="temple"]',
    ],
    "ghats": [
        'node["leisure"="bathing_place"]',
        'node["name"~"ghat",i]',
        'way["name"~"ghat",i]',
    ],
    "hospitals": [
        'node["amenity"="hospital"]',
        'way["amenity"="hospital"]',
        'node["amenity"="clinic"]',
        'node["healthcare"]',
    ],
    "police": [
        'node["amenity"="police"]',
        'way["amenity"="police"]',
    ],
    "transport": [
        'node["highway"="bus_stop"]',
        'node["amenity"="bus_station"]',
        'node["railway"="station"]',
        'node["amenity"="parking"]',
        'way["amenity"="parking"]',
    ],
    "tourist_attractions": [
        'node["tourism"="attraction"]',
        'node["tourism"="museum"]',
        'node["historic"]',
        'way["tourism"="attraction"]',
    ],
    "dharamshalas": [
        'node["tourism"="guest_house"]',
        'node["amenity"="shelter"]',
        'node["tourism"="hostel"]',
    ],
    "water_points": [
        'node["amenity"="drinking_water"]',
        'node["amenity"="water_point"]',
        'node["natural"="spring"]',
    ],
    "first_aid": [
        'node["emergency"="ambulance_station"]',
        'node["amenity"="pharmacy"]',
        'node["amenity"="first_aid"]',
    ],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("osm_places")


def extract_place(element: Any, category: str) -> dict:
    tags = element.tags if hasattr(element, "tags") else {}
    lat, lon = None, None
    if hasattr(element, "lat"):
        lat, lon = float(element.lat), float(element.lon)
    elif hasattr(element, "center_lat") and element.center_lat:
        lat, lon = float(element.center_lat), float(element.center_lon)

    return {
        "osm_id": element.id,
        "osm_type": type(element).__name__.lower(),
        "category": category,
        "name": tags.get("name", ""),
        "name_hi": tags.get("name:hi", ""),
        "name_mr": tags.get("name:mr", ""),
        "name_en": tags.get("name:en", ""),
        "lat": lat, "lon": lon,
        "amenity": tags.get("amenity", ""),
        "tourism": tags.get("tourism", ""),
        "historic": tags.get("historic", ""),
        "religion": tags.get("religion", ""),
        "opening_hours": tags.get("opening_hours", ""),
        "phone": tags.get("phone", tags.get("contact:phone", "")),
        "website": tags.get("website", ""),
        "description": tags.get("description", ""),
        "addr_street": tags.get("addr:street", ""),
        "addr_city": tags.get("addr:city", ""),
    }


def query_category(api: overpy.API, category: str, filters: list[str], bbox: tuple) -> list[dict]:
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"
    union_parts = "\n".join(f'  {f}({bbox_str});' for f in filters)
    query = f"[out:json][timeout:90];\n(\n{union_parts}\n);\nout center tags;"

    log.info("Querying OSM: %s", category)
    try:
        result = api.query(query)
        places = []
        for el in list(result.nodes) + list(result.ways):
            places.append(extract_place(el, category))
        log.info("  → %d places for '%s'", len(places), category)
        return places
    except Exception as exc:
        log.error("Overpass failed for '%s': %s", category, exc)
        return []


def run(target_categories=None, bbox=BBOX, delay=2.0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = overpy.API(max_retry_num=3, retry_timeout=10)
    cats = target_categories or list(CATEGORIES.keys())
    all_places: list[dict] = []

    for category in cats:
        if category not in CATEGORIES:
            log.warning("Unknown category: %s", category)
            continue

        out_path = OUTPUT_DIR / f"{category}.json"
        if out_path.exists():
            log.info("Cached — loading %s", out_path.name)
            with open(out_path, encoding="utf-8") as f:
                all_places.extend(json.load(f))
            continue

        places = query_category(api, category, CATEGORIES[category], bbox)
        if places:
            out_path.write_text(json.dumps(places, ensure_ascii=False, indent=2), encoding="utf-8")
            all_places.extend(places)

        time.sleep(delay)

    # Deduplicate
    seen = set()
    unique = []
    for p in all_places:
        key = (p["osm_type"], p["osm_id"])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    master = OUTPUT_DIR.parent / "osm_places.json"
    master.write_text(json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("OSM complete — %d unique places saved to %s", len(unique), master)
    return unique


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", choices=list(CATEGORIES.keys()))
    p.add_argument("--delay", type=float, default=2.0)
    args = p.parse_args()
    run(target_categories=args.categories, delay=args.delay)


if __name__ == "__main__":
    main()
