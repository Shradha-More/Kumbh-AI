"""
flatten_seed.py — Converts multilingual seed JSON files into flat per-language
documents that clean.py can process (each doc has 'content', 'language', 'title').

Run once before pipeline/run_pipeline.py:
    python pipeline/flatten_seed.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = PROJECT_ROOT / "knowledge_base" / "raw" / "seed"

LANG_SUFFIX = {
    "en": "_en", "hi": "_hi", "mr": "_mr", "gu": "_gu",
    "ta": "_ta", "te": "_te", "kn": "_kn", "ml": "_ml",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


def emit(docs: list[dict], lang: str, title: str, content: str, domain: str, source: str) -> None:
    content = (content or "").strip()
    if len(content.split()) >= 20:
        docs.append({"language": lang, "title": title, "content": content,
                     "domain": domain, "source_url": source})


def flatten_schedule(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: list[dict] = []

    # Event name block
    event_en = data.get("event", "")
    for lang, suffix in LANG_SUFFIX.items():
        val = data.get(f"event{suffix}", event_en)
        if val:
            emit(docs, lang, val, val, "schedule", "seed:kumbh_schedule")

    # General info per language
    general = data.get("general_info", {})
    if isinstance(general, dict):
        for lang, suffix in LANG_SUFFIX.items():
            key = f"en" if lang == "en" else lang
            block = general.get(key) or general.get(lang, {})
            if isinstance(block, dict):
                text = "\n".join(f"{k}: {v}" for k, v in block.items() if isinstance(v, str))
            elif isinstance(block, str):
                text = block
            else:
                text = ""
            emit(docs, lang, f"Kumbh 2027 General Info ({lang})", text, "schedule", "seed:kumbh_schedule")

    # Shahi Snan dates — build a text block in each language
    shahi = data.get("shahi_snan", [])
    if isinstance(shahi, list):
        for lang, suffix in LANG_SUFFIX.items():
            lines = []
            for snan in shahi:
                if isinstance(snan, dict):
                    date = snan.get("date", "")
                    name = snan.get(f"name{suffix}", snan.get("name_en", snan.get("name", "")))
                    significance = snan.get(f"significance{suffix}", snan.get("significance_en", snan.get("significance", "")))
                    lines.append(f"{date}: {name} — {significance}")
            if lines:
                emit(docs, lang, f"Shahi Snan Dates ({lang})", "\n".join(lines), "schedule", "seed:kumbh_schedule")

    # Akhara bathing order
    akhara_order = data.get("akhara_bathing_order", [])
    if isinstance(akhara_order, list):
        text = "\n".join(
            f"{i+1}. {a.get('name', a) if isinstance(a, dict) else a}"
            for i, a in enumerate(akhara_order)
        )
        if text:
            emit(docs, "en", "Akhara Bathing Order", text, "schedule", "seed:kumbh_schedule")
            emit(docs, "hi", "अखाड़ा स्नान क्रम", text, "schedule", "seed:kumbh_schedule")

    log.info("schedule → %d documents", len(docs))
    return docs


def flatten_places(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: list[dict] = []
    places = data.get("places", [])

    for place in places:
        pid = place.get("id", "place")
        for lang, suffix in LANG_SUFFIX.items():
            name = place.get(f"name{suffix}", place.get("name", ""))
            desc = place.get(f"description{suffix}", "")
            reach = place.get(f"how_to_reach{suffix}", "")
            tips  = place.get(f"tips{suffix}", "")
            timings = place.get("timings", "")
            entry = place.get("entry_fee", "")

            parts = []
            if desc:
                parts.append(desc)
            if reach:
                parts.append(f"How to reach: {reach}")
            if tips:
                parts.append(f"Tips: {tips}")
            if timings:
                parts.append(f"Timings: {timings}")
            if entry:
                parts.append(f"Entry fee: {entry}")

            coords = place.get("coordinates", {})
            if coords.get("lat") and coords.get("lon"):
                parts.append(f"Location: {coords['lat']}, {coords['lon']}")

            content = "\n".join(p for p in parts if p)
            title = name or f"Place {pid}"
            emit(docs, lang, title, content, "places", f"seed:nashik_places:{pid}")

    log.info("nashik_places → %d documents", len(docs))
    return docs


def flatten_ghats(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: list[dict] = []

    # Ghats
    for ghat in data.get("ghats", []):
        if not isinstance(ghat, dict):
            continue
        gid = ghat.get("id", "ghat")
        for lang, suffix in LANG_SUFFIX.items():
            name = ghat.get(f"name{suffix}", ghat.get("name", ""))
            desc = ghat.get(f"description{suffix}", ghat.get("description_en", ""))
            significance = ghat.get(f"significance{suffix}", ghat.get("significance_en", ""))
            facilities = ghat.get("facilities", [])
            fac_text = ", ".join(facilities) if isinstance(facilities, list) else str(facilities)

            parts = [p for p in [desc, significance, f"Facilities: {fac_text}" if fac_text else ""] if p]
            emit(docs, lang, name or gid, "\n".join(parts), "transport", f"seed:ghats:{gid}")

    # Transport
    transport = data.get("transport", {})
    for mode, info in transport.items():
        if not isinstance(info, dict):
            continue
        for lang, suffix in LANG_SUFFIX.items():
            name = info.get(f"name{suffix}", info.get("name", mode))
            desc = info.get(f"description{suffix}", info.get("description_en", ""))
            details_key = f"details{suffix}" if f"details{suffix}" in info else "details"
            details = info.get(details_key, "")
            if isinstance(details, list):
                details = "\n".join(str(d) for d in details)
            parts = [p for p in [desc, details] if p]
            emit(docs, lang, name or mode, "\n".join(parts), "transport", f"seed:transport:{mode}")

    # Accommodation
    for acc in data.get("accommodation", []):
        if not isinstance(acc, dict):
            continue
        for lang, suffix in LANG_SUFFIX.items():
            name = acc.get(f"name{suffix}", acc.get("name", ""))
            desc = acc.get(f"description{suffix}", acc.get("description_en", ""))
            contact = acc.get("contact", "")
            parts = [p for p in [desc, f"Contact: {contact}" if contact else ""] if p]
            emit(docs, lang, name, "\n".join(parts), "transport", "seed:accommodation")

    log.info("ghats_and_transport → %d documents", len(docs))
    return docs


def flatten_emergency(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: list[dict] = []

    # Helplines block — one doc per language
    helplines = data.get("helplines", {})
    if helplines:
        for lang in LANG_SUFFIX:
            lines = [f"{k}: {v}" for k, v in helplines.items()]
            emit(docs, lang, "Emergency Helplines Nashik Kumbh 2027", "\n".join(lines),
                 "emergency", "seed:emergency_helplines")

    # Hospitals
    for hosp in data.get("hospitals", []):
        if not isinstance(hosp, dict):
            continue
        name = hosp.get("name", "Hospital")
        address = hosp.get("address", "")
        phone = hosp.get("phone", "")
        specialties = ", ".join(hosp.get("specialties", []))
        text = f"{name}\nAddress: {address}\nPhone: {phone}\nSpecialties: {specialties}"
        emit(docs, "en", name, text, "emergency", "seed:hospitals")

    # Emergency scenarios — extract response texts per language
    scenarios = data.get("scenarios", data.get("emergencies", []))
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            sid = scenario.get("id", "emergency")
            for lang, suffix in LANG_SUFFIX.items():
                response = scenario.get(f"response{suffix}", scenario.get("response_en", ""))
                instructions = scenario.get(f"instructions{suffix}", scenario.get("instructions_en", ""))
                if isinstance(instructions, list):
                    instructions = "\n".join(instructions)
                parts = [p for p in [response, instructions] if p]
                title_key = f"title{suffix}" if f"title{suffix}" in scenario else "title"
                title = scenario.get(title_key, sid)
                emit(docs, lang, title, "\n".join(parts), "emergency", f"seed:emergency:{sid}")

    log.info("emergency_responses → %d documents", len(docs))
    return docs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    handlers: list[tuple[str, Any]] = [
        ("kumbh_2027_schedule.json",  flatten_schedule),
        ("nashik_places.json",        flatten_places),
        ("ghats_and_transport.json",  flatten_ghats),
        ("emergency_responses.json",  flatten_emergency),
    ]

    total = 0
    for filename, fn in handlers:
        src = DATA_DIR / filename
        if not src.exists():
            log.warning("Not found: %s — skipping", src)
            continue
        try:
            docs = fn(src)
        except Exception as exc:
            log.error("Error flattening %s: %s", filename, exc)
            continue

        out = OUT_DIR / filename
        out.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Wrote %d docs → %s", len(docs), out.name)
        total += len(docs)

    log.info("Done. Total flat documents: %d → %s", total, OUT_DIR)


if __name__ == "__main__":
    main()
