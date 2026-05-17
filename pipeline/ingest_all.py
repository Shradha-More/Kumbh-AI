"""
ingest_all.py — Master ingestion script for the Kumbh Mela 2027 knowledge base.

Reads ALL JSON files from data/, flattens them into per-language documents,
runs the full pipeline (clean -> chunk -> deduplicate), and re-ingests into ChromaDB.

Usage:
    python pipeline/ingest_all.py
    python pipeline/ingest_all.py --skip-chroma   # skip ChromaDB ingestion
    python pipeline/ingest_all.py --only-flatten   # just flatten, don't run pipeline
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
SEED_OUT_DIR = PROJECT_ROOT / "knowledge_base" / "raw" / "seed"

LANG_SUFFIX = {
    "en": "_en", "hi": "_hi", "mr": "_mr", "gu": "_gu",
    "ta": "_ta", "te": "_te", "kn": "_kn", "ml": "_ml",
}

# All known JSON data files (existing + new massive ones)
KNOWN_DATA_FILES = [
    "kumbh_2027_schedule.json",
    "nashik_places.json",
    "emergency_responses.json",
    "ghats_and_transport.json",
    "nashik_complete_places.json",
    "nashik_culture_history.json",
    "kumbh_2027_detailed.json",
    "nashik_routes_transport.json",
    "nashik_food_wine.json",
]

# Text fields to harvest from any JSON object, in priority order
TEXT_FIELDS = [
    "name", "title", "heading", "label",
    "description", "content", "text", "body", "summary", "overview",
    "significance", "history", "about", "details", "info", "note", "notes",
    "how_to_reach", "directions", "route", "access",
    "tips", "advice", "recommendations", "best_time",
    "address", "location", "area", "region",
    "timings", "timing", "hours", "schedule", "opening_hours",
    "entry_fee", "ticket", "cost", "price", "fee",
    "facilities", "amenities", "services",
    "specialties", "features", "highlights", "attractions",
    "cuisine", "food", "menu", "dishes",
    "ingredients", "recipe", "preparation",
    "wine", "varieties", "grapes",
    "response", "instructions", "steps", "procedure",
    "rituals", "traditions", "customs", "practices",
    "mythology", "legend", "story", "stories",
    "architecture", "structure", "design",
    "flora", "fauna", "wildlife", "nature",
    "distance", "duration", "frequency",
    "contact", "phone", "website", "email",
    "capacity", "crowd_level", "estimated_pilgrims",
    "bathing_rules", "dress_code", "rules",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _str(val: Any) -> str:
    """Convert a value to a string, flattening lists."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (list, tuple)):
        return ", ".join(_str(v) for v in val if v)
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            sv = _str(v)
            if sv:
                parts.append(f"{k}: {sv}")
        return "; ".join(parts)
    return str(val).strip()


def _gather_text_parts(obj: dict, suffix: str = "") -> list[str]:
    """Gather all text from an object using known field names + optional lang suffix."""
    parts: list[str] = []
    for base_field in TEXT_FIELDS:
        # Try suffixed version first (e.g., description_en), then plain
        for key in [f"{base_field}{suffix}", base_field]:
            val = obj.get(key)
            if val is not None:
                text = _str(val)
                if text and len(text) > 2:
                    parts.append(text)
                break  # don't add both suffixed and plain for same field
    return parts


def _gather_all_text(obj: dict, suffix: str = "") -> str:
    """Build the LONGEST possible content string from an object."""
    parts = _gather_text_parts(obj, suffix)

    # Also grab any remaining string fields not in TEXT_FIELDS
    seen_keys = set()
    for base_field in TEXT_FIELDS:
        seen_keys.add(f"{base_field}{suffix}")
        seen_keys.add(base_field)

    for key, val in obj.items():
        if key in seen_keys or key.startswith("_") or key == "id":
            continue
        if key in ("coordinates", "lat", "lon", "kumbh_only"):
            continue
        text = _str(val)
        if text and len(text) > 10:
            parts.append(f"{key}: {text}")

    return "\n".join(p for p in parts if p)


def _domain_from_filename(filename: str) -> str:
    """Infer domain from filename."""
    name = filename.lower()
    if "emergency" in name:
        return "emergency"
    if "transport" in name or "route" in name:
        return "transport"
    if "schedule" in name or "kumbh_2027" in name:
        return "schedule"
    if "food" in name or "wine" in name or "cuisine" in name:
        return "food"
    if "culture" in name or "history" in name:
        return "culture"
    if "place" in name or "ghat" in name:
        return "places"
    return "general"


def _emit(docs: list[dict], lang: str, title: str, content: str,
          domain: str, source: str) -> None:
    """Append a document if content is substantial enough (>= 15 words)."""
    content = (content or "").strip()
    if len(content.split()) < 15:
        return
    docs.append({
        "language": lang,
        "title": title.strip() if title else "Untitled",
        "content": content,
        "domain": domain,
        "source_url": f"seed:{source}",
    })


# ---------------------------------------------------------------------------
# Generic flattener that handles any JSON structure
# ---------------------------------------------------------------------------

def flatten_generic_json(data: Any, filename: str) -> list[dict]:
    """Flatten any JSON structure into per-language documents."""
    docs: list[dict] = []
    domain = _domain_from_filename(filename)
    source = filename.replace(".json", "")

    if isinstance(data, list):
        _flatten_array(data, docs, domain, source)
    elif isinstance(data, dict):
        _flatten_dict(data, docs, domain, source)
    return docs


def _flatten_array(arr: list, docs: list[dict], domain: str, source: str) -> None:
    """Flatten an array of objects."""
    for i, item in enumerate(arr):
        if isinstance(item, dict):
            _flatten_entry(item, docs, domain, source, idx=i)
        elif isinstance(item, str) and len(item) > 30:
            _emit(docs, "en", f"Entry {i}", item, domain, source)


def _flatten_dict(obj: dict, docs: list[dict], domain: str, source: str) -> None:
    """Flatten a top-level dict, looking for nested arrays and objects."""
    # First, try to extract top-level text fields per language
    top_text_found = False
    for lang, suffix in LANG_SUFFIX.items():
        content = _gather_all_text(obj, suffix)
        if content and len(content.split()) >= 15:
            name = obj.get(f"name{suffix}") or obj.get("name") or obj.get(f"event{suffix}") or obj.get("event") or obj.get(f"title{suffix}") or obj.get("title") or "Overview"
            _emit(docs, lang, _str(name), content, domain, source)
            top_text_found = True

    if not top_text_found:
        # Try just English
        content = _gather_all_text(obj, "_en")
        if not content or len(content.split()) < 15:
            content = _gather_all_text(obj, "")
        if content and len(content.split()) >= 15:
            name = obj.get("name") or obj.get("event") or obj.get("title") or "Overview"
            _emit(docs, "en", _str(name), content, domain, source)

    # Now recurse into nested arrays and dicts
    for key, val in obj.items():
        if isinstance(val, list) and val:
            sub_domain = _infer_subdomain(key, domain)
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    _flatten_entry(item, docs, sub_domain, f"{source}:{key}", idx=i)
                elif isinstance(item, str) and len(item) > 50:
                    _emit(docs, "en", f"{key} item {i}", item, sub_domain, f"{source}:{key}")
        elif isinstance(val, dict) and not key.startswith("_") and key not in ("coordinates", "main_period"):
            # Nested object — could be a subsection
            sub_domain = _infer_subdomain(key, domain)
            _flatten_dict(val, docs, sub_domain, f"{source}:{key}")


def _flatten_entry(entry: dict, docs: list[dict], domain: str, source: str,
                   idx: int = 0) -> None:
    """Flatten a single entry object (place, event, scenario, etc.) into per-language docs."""
    entry_id = entry.get("id", f"item_{idx}")
    has_multilingual = False

    for lang, suffix in LANG_SUFFIX.items():
        content = _gather_all_text(entry, suffix)
        if content and len(content.split()) >= 15:
            name = (entry.get(f"name{suffix}")
                    or entry.get(f"title{suffix}")
                    or entry.get("name")
                    or entry.get("title")
                    or f"Entry {entry_id}")
            _emit(docs, lang, _str(name), content, domain, f"{source}:{entry_id}")
            has_multilingual = True

    # If no per-language content found, try generic fields
    if not has_multilingual:
        content = _gather_all_text(entry, "")
        if content and len(content.split()) >= 15:
            name = entry.get("name") or entry.get("title") or f"Entry {entry_id}"
            _emit(docs, "en", _str(name), content, domain, f"{source}:{entry_id}")

    # Recurse into nested arrays within this entry
    for key, val in entry.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            sub_domain = _infer_subdomain(key, domain)
            for i, sub_item in enumerate(val):
                if isinstance(sub_item, dict):
                    _flatten_entry(sub_item, docs, sub_domain,
                                   f"{source}:{entry_id}:{key}", idx=i)


def _infer_subdomain(key: str, parent_domain: str) -> str:
    """Infer a subdomain from a dict key."""
    k = key.lower()
    if any(w in k for w in ("hospital", "police", "emergency", "helpline", "ambulance")):
        return "emergency"
    if any(w in k for w in ("transport", "bus", "train", "railway", "taxi", "route", "parking")):
        return "transport"
    if any(w in k for w in ("ghat", "temple", "place", "monument", "site")):
        return "places"
    if any(w in k for w in ("schedule", "snan", "akhara", "shahi", "event", "date")):
        return "schedule"
    if any(w in k for w in ("food", "wine", "restaurant", "cuisine", "dish")):
        return "food"
    if any(w in k for w in ("culture", "history", "tradition", "festival")):
        return "culture"
    if any(w in k for w in ("accommodation", "hotel", "dharamshala", "stay", "lodge")):
        return "accommodation"
    return parent_domain


# ---------------------------------------------------------------------------
# Specialized flatteners for known file structures (reuse existing logic)
# ---------------------------------------------------------------------------

def flatten_schedule(path: Path) -> list[dict]:
    """Use existing flatten_seed.py logic for schedule, enhanced."""
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = flatten_generic_json(data, path.name)
    log.info("  %s -> %d documents (generic)", path.name, len(docs))
    return docs


def flatten_places(path: Path) -> list[dict]:
    """Enhanced places flattener — extracts maximum content per entry."""
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = flatten_generic_json(data, path.name)
    log.info("  %s -> %d documents (generic)", path.name, len(docs))
    return docs


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def discover_data_files() -> list[Path]:
    """Find all JSON files in data/ (excluding synthetic_qa, seed, and db files)."""
    files: list[Path] = []
    if not DATA_DIR.exists():
        log.error("Data directory not found: %s", DATA_DIR)
        return files

    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix == ".json" and path.is_file():
            files.append(path)

    # Also check for known files that might not exist yet
    for name in KNOWN_DATA_FILES:
        candidate = DATA_DIR / name
        if candidate.exists() and candidate not in files:
            files.append(candidate)

    return files


def flatten_all_data_files() -> tuple[list[dict], dict[str, int]]:
    """Read and flatten ALL data JSON files."""
    all_docs: list[dict] = []
    file_counts: dict[str, int] = {}

    data_files = discover_data_files()
    if not data_files:
        log.error("No JSON data files found in %s", DATA_DIR)
        return all_docs, file_counts

    log.info("Found %d JSON data files to process:", len(data_files))
    for f in data_files:
        log.info("  - %s", f.name)

    for data_file in data_files:
        try:
            raw = json.loads(data_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to read %s: %s", data_file, exc)
            continue

        docs = flatten_generic_json(raw, data_file.name)
        file_counts[data_file.name] = len(docs)
        all_docs.extend(docs)
        log.info("  %s -> %d flat documents", data_file.name, len(docs))

    return all_docs, file_counts


def deduplicate_flat_docs(docs: list[dict]) -> list[dict]:
    """Remove exact duplicate documents (same content hash)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for doc in docs:
        h = hashlib.md5(doc["content"].lower().strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    removed = len(docs) - len(unique)
    if removed > 0:
        log.info("  Removed %d exact-duplicate flat documents", removed)
    return unique


def write_seed_files(docs: list[dict]) -> dict[str, int]:
    """Write flattened docs to knowledge_base/raw/seed/ grouped by domain."""
    SEED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group by source file domain
    domain_groups: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        domain = doc.get("domain", "general")
        domain_groups[domain].append(doc)

    file_counts: dict[str, int] = {}
    for domain, group_docs in domain_groups.items():
        out_path = SEED_OUT_DIR / f"{domain}_flattened.json"
        out_path.write_text(
            json.dumps(group_docs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        file_counts[domain] = len(group_docs)
        log.info("  Wrote %d docs -> %s", len(group_docs), out_path.name)

    return file_counts


def run_pipeline_stage(script_path: Path, stage_name: str) -> bool:
    """Run a pipeline stage script."""
    if not script_path.exists():
        log.warning("Script not found: %s — skipping %s", script_path, stage_name)
        return False

    log.info("Running %s ...", stage_name)
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        log.error("%s FAILED (exit %d, %.1fs)", stage_name, result.returncode, elapsed)
        return False

    log.info("%s completed in %.1fs", stage_name, elapsed)
    return True


def run_chroma_ingest() -> int:
    """Run ChromaDB ingestion and return the document count."""
    chroma_script = PROJECT_ROOT / "vectordb" / "ingest_chroma.py"
    if not chroma_script.exists():
        log.warning("ChromaDB ingest script not found: %s", chroma_script)
        return -1

    log.info("Running ChromaDB ingestion...")
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(chroma_script)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        log.error("ChromaDB ingestion FAILED (exit %d, %.1fs)", result.returncode, elapsed)
        return -1

    log.info("ChromaDB ingestion completed in %.1fs", elapsed)

    # Try to get ChromaDB count
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import chromadb
        from chromadb.config import Settings
        chroma_dir = PROJECT_ROOT / "vectordb" / "chroma_db"
        if chroma_dir.exists():
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection(name="kumbh_mela_2027")
            return collection.count()
    except Exception as exc:
        log.warning("Could not read ChromaDB count: %s", exc)

    return -1


def count_items_in_dir(directory: Path) -> int:
    """Count JSON items in a directory."""
    total = 0
    if not directory.exists():
        return 0
    for path in directory.rglob("*.json"):
        if "manifest" in path.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            total += len(data) if isinstance(data, list) else 1
        except Exception:
            pass
    return total


def print_summary(
    all_docs: list[dict],
    file_counts: dict[str, int],
    seed_counts: dict[str, int],
    chroma_count: int,
    elapsed: float,
) -> None:
    """Print a comprehensive summary."""
    # Per-language breakdown
    lang_counts: dict[str, int] = defaultdict(int)
    domain_counts: dict[str, int] = defaultdict(int)
    for doc in all_docs:
        lang_counts[doc.get("language", "unknown")] += 1
        domain_counts[doc.get("domain", "general")] += 1

    cleaned_count = count_items_in_dir(PROJECT_ROOT / "knowledge_base" / "cleaned")
    chunked_count = count_items_in_dir(PROJECT_ROOT / "knowledge_base" / "chunked")
    dedup_count = count_items_in_dir(PROJECT_ROOT / "knowledge_base" / "deduplicated")

    print("\n" + "=" * 72)
    print("  KUMBH MELA 2027 — MASTER INGESTION SUMMARY")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Duration: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 72)

    print("\n  Source files processed:")
    for fname, count in sorted(file_counts.items()):
        print(f"    {fname:<40} {count:>6} docs")
    print(f"    {'TOTAL':<40} {sum(file_counts.values()):>6} docs")

    print(f"\n  After deduplication: {len(all_docs):>6} flat documents")

    print("\n  Seed files written (by domain):")
    for domain, count in sorted(seed_counts.items()):
        print(f"    {domain:<20} {count:>6} docs")

    print("\n  Per-language breakdown:")
    for lang in ["en", "hi", "mr", "gu", "ta", "te", "kn", "ml"]:
        count = lang_counts.get(lang, 0)
        if count > 0:
            print(f"    {lang:<10} {count:>6}")
    other = sum(c for l, c in lang_counts.items()
                if l not in ["en", "hi", "mr", "gu", "ta", "te", "kn", "ml"])
    if other:
        print(f"    {'other':<10} {other:>6}")

    print("\n  Per-domain breakdown:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {domain:<20} {count:>6}")

    print("\n  Pipeline stage outputs:")
    print(f"    Cleaned documents  : {cleaned_count:>8,}")
    print(f"    Chunked documents  : {chunked_count:>8,}")
    print(f"    After dedup chunks : {dedup_count:>8,}")

    if chroma_count >= 0:
        print(f"\n  ChromaDB total docs  : {chroma_count:>8,}")
    else:
        print("\n  ChromaDB: not available / skipped")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Kumbh Mela 2027 — Master ingestion pipeline")
    p.add_argument("--skip-chroma", action="store_true",
                   help="Skip ChromaDB re-ingestion step")
    p.add_argument("--only-flatten", action="store_true",
                   help="Only flatten data files, skip pipeline and ChromaDB")
    p.add_argument("--skip-pipeline", action="store_true",
                   help="Skip clean/chunk/dedup pipeline (only flatten + chroma)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("\n" + "=" * 72)
    print("  KUMBH MELA 2027 — MASTER INGESTION")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 72 + "\n")

    # --- Step 1: Flatten all data files ---
    log.info("STEP 1: Flattening all data JSON files...")
    all_docs, file_counts = flatten_all_data_files()

    if not all_docs:
        log.error("No documents extracted. Check your data files in %s", DATA_DIR)
        sys.exit(1)

    log.info("Total raw flat documents: %d", len(all_docs))

    # --- Step 2: Deduplicate flat docs ---
    log.info("STEP 2: Deduplicating flat documents...")
    all_docs = deduplicate_flat_docs(all_docs)
    log.info("Unique flat documents: %d", len(all_docs))

    # --- Step 3: Write seed files ---
    log.info("STEP 3: Writing seed files to %s", SEED_OUT_DIR)
    seed_counts = write_seed_files(all_docs)

    if args.only_flatten:
        print_summary(all_docs, file_counts, seed_counts, -1, time.time() - t0)
        log.info("Done (flatten only).")
        return

    # --- Step 4: Run pipeline stages ---
    chroma_count = -1

    if not args.skip_pipeline:
        log.info("STEP 4: Running pipeline stages (clean -> chunk -> deduplicate)...")

        stages = [
            (PROJECT_ROOT / "pipeline" / "clean.py", "CLEAN (stage 1)"),
            (PROJECT_ROOT / "pipeline" / "chunk.py", "CHUNK (stage 2)"),
            (PROJECT_ROOT / "pipeline" / "deduplicate.py", "DEDUPLICATE (stage 3)"),
        ]

        for script, name in stages:
            success = run_pipeline_stage(script, name)
            if not success:
                log.error("Pipeline halted at %s", name)
                break
    else:
        log.info("STEP 4: Skipped pipeline (--skip-pipeline)")

    # --- Step 5: ChromaDB ingestion ---
    if not args.skip_chroma:
        log.info("STEP 5: Re-ingesting into ChromaDB...")
        chroma_count = run_chroma_ingest()
    else:
        log.info("STEP 5: Skipped ChromaDB (--skip-chroma)")

    # --- Summary ---
    print_summary(all_docs, file_counts, seed_counts, chroma_count, time.time() - t0)
    log.info("Master ingestion complete.")


if __name__ == "__main__":
    main()
