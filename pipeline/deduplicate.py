"""
deduplicate.py — Stage 3 of the Kumbh Mela 2027 data pipeline.

Removes exact duplicates (MD5) and near-duplicates (MinHash / Jaccard > 0.80)
from chunked documents. Saves to knowledge_base/deduplicated/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    logging.warning("datasketch not installed — near-duplicate detection disabled. pip install datasketch")

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CHUNKED_DIR    = PROJECT_ROOT / "knowledge_base" / "chunked"
DEDUP_DIR      = PROJECT_ROOT / "knowledge_base" / "deduplicated"

JACCARD_THRESH = 0.80
NUM_PERM       = 128

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", text.lower().strip())


def md5_of(text: str) -> str:
    return hashlib.md5(normalise(text).encode()).hexdigest()


def shingles(text: str, k: int = 5) -> set[bytes]:
    norm = normalise(text)
    return {norm[i:i+k].encode() for i in range(max(1, len(norm) - k + 1))}


def make_minhash(text: str) -> "MinHash":
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles(text):
        m.update(s)
    return m


def main() -> None:
    if not CHUNKED_DIR.exists():
        log.error("Chunked directory not found: %s", CHUNKED_DIR)
        sys.exit(1)

    DEDUP_DIR.mkdir(parents=True, exist_ok=True)

    chunk_files = [p for p in CHUNKED_DIR.rglob("*.json") if p.name != "chunked_manifest.json"]
    if not chunk_files:
        log.warning("No chunk JSON files found under %s", CHUNKED_DIR)
        return

    # Pass 1: load all + exact dedup
    log.info("Pass 1 — loading %d files and removing exact duplicates…", len(chunk_files))
    all_chunks: list[dict[str, Any]] = []
    source_map: dict[str, Path] = {}
    seen_md5: set[str] = set()
    total_loaded = 0

    for path in chunk_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            chunks = data if isinstance(data, list) else [data]
        except Exception as exc:
            log.warning("Skipping %s — %s", path, exc)
            continue

        for chunk in chunks:
            total_loaded += 1
            h = md5_of(chunk.get("text", ""))
            if h in seen_md5:
                continue
            seen_md5.add(h)
            all_chunks.append(chunk)
            source_map[chunk["id"]] = path

    exact_removed = total_loaded - len(all_chunks)
    log.info("  Removed %d exact duplicates → %d remain", exact_removed, len(all_chunks))

    # Pass 2: MinHash near-dedup
    near_removed = 0
    if DATASKETCH_AVAILABLE:
        log.info("Pass 2 — MinHash near-dedup (threshold=%.0f%%)…", JACCARD_THRESH * 100)
        lsh = MinHashLSH(threshold=JACCARD_THRESH, num_perm=NUM_PERM)
        to_keep: list[dict[str, Any]] = []

        for chunk in all_chunks:
            cid = chunk["id"]
            mh = make_minhash(chunk.get("text", ""))
            try:
                if lsh.query(mh):
                    near_removed += 1
                    continue
            except Exception:
                pass
            try:
                lsh.insert(cid, mh)
            except ValueError:
                pass
            to_keep.append(chunk)

        log.info("  Removed %d near-duplicates → %d remain", near_removed, len(to_keep))
    else:
        log.warning("datasketch not available — skipping near-duplicate pass.")
        to_keep = all_chunks

    # Write output
    file_groups: dict[Path, list[dict[str, Any]]] = {}
    for chunk in to_keep:
        src = source_map.get(chunk["id"])
        if src:
            file_groups.setdefault(src, []).append(chunk)

    for src_path, chunks in file_groups.items():
        relative = src_path.relative_to(CHUNKED_DIR)
        out_path = DEDUP_DIR / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(chunks, fh, ensure_ascii=False, indent=2)

    total_removed = exact_removed + near_removed
    print("\n" + "=" * 60)
    print("  DEDUPLICATION — SUMMARY")
    print("=" * 60)
    print(f"  Total loaded     : {total_loaded:>8,}")
    print(f"  Exact removed    : {exact_removed:>8,}")
    print(f"  Near-dup removed : {near_removed:>8,}")
    print(f"  Remaining        : {len(to_keep):>8,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
