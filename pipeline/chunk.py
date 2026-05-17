"""
chunk.py — Stage 2 of the Kumbh Mela 2027 data pipeline.

Reads cleaned documents from knowledge_base/cleaned/, splits them into
300-500 word chunks with 50-word overlap (respecting sentence boundaries),
and writes JSON chunk files + a manifest to knowledge_base/chunked/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CLEANED_DIR   = PROJECT_ROOT / "knowledge_base" / "cleaned"
CHUNKED_DIR   = PROJECT_ROOT / "knowledge_base" / "chunked"
MANIFEST_FILE = CHUNKED_DIR / "chunked_manifest.json"

TARGET_MIN = 300
TARGET_MAX = 500
OVERLAP    = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Sentence boundary: after .!?।॥ followed by whitespace + uppercase/letter
_SENT_BOUNDARY = re.compile(
    r'(?<=[.!?।॥\u0964\u0965])\s+(?=[A-Za-z\u0900-\u097F\u0B80-\u0BFF'
    r'\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0A80-\u0AFF\u0B00-\u0B7F"\'0-9])'
)


def split_sentences(text: str) -> list[str]:
    parts = _SENT_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def make_chunks(text: str, target_min: int = TARGET_MIN, target_max: int = TARGET_MAX, overlap: int = OVERLAP) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    sw_pairs = [(s, len(s.split())) for s in sentences]
    chunks: list[str] = []
    start_idx = 0

    while start_idx < len(sw_pairs):
        chunk_sents: list[str] = []
        word_count = 0
        idx = start_idx

        while idx < len(sw_pairs):
            sent, wc = sw_pairs[idx]
            if word_count + wc > target_max and word_count >= target_min:
                break
            chunk_sents.append(sent)
            word_count += wc
            idx += 1

        if not chunk_sents:
            chunk_sents = [sw_pairs[start_idx][0]]
            idx = start_idx + 1

        chunks.append(" ".join(chunk_sents))

        # Rewind by ~overlap words for next chunk start
        overlap_words = 0
        next_start = idx
        while next_start > start_idx + 1:
            overlap_words += sw_pairs[next_start - 1][1]
            if overlap_words >= overlap:
                break
            next_start -= 1

        start_idx = max(start_idx + 1, next_start)

    return chunks


def chunk_id_for(parent_id: str, index: int) -> str:
    return hashlib.sha1(f"{parent_id}__chunk_{index:04d}".encode()).hexdigest()[:16]


def process_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    text = doc.get("text", "")
    parent_id = doc.get("id", hashlib.md5(text.encode()).hexdigest()[:12])
    raw_chunks = make_chunks(text)

    result: list[dict[str, Any]] = []
    for i, chunk_text in enumerate(raw_chunks):
        result.append({
            "id":           chunk_id_for(parent_id, i),
            "text":         chunk_text,
            "language":     doc.get("language"),
            "domain":       doc.get("domain", "general"),
            "source_url":   doc.get("source_url", ""),
            "chunk_index":  i,
            "total_chunks": len(raw_chunks),
            "parent_doc_id": parent_id,
            "word_count":   len(chunk_text.split()),
        })
    return result


def main() -> None:
    if not CLEANED_DIR.exists():
        log.error("Cleaned directory not found: %s", CLEANED_DIR)
        sys.exit(1)

    CHUNKED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_files = list(CLEANED_DIR.rglob("*.json"))

    if not cleaned_files:
        log.warning("No cleaned JSON files found under %s", CLEANED_DIR)
        return

    log.info("Found %d cleaned file(s) — chunking…", len(cleaned_files))

    total_docs = 0
    total_chunks = 0
    lang_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}

    for cleaned_path in cleaned_files:
        try:
            with cleaned_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            docs = data if isinstance(data, list) else [data]
        except Exception as exc:
            log.warning("Skipping %s — %s", cleaned_path, exc)
            continue

        file_chunks: list[dict[str, Any]] = []
        for doc in docs:
            total_docs += 1
            chunks = process_document(doc)
            file_chunks.extend(chunks)
            lang = doc.get("language") or "unknown"
            domain = doc.get("domain") or "general"
            lang_counts[lang] = lang_counts.get(lang, 0) + len(chunks)
            domain_counts[domain] = domain_counts.get(domain, 0) + len(chunks)

        total_chunks += len(file_chunks)

        relative = cleaned_path.relative_to(CLEANED_DIR)
        out_path = CHUNKED_DIR / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(file_chunks, fh, ensure_ascii=False, indent=2)

        log.info("%s  →  %d chunk(s)", cleaned_path.name, len(file_chunks))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "chunk_params": {"target_min_words": TARGET_MIN, "target_max_words": TARGET_MAX, "overlap_words": OVERLAP},
        "language_breakdown": lang_counts,
        "domain_breakdown": domain_counts,
    }
    with MANIFEST_FILE.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  CHUNKING PIPELINE — SUMMARY")
    print("=" * 60)
    print(f"  Source documents : {total_docs:>8,}")
    print(f"  Total chunks     : {total_chunks:>8,}")
    print("\n  Chunks per language:")
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang:<12} {cnt:>6,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
