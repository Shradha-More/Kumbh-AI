"""
clean.py — Stage 1 of the Kumbh Mela 2027 data pipeline.

Reads raw JSON documents from knowledge_base/raw/, cleans and normalises them,
detects language, redacts personal data, and writes results to
knowledge_base/cleaned/ while preserving subfolder structure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from langdetect import DetectorFactory, detect, LangDetectException
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "knowledge_base" / "raw"
CLEANED_DIR  = PROJECT_ROOT / "knowledge_base" / "cleaned"

MIN_WORDS  = 50
MAX_WORDS  = 5000

_PHONE_RE       = re.compile(r"(?<!\w)(\+91[-\s]?)?[6-9]\d{9}|(\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)", re.ASCII)
_EMAIL_RE       = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.ASCII)
_HTML_TAG_RE    = re.compile(r"<[^>]+>", re.DOTALL)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE    = re.compile(r"\n{3,}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    for ent, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&nbsp;", " ")]:
        text = text.replace(ent, char)
    return text


def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalise_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def redact_personal_data(text: str) -> str:
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    return text


def remove_duplicate_paragraphs(text: str) -> str:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    seen: set[str] = set()
    unique: list[str] = []
    for para in paragraphs:
        key = re.sub(r"\s+", " ", para).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(para)
    return "\n\n".join(unique)


def count_words(text: str) -> int:
    return len(text.split())


def split_long_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_wc = 0
    for para in paragraphs:
        wc = count_words(para)
        if current_wc + wc > max_words and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_wc = 0
        if wc > max_words:
            words = para.split()
            for start in range(0, len(words), max_words):
                chunks.append(" ".join(words[start:start + max_words]))
        else:
            current_parts.append(para)
            current_wc += wc
    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks


def detect_language(text: str) -> str | None:
    if not LANGDETECT_AVAILABLE:
        return None
    try:
        return detect(text[:1000])
    except Exception:
        return None


def clean_document(raw: dict[str, Any]) -> list[dict[str, Any]]:
    text: str = raw.get("content") or raw.get("text") or raw.get("body") or ""
    if not isinstance(text, str):
        text = str(text)

    text = strip_html(text)
    text = normalise_unicode(text)
    text = redact_personal_data(text)
    text = normalise_whitespace(text)
    text = remove_duplicate_paragraphs(text)

    if count_words(text) < MIN_WORDS:
        return []

    segments = split_long_text(text) if count_words(text) > MAX_WORDS else [text]
    lang = raw.get("language") or detect_language(text)

    results: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        if count_words(segment) < MIN_WORDS:
            continue
        doc_id = raw.get("id") or hashlib.md5(segment.encode()).hexdigest()[:12]
        seg_id = f"{doc_id}_part{idx}" if len(segments) > 1 else doc_id
        results.append({
            "id":         seg_id,
            "text":       segment,
            "language":   lang,
            "domain":     raw.get("domain", "general"),
            "source_url": raw.get("source_url") or raw.get("url", ""),
            "title":      raw.get("title", ""),
            "word_count": count_words(segment),
        })
    return results


class Stats:
    def __init__(self):
        self.input_docs = 0
        self.output_docs = 0
        self.removed_docs = 0
        self.lang_counts: dict[str, int] = defaultdict(int)

    def record(self, raw_count: int, cleaned: list[dict[str, Any]]) -> None:
        self.input_docs += raw_count
        self.output_docs += len(cleaned)
        self.removed_docs += max(0, raw_count - len(cleaned))
        for doc in cleaned:
            self.lang_counts[doc.get("language") or "unknown"] += 1

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  CLEANING PIPELINE — SUMMARY")
        print("=" * 60)
        print(f"  Input  documents : {self.input_docs:>8,}")
        print(f"  Output documents : {self.output_docs:>8,}")
        print(f"  Removed          : {self.removed_docs:>8,}")
        print("\n  Per-language breakdown:")
        for lang, count in sorted(self.lang_counts.items(), key=lambda x: -x[1]):
            print(f"    {lang:<10} {count:>6,}")
        print("=" * 60 + "\n")


def main() -> None:
    if not RAW_DIR.exists():
        log.error("Raw directory not found: %s", RAW_DIR)
        sys.exit(1)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    stats = Stats()
    seen_hashes: set[str] = set()
    raw_files = list(RAW_DIR.rglob("*.json"))

    if not raw_files:
        log.warning("No JSON files found under %s", RAW_DIR)
        return

    log.info("Found %d raw JSON file(s) — beginning cleaning…", len(raw_files))

    for raw_path in raw_files:
        try:
            with raw_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            raw_docs = data if isinstance(data, list) else [data]
        except Exception as exc:
            log.warning("Skipping %s — %s", raw_path, exc)
            continue

        cleaned_docs: list[dict[str, Any]] = []
        for raw_doc in raw_docs:
            for cdoc in clean_document(raw_doc):
                fp = hashlib.md5(re.sub(r"\s+", " ", cdoc["text"]).encode()).hexdigest()
                if fp in seen_hashes:
                    continue
                seen_hashes.add(fp)
                cleaned_docs.append(cdoc)

        stats.record(len(raw_docs), cleaned_docs)

        if cleaned_docs:
            relative = raw_path.relative_to(RAW_DIR)
            out_path = CLEANED_DIR / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(cleaned_docs, fh, ensure_ascii=False, indent=2)
            log.info("%s  →  %d doc(s) cleaned", raw_path.name, len(cleaned_docs))

    stats.print_summary()


if __name__ == "__main__":
    main()
