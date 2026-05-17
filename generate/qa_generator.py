"""
qa_generator.py — Stage 5 of the Kumbh Mela 2027 data pipeline.

Reads chunks from knowledge_base/translated/ and knowledge_base/deduplicated/,
calls local Ollama (qwen2.5:7b) to generate 5-8 multilingual QA pairs per chunk,
and saves Alpaca-format JSONL to data/synthetic_qa/.

Target: 15,000–25,000 QA pairs across 8 languages.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEDUP_DIR    = PROJECT_ROOT / "knowledge_base" / "deduplicated"
TRANS_DIR    = PROJECT_ROOT / "knowledge_base" / "translated"
QA_DIR       = PROJECT_ROOT / "data" / "synthetic_qa"

OLLAMA_BASE  = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"  # overridable via --model CLI arg
TARGET_MAX   = 25_000

SUPPORTED_LANGUAGES = ["en", "hi", "mr", "gu", "ta", "te", "kn", "ml"]
QA_TYPES = ["factual", "procedural", "comparative", "emergency", "recommendation", "timing", "conversational", "followup"]

LANG_NAMES = {
    "en": "English", "hi": "Hindi (हिंदी)", "mr": "Marathi (मराठी)",
    "gu": "Gujarati (ગુજરાતી)", "ta": "Tamil (தமிழ்)", "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)", "ml": "Malayalam (മലയാളം)",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def build_prompt(chunk_text: str, language: str, qa_type: str, num_pairs: int) -> str:
    lang_name = LANG_NAMES.get(language, "English")
    type_hints = {
        "factual": "Direct factual question answered from the text",
        "procedural": "How-to question answered with numbered steps",
        "comparative": "Comparison between two Kumbh-related places or events",
        "emergency": "What to do in an emergency situation",
        "recommendation": "What to see/do first at the Kumbh",
        "timing": "Best time to visit a place or witness an event",
        "conversational": "Natural spoken question, as if asking a guide",
        "followup": "Follow-up question referencing a previous answer",
    }
    type_hint = type_hints.get(qa_type, qa_type)

    return f"""You are an expert QA dataset creator for the Nashik Kumbh Mela 2027 multilingual AI assistant.

CONTEXT:
\"\"\"
{chunk_text}
\"\"\"

TASK: Generate exactly {num_pairs} question-answer pairs in {lang_name}.
QA type: {qa_type} — {type_hint}

Rules:
1. Questions and answers MUST be in {lang_name} only.
2. Answers must be grounded in the provided context.
3. Format as JSON array: [{{"q": "...", "a": "..."}}]
4. Output ONLY the JSON array — no other text.

JSON array:"""


def ollama_generate(prompt: str, retries: int = 3) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 1024},
    }).encode()

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode()).get("response", "")
        except Exception as exc:
            log.warning("Ollama attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return ""


def parse_qa_pairs(response: str) -> list[dict[str, str]]:
    for text in [response.strip(), re.search(r"\[.*?\]", response, re.DOTALL) and re.search(r"\[.*?\]", response, re.DOTALL).group()]:
        if not text:
            continue
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [{"q": str(item.get("q", "")), "a": str(item.get("a", ""))}
                        for item in data if isinstance(item, dict) and item.get("q") and item.get("a")]
        except json.JSONDecodeError:
            continue
    return []


def iter_all_chunks() -> Iterator[dict[str, Any]]:
    for path in sorted(DEDUP_DIR.rglob("*.json")) if DEDUP_DIR.exists() else []:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for chunk in (data if isinstance(data, list) else [data]):
                if isinstance(chunk, dict) and chunk.get("text"):
                    chunk.setdefault("language", "en")
                    yield chunk
        except Exception:
            continue

    if TRANS_DIR.exists():
        for lang_dir in sorted(TRANS_DIR.iterdir()):
            if not lang_dir.is_dir():
                continue
            for path in sorted(lang_dir.rglob("*.json")):
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    for chunk in (data if isinstance(data, list) else [data]):
                        if isinstance(chunk, dict) and chunk.get("text"):
                            yield chunk
                except Exception:
                    continue


def main() -> None:
    import argparse
    global OLLAMA_MODEL, TARGET_MAX
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model to use")
    p.add_argument("--max-pairs", type=int, default=TARGET_MAX)
    args = p.parse_args()
    OLLAMA_MODEL = args.model
    TARGET_MAX = args.max_pairs

    QA_DIR.mkdir(parents=True, exist_ok=True)

    writers = {lang: (QA_DIR / f"{lang}_qa_pairs.jsonl").open("a", encoding="utf-8")
               for lang in SUPPORTED_LANGUAGES}
    lang_counts: dict[str, int] = {lang: 0 for lang in SUPPORTED_LANGUAGES}
    total_pairs = 0

    all_chunks = list(iter_all_chunks())
    if not all_chunks:
        log.error("No chunks found. Run pipeline stages 1-4 first.")
        for fh in writers.values():
            fh.close()
        sys.exit(1)

    log.info("Loaded %d chunks — targeting %d QA pairs", len(all_chunks), TARGET_MAX)
    random.shuffle(all_chunks)

    chunk_iter = tqdm(all_chunks, desc="Generating QA", unit="chunk") if TQDM_AVAILABLE else all_chunks

    for i, chunk in enumerate(chunk_iter):
        if total_pairs >= TARGET_MAX:
            break

        qa_type = QA_TYPES[i % len(QA_TYPES)]
        num_pairs = random.randint(3, 5)
        language = chunk.get("language", "en")
        domain = chunk.get("domain", "general")

        prompt = build_prompt(chunk.get("text", ""), language, qa_type, num_pairs)
        response = ollama_generate(prompt)
        raw_pairs = parse_qa_pairs(response)

        for pair in raw_pairs:
            q, a = pair["q"].strip(), pair["a"].strip()
            if not q or not a:
                continue
            record = {
                "instruction": q, "input": "", "output": a,
                "language": language, "domain": domain, "type": qa_type,
                "source_url": chunk.get("source_url", ""), "chunk_id": chunk.get("id", ""),
            }
            fh = writers.get(language)
            if fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            lang_counts[language] = lang_counts.get(language, 0) + 1
            total_pairs += 1

        if TQDM_AVAILABLE:
            chunk_iter.set_postfix(pairs=total_pairs)

    for fh in writers.values():
        fh.close()

    # Combine all into one file
    combined = QA_DIR / "all_languages_combined.jsonl"
    combined_total = 0
    with combined.open("w", encoding="utf-8") as out:
        for lang in SUPPORTED_LANGUAGES:
            lf = QA_DIR / f"{lang}_qa_pairs.jsonl"
            if lf.exists():
                with lf.open("r", encoding="utf-8") as inp:
                    for line in inp:
                        if line.strip():
                            out.write(line)
                            combined_total += 1

    print("\n" + "=" * 60)
    print("  QA GENERATION — SUMMARY")
    print("=" * 60)
    print(f"  Total QA pairs   : {total_pairs:>8,}")
    print(f"  Combined file    : {combined_total:>8,}")
    print("\n  Per-language:")
    for lang in SUPPORTED_LANGUAGES:
        print(f"    {lang:<8}  {lang_counts.get(lang, 0):>6,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
