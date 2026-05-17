"""
paraphrase.py — Stage 6 of the Kumbh Mela 2027 data pipeline.

Reads QA pairs, generates 2-3 paraphrased question variants per pair
(formal, informal, voice-style, dialect), saves to data/synthetic_qa/augmented/.
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

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
QA_DIR        = PROJECT_ROOT / "data" / "synthetic_qa"
AUGMENTED_DIR = QA_DIR / "augmented"

OLLAMA_BASE  = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

SUPPORTED_LANGUAGES = ["en", "hi", "mr", "gu", "ta", "te", "kn", "ml"]

LANG_NAMES = {
    "en": "English", "hi": "Hindi (हिंदी)", "mr": "Marathi (मराठी)",
    "gu": "Gujarati (ગુજરાતી)", "ta": "Tamil (தமிழ்)", "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)", "ml": "Malayalam (മലയാളം)",
}

DIALECT_LABELS: dict[str, list[str]] = {
    "hi": ["formal_hi", "informal_hi", "colloquial_hi"],
    "mr": ["standard_mr", "nashik_dialect_mr", "informal_mr"],
    "en": ["formal_en", "casual_en", "voice_en"],
    "gu": ["standard_gu", "informal_gu", "voice_gu"],
    "ta": ["formal_ta", "colloquial_ta", "voice_ta"],
    "te": ["formal_te", "informal_te", "voice_te"],
    "kn": ["formal_kn", "colloquial_kn", "voice_kn"],
    "ml": ["formal_ml", "informal_ml", "voice_ml"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def build_paraphrase_prompt(question: str, language: str, num_variants: int) -> str:
    lang_name = LANG_NAMES.get(language, "the same language")
    dialect_hint = (
        "Include: (1) formal written, (2) informal/casual, (3) voice-style (shorter, spoken)."
        if language == "en" else
        f"Include: (1) formal {lang_name}, (2) informal {lang_name}, (3) natural spoken voice query."
    )
    if language == "mr":
        dialect_hint = "Include: (1) standard Marathi, (2) Nashik-dialect Marathi, (3) informal Marathi."

    return f"""Paraphrase the following question {num_variants} ways in {lang_name}.

Original: "{question}"

{dialect_hint}

Rules:
1. All paraphrases must be in {lang_name} ONLY.
2. Same meaning, different phrasing/structure.
3. Format: JSON array [{{"paraphrase": "...", "style": "formal|informal|voice|dialect"}}]
4. Output ONLY the JSON array.

JSON:"""


def ollama_generate(prompt: str, retries: int = 3) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.8, "top_p": 0.9, "num_predict": 1024},
    }).encode()

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode()).get("response", "")
        except Exception as exc:
            log.warning("Ollama attempt %d/%d: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return ""


def parse_paraphrases(response: str) -> list[dict[str, str]]:
    for text in [response.strip(), (re.search(r"\[.*?\]", response, re.DOTALL) or type("", (), {"group": lambda s: ""})()).group()]:
        if not text:
            continue
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [{"paraphrase": str(item.get("paraphrase", "")), "style": str(item.get("style", "variant"))}
                        for item in data if isinstance(item, dict) and item.get("paraphrase")]
        except json.JSONDecodeError:
            continue
    # Fallback: extract quoted strings
    quoted = re.findall(r'"([^"]{10,})"', response)
    return [{"paraphrase": q, "style": "variant"} for q in quoted[:3]]


def augment_pair(pair: dict[str, Any]) -> list[dict[str, Any]]:
    question = pair.get("instruction", "")
    language = pair.get("language", "en")
    answer   = pair.get("output", "")

    num_variants = random.randint(2, 3)
    response = ollama_generate(build_paraphrase_prompt(question, language, num_variants))
    if not response:
        return []

    variants = parse_paraphrases(response)
    dialect_labels = DIALECT_LABELS.get(language, ["variant"])
    augmented: list[dict[str, Any]] = []

    for i, variant in enumerate(variants[:num_variants]):
        paraphrase = variant.get("paraphrase", "").strip()
        if not paraphrase:
            continue
        augmented.append({
            "instruction": paraphrase,
            "input": "",
            "output": answer,
            "language": language,
            "domain": pair.get("domain", "general"),
            "type": pair.get("type", "factual"),
            "source_url": pair.get("source_url", ""),
            "chunk_id": pair.get("chunk_id", ""),
            "augmented": True,
            "original_question": question,
            "paraphrase_style": variant.get("style", "variant"),
            "dialect": dialect_labels[i % len(dialect_labels)],
        })
    return augmented


def iter_qa_pairs() -> Iterator[tuple[str, dict[str, Any]]]:
    for lang in SUPPORTED_LANGUAGES:
        lang_file = QA_DIR / f"{lang}_qa_pairs.jsonl"
        if not lang_file.exists():
            continue
        with lang_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield lang, json.loads(line)
                    except json.JSONDecodeError:
                        continue


def main() -> None:
    if not QA_DIR.exists():
        log.error("QA directory not found: %s — run qa_generator.py first", QA_DIR)
        sys.exit(1)

    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    aug_writers = {lang: (AUGMENTED_DIR / f"{lang}_augmented.jsonl").open("a", encoding="utf-8")
                   for lang in SUPPORTED_LANGUAGES}
    combined_fh = (AUGMENTED_DIR / "all_augmented_combined.jsonl").open("w", encoding="utf-8")

    lang_counts: dict[str, int] = {lang: 0 for lang in SUPPORTED_LANGUAGES}
    total_original = 0
    total_augmented = 0

    all_pairs = list(iter_qa_pairs())
    if not all_pairs:
        log.error("No QA pairs found. Run qa_generator.py first.")
        for fh in aug_writers.values():
            fh.close()
        combined_fh.close()
        sys.exit(1)

    log.info("Loaded %d QA pairs — starting augmentation…", len(all_pairs))
    pair_iter = tqdm(all_pairs, desc="Paraphrasing", unit="pair") if TQDM_AVAILABLE else all_pairs

    for lang, pair in pair_iter:
        total_original += 1
        combined_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

        for aug in augment_pair(pair):
            aug_lang = aug.get("language", lang)
            fh = aug_writers.get(aug_lang)
            if fh:
                fh.write(json.dumps(aug, ensure_ascii=False) + "\n")
            combined_fh.write(json.dumps(aug, ensure_ascii=False) + "\n")
            lang_counts[aug_lang] = lang_counts.get(aug_lang, 0) + 1
            total_augmented += 1

    for fh in aug_writers.values():
        fh.close()
    combined_fh.close()

    print("\n" + "=" * 60)
    print("  PARAPHRASE AUGMENTATION — SUMMARY")
    print("=" * 60)
    print(f"  Original pairs   : {total_original:>8,}")
    print(f"  Augmented pairs  : {total_augmented:>8,}")
    print(f"  Total combined   : {total_original + total_augmented:>8,}")
    print("\n  Augmented per language:")
    for lang in SUPPORTED_LANGUAGES:
        print(f"    {lang:<8}  {lang_counts.get(lang, 0):>6,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
