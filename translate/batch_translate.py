"""
batch_translate.py — Stage 4 of the Kumbh Mela 2027 data pipeline.

Translates English chunks from knowledge_base/deduplicated/ into 7 Indic
languages using IndicTrans2 (ai4bharat/indictrans2-en-indic-1B).
Outputs go to knowledge_base/translated/{lang_code}/.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEDUP_DIR    = PROJECT_ROOT / "knowledge_base" / "deduplicated"
TRANS_DIR    = PROJECT_ROOT / "knowledge_base" / "translated"

BATCH_SIZE = 16
SOURCE_LANG = "eng_Latn"
MODEL_NAME  = "ai4bharat/indictrans2-en-indic-1B"

TARGET_LANGUAGES = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError as exc:
        log.error("Missing: %s — install transformers and torch", exc)
        sys.exit(1)

    log.info("Loading IndicTrans2: %s", MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    _model.to(device)
    _model.eval()
    log.info("Model loaded on: %s", device)
    return _tokenizer, _model


def translate_batch(sentences: list[str], tgt_lang: str) -> list[str]:
    import torch
    tokenizer, model = load_model()
    device = next(model.parameters()).device
    tokenizer.src_lang = SOURCE_LANG
    encoded = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=512,
            num_beams=4,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def translate_chunk_text(text: str, tgt_lang_code: str) -> str:
    sent_re = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sent_re.split(text) if s.strip()]
    if not sentences:
        return text

    translated: list[str] = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        try:
            results = translate_batch(batch, tgt_lang_code)
            translated.extend(results)
        except Exception as exc:
            log.warning("Translation batch failed (%s)", exc)
            translated.extend(batch)
    return " ".join(translated)


def main() -> None:
    if not DEDUP_DIR.exists():
        log.error("Deduplicated directory not found: %s", DEDUP_DIR)
        sys.exit(1)

    for lang in TARGET_LANGUAGES:
        (TRANS_DIR / lang).mkdir(parents=True, exist_ok=True)

    en_files = []
    for path in sorted(DEDUP_DIR.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                data = [data]
            en_chunks = [c for c in data if (c.get("language") or "").startswith("en")]
            if en_chunks:
                en_files.append((path, en_chunks))
        except Exception:
            continue

    if not en_files:
        log.warning("No English chunks found under %s", DEDUP_DIR)
        return

    log.info("Found %d file(s) with English chunks", len(en_files))
    stats: dict[str, int] = {lang: 0 for lang in TARGET_LANGUAGES}

    file_iter = tqdm(en_files, desc="Files", unit="file") if TQDM_AVAILABLE else en_files

    for src_path, en_chunks in file_iter:
        for lang, lang_code in TARGET_LANGUAGES.items():
            relative = src_path.relative_to(DEDUP_DIR)
            out_path = TRANS_DIR / lang / relative

            if out_path.exists() and out_path.stat().st_size > 0:
                log.debug("Already translated [%s] %s — skipping", lang, src_path.name)
                continue

            translated_chunks: list[dict[str, Any]] = []
            for chunk in en_chunks:
                original_text = chunk.get("text", "")
                try:
                    translated_text = translate_chunk_text(original_text, lang_code)
                except Exception as exc:
                    log.warning("Failed to translate chunk %s to %s: %s", chunk.get("id"), lang, exc)
                    translated_text = original_text

                translated_chunks.append({
                    **chunk,
                    "text":              translated_text,
                    "language":          lang,
                    "translated_from":   "en",
                    "translation_model": "indictrans2",
                    "original_text":     original_text,
                })
                stats[lang] += 1

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(translated_chunks, fh, ensure_ascii=False, indent=2)
            log.info("[%s] %s → %d chunk(s)", lang, src_path.name, len(translated_chunks))

    print("\n" + "=" * 60)
    print("  TRANSLATION — SUMMARY")
    print("=" * 60)
    total = 0
    for lang, count in stats.items():
        print(f"  {lang:<8} : {count:>6,} chunks translated")
        total += count
    print(f"  {'TOTAL':<8} : {total:>6,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
