"""
Indic Datasets Spider for Nashik Kumbh Mela 2027.
Downloads IndicQA, Samanantar, and AI4Bharat news datasets from HuggingFace,
filters for Kumbh/pilgrimage relevance, and saves to knowledge_base/raw/.

Dependencies:
    pip install datasets huggingface_hub

Usage:
    python indic_datasets.py
    python indic_datasets.py --datasets indicqa samanantar --max-samples 5000
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

try:
    from datasets import load_dataset, DownloadConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "raw"

KUMBH_KEYWORDS = [
    # English
    "kumbh", "kumbha", "nashik", "simhastha", "godavari", "trimbakeshwar",
    "pilgrimage", "akhara", "mela", "ghat", "peshwai", "shahi snan",
    "ramkund", "panchvati", "panchavati", "kalaram", "saptashrungi",
    "hindu festival", "religious gathering", "tirtha", "mandir", "temple",
    # Hindi
    "कुंभ", "नाशिक", "सिंहस्थ", "गोदावरी", "त्र्यंबकेश्वर", "अखाड़ा",
    "शाही स्नान", "रामकुंड", "पंचवटी", "तीर्थ", "मेला", "घाट",
    # Marathi
    "कुंभमेळा", "आखाडा", "त्र्यंबकेश्वर", "गोदावरी", "नाशिक",
    # Gujarati
    "કુંભ", "નાસિક", "ગોદાવરી",
    # Tamil
    "கும்பமேளா", "நாசிக்",
    # Telugu
    "కుంభమేళా", "నాసిక్",
]

PILGRIMAGE_KEYWORDS = [
    "pilgrim", "yatra", "devotee", "festival", "religious", "sacred",
    "holy", "worship", "ritual", "bathing", "dip", "snaan", "tirtha",
    "dharma", "spiritual", "saint", "sadhu", "akhara",
]

SUPPORTED_LANGUAGES = ["en", "hi", "mr", "gu", "ta", "te", "kn", "ml"]

LANG_TO_FLORES = {
    "en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva",
    "gu": "guj_Gujr", "ta": "tam_Taml", "te": "tel_Telu",
    "kn": "kan_Knda", "ml": "mal_Mlym",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("indic_datasets")


def is_relevant(text: str) -> bool:
    tl = text.lower()
    return any(kw.lower() in tl for kw in KUMBH_KEYWORDS + PILGRIMAGE_KEYWORDS)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"<[^>]+>", "", text)
    return text


def iter_indicqa(max_samples: int) -> Iterator[dict[str, Any]]:
    """Download AI4Bharat IndicQA dataset — QA pairs in Hindi and Marathi."""
    if not HF_AVAILABLE:
        log.error("datasets library not installed. Run: pip install datasets")
        return

    configs = [
        ("ai4bharat/IndicQA", "indicqa.hi", "hi"),
        ("ai4bharat/IndicQA", "indicqa.mr", "mr"),
    ]

    for dataset_name, config, lang in configs:
        log.info("Loading IndicQA config=%s ...", config)
        try:
            ds = load_dataset(
                dataset_name, config,
                split="test",
                trust_remote_code=True,
                download_config=DownloadConfig(max_retries=3),
            )
        except Exception as exc:
            log.error("Failed to load %s/%s: %s", dataset_name, config, exc)
            continue

        count = 0
        for item in ds:
            if count >= max_samples:
                break
            context = clean_text(item.get("context", ""))
            question = clean_text(item.get("question", ""))
            answers = item.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []

            if not context or not question:
                continue
            if not is_relevant(context + " " + question):
                continue

            yield {
                "source": "IndicQA",
                "language": lang,
                "context": context,
                "question": question,
                "answer": answer_texts[0] if answer_texts else "",
                "dataset_config": config,
            }
            count += 1

        log.info("IndicQA [%s]: %d relevant items", lang, count)


def iter_samanantar(max_samples: int) -> Iterator[dict[str, Any]]:
    """Download Samanantar parallel corpus — EN-Indic sentence pairs."""
    if not HF_AVAILABLE:
        log.error("datasets library not installed. Run: pip install datasets")
        return

    lang_pairs = [
        ("ai4bharat/samanantar", "en-hi", "hi"),
        ("ai4bharat/samanantar", "en-mr", "mr"),
        ("ai4bharat/samanantar", "en-gu", "gu"),
    ]

    for dataset_name, config, tgt_lang in lang_pairs:
        log.info("Loading Samanantar config=%s ...", config)
        try:
            ds = load_dataset(
                dataset_name, config,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            log.error("Failed to load %s/%s: %s", dataset_name, config, exc)
            continue

        count = 0
        for item in ds:
            if count >= max_samples:
                break
            src = clean_text(item.get("src", ""))
            tgt = clean_text(item.get("tgt", ""))

            if not src or not tgt:
                continue
            if not is_relevant(src + " " + tgt):
                continue

            yield {
                "source": "Samanantar",
                "language_pair": config,
                "src_language": "en",
                "tgt_language": tgt_lang,
                "src_text": src,
                "tgt_text": tgt,
            }
            count += 1

        log.info("Samanantar [%s]: %d relevant pairs", config, count)


def iter_ai4bharat_news(max_samples: int) -> Iterator[dict[str, Any]]:
    """Download AI4Bharat news articles in Indic languages."""
    if not HF_AVAILABLE:
        log.error("datasets library not installed. Run: pip install datasets")
        return

    lang_configs = [
        ("ai4bharat/IndicNLPSuite", "hindi-news", "hi"),
        ("ai4bharat/IndicNLPSuite", "marathi-news", "mr"),
    ]

    for dataset_name, config, lang in lang_configs:
        log.info("Loading AI4Bharat news config=%s ...", config)
        try:
            ds = load_dataset(
                dataset_name, config,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            log.warning("Failed to load %s/%s (may not exist): %s", dataset_name, config, exc)
            continue

        count = 0
        for item in ds:
            if count >= max_samples:
                break
            text = clean_text(item.get("text", item.get("content", item.get("article", ""))))
            title = clean_text(item.get("title", item.get("headline", "")))

            if not text or len(text) < 50:
                continue
            if not is_relevant(text + " " + title):
                continue

            yield {
                "source": "AI4Bharat-News",
                "language": lang,
                "title": title,
                "content": text,
                "dataset_config": config,
            }
            count += 1

        log.info("AI4Bharat News [%s]: %d relevant articles", lang, count)


def iter_cc100_indic(max_samples: int) -> Iterator[dict[str, Any]]:
    """Download CC-100 monolingual Indic text as fallback corpus."""
    if not HF_AVAILABLE:
        return

    lang_codes = ["hi", "mr"]  # CC-100 uses ISO codes

    for lang in lang_codes:
        log.info("Loading CC-100 [%s] ...", lang)
        try:
            ds = load_dataset(
                "cc100", lang,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            log.warning("Failed to load CC-100 [%s]: %s", lang, exc)
            continue

        count = 0
        for item in ds:
            if count >= max_samples:
                break
            text = clean_text(item.get("text", ""))
            if not text or len(text) < 50:
                continue
            if not is_relevant(text):
                continue

            yield {
                "source": "CC-100",
                "language": lang,
                "content": text,
            }
            count += 1

        log.info("CC-100 [%s]: %d relevant items", lang, count)


DATASET_RUNNERS = {
    "indicqa":      iter_indicqa,
    "samanantar":   iter_samanantar,
    "ai4bharat_news": iter_ai4bharat_news,
    "cc100":        iter_cc100_indic,
}


def run(target_datasets=None, max_samples=5000):
    if not HF_AVAILABLE:
        log.error("Install datasets: pip install datasets huggingface_hub")
        return {}

    datasets_to_run = target_datasets or list(DATASET_RUNNERS.keys())
    summary: dict[str, int] = {}

    for ds_name in datasets_to_run:
        if ds_name not in DATASET_RUNNERS:
            log.warning("Unknown dataset: %s", ds_name)
            continue

        out_dir = OUTPUT_DIR / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []
        try:
            for record in DATASET_RUNNERS[ds_name](max_samples):
                records.append(record)
        except Exception as exc:
            log.error("Error running %s: %s", ds_name, exc)

        if not records:
            log.warning("[%s] No relevant records found.", ds_name)
            summary[ds_name] = 0
            continue

        # Group by language/config
        groups: dict[str, list] = {}
        for r in records:
            key = r.get("language", r.get("language_pair", "unknown"))
            groups.setdefault(key, []).append(r)

        for key, items in groups.items():
            out_path = out_dir / f"{ds_name}_{key}.json"
            out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("[%s/%s] Saved %d records → %s", ds_name, key, len(items), out_path.name)

        summary[ds_name] = len(records)
        log.info("[%s] Total: %d records", ds_name, len(records))

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=list(DATASET_RUNNERS.keys()))
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Max relevant items to collect per language config")
    args = p.parse_args()
    summary = run(target_datasets=args.datasets, max_samples=args.max_samples)

    print("\n=== Indic Datasets Summary ===")
    for ds, count in summary.items():
        print(f"  {ds:<20} {count:>6,} records")
    print("=" * 35)


if __name__ == "__main__":
    main()
