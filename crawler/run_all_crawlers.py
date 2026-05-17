"""
run_all_crawlers.py — Master crawler runner for Kumbh Mela 2027.

Runs all 4 crawlers sequentially and produces a crawl_summary.json.

Crawlers:
  1. wikipedia_spider   — 128 Wikipedia articles (16 topics × 8 languages)
  2. osm_places         — OSM/Overpass API (temples, ghats, hospitals, etc.)
  3. news_crawler        — Lokmat, Dainik Bhaskar, Maharashtra Times, etc.
  4. indic_datasets      — IndicQA, Samanantar, AI4Bharat HuggingFace datasets

Usage:
  python crawler/run_all_crawlers.py
  python crawler/run_all_crawlers.py --crawlers wikipedia osm
  python crawler/run_all_crawlers.py --skip indic_datasets --force
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CRAWLER_DIR  = PROJECT_ROOT / "crawler" / "spiders"
OUTPUT_DIR   = PROJECT_ROOT / "knowledge_base" / "raw"

LOG_FILE = PROJECT_ROOT / "crawler" / "crawler_errors.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger("run_all_crawlers")

CRAWLERS: list[dict[str, Any]] = [
    {
        "num": 1,
        "key": "wikipedia",
        "name": "Wikipedia Spider",
        "module": "wikipedia_spider",
        "run_fn": "crawl",
        "run_kwargs": {},
        "output_dir": OUTPUT_DIR / "wikipedia",
        "description": "128 Wikipedia articles in 8 languages",
    },
    {
        "num": 2,
        "key": "osm",
        "name": "OSM / Overpass Spider",
        "module": "osm_places",
        "run_fn": "run",
        "run_kwargs": {},
        "output_dir": OUTPUT_DIR / "osm",
        "description": "OpenStreetMap places (temples, ghats, hospitals, transport …)",
    },
    {
        "num": 3,
        "key": "news",
        "name": "News Crawler",
        "module": "news_crawler",
        "run_fn": "run",
        "run_kwargs": {"max_articles_per_source": 30, "delay": 2.0},
        "output_dir": OUTPUT_DIR / "news",
        "description": "Lokmat, Dainik Bhaskar, Maharashtra Times, Navbharat Times, The Hindu, TOI",
    },
    {
        "num": 4,
        "key": "indic_datasets",
        "name": "Indic HuggingFace Datasets",
        "module": "indic_datasets",
        "run_fn": "run",
        "run_kwargs": {"max_samples": 5000},
        "output_dir": OUTPUT_DIR / "indicqa",
        "description": "IndicQA, Samanantar, AI4Bharat news (HuggingFace)",
    },
]


def load_module(module_name: str):
    """Dynamically import a spider module from the spiders/ directory."""
    module_path = CRAWLER_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Spider not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def output_exists(crawler: dict[str, Any]) -> bool:
    d: Path = crawler["output_dir"]
    if not d.exists():
        return False
    return any(d.rglob("*.json")) or any(d.rglob("*.jsonl"))


def run_crawler(crawler: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num": crawler["num"],
        "key": crawler["key"],
        "name": crawler["name"],
        "ran": True,
        "success": False,
        "elapsed": 0.0,
        "error": None,
        "records": None,
    }

    start = time.time()
    try:
        mod = load_module(crawler["module"])
        fn = getattr(mod, crawler["run_fn"])
        ret = fn(**crawler["run_kwargs"])
        result["success"] = True
        result["records"] = ret
        log.info("✔  [%s] done in %.1fs | result: %s", crawler["key"], time.time() - start, ret)
    except Exception:
        tb = traceback.format_exc()
        log.error("✘  [%s] FAILED:\n%s", crawler["key"], tb)
        result["error"] = tb
    finally:
        result["elapsed"] = round(time.time() - start, 2)

    return result


def count_raw_files() -> dict[str, int]:
    counts: dict[str, int] = {}
    for subdir in sorted(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else []:
        if subdir.is_dir():
            n = len(list(subdir.rglob("*.json"))) + len(list(subdir.rglob("*.jsonl")))
            counts[subdir.name] = n
    return counts


def print_summary(results: list[dict], total_elapsed: float) -> None:
    print("\n" + "=" * 70)
    print("  KUMBH MELA 2027 — CRAWL SUMMARY")
    print(f"  Finished : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    print("\n  Crawler results:")
    for r in results:
        if not r["ran"]:
            status = "SKIP"
            icon = "—"
        elif r["success"]:
            status = "PASS"
            icon = "✔"
        else:
            status = "FAIL"
            icon = "✘"
        print(f"    {icon}  {r['num']}. {r['name']:<35} {status}  ({r['elapsed']:.1f}s)")

    print("\n  Files in knowledge_base/raw/:")
    file_counts = count_raw_files()
    grand_total = 0
    for subdir, count in file_counts.items():
        print(f"    {subdir:<25} {count:>5} files")
        grand_total += count
    print(f"    {'TOTAL':<25} {grand_total:>5} files")

    print(f"\n  Total crawl time : {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("=" * 70)
    print("\n  Next step: run the data pipeline")
    print("    python pipeline/run_pipeline.py\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kumbh Mela 2027 — Master Crawler")
    p.add_argument("--crawlers", nargs="+",
                   choices=[c["key"] for c in CRAWLERS],
                   help="Which crawlers to run (default: all)")
    p.add_argument("--skip", nargs="+",
                   choices=[c["key"] for c in CRAWLERS],
                   help="Crawlers to skip")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output already exists")
    return p.parse_args()


def main() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    only = set(args.crawlers) if args.crawlers else set()
    skip = set(args.skip) if args.skip else set()

    crawlers = [
        c for c in CRAWLERS
        if (not only or c["key"] in only) and c["key"] not in skip
    ]

    if not crawlers:
        log.error("No crawlers selected.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  KUMBH MELA 2027 — STARTING CRAWL")
    print(f"  Time    : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Crawlers: {', '.join(c['key'] for c in crawlers)}")
    print(f"  Force   : {args.force}")
    print("=" * 70 + "\n")

    results: list[dict] = []
    t0 = time.time()
    any_failed = False

    for crawler in crawlers:
        log.info("=" * 50)
        log.info("▶  Crawler %d/%d — %s", crawler["num"], len(crawlers), crawler["description"])
        log.info("=" * 50)

        sr: dict[str, Any] = {
            "num": crawler["num"], "key": crawler["key"],
            "name": crawler["name"], "ran": False,
            "success": True, "elapsed": 0.0, "error": None, "records": None,
        }

        if not args.force and output_exists(crawler):
            log.info("—  [%s] already crawled — skipping (use --force to re-run)", crawler["key"])
            results.append(sr)
            continue

        crawler["output_dir"].mkdir(parents=True, exist_ok=True)
        sr = run_crawler(crawler)
        results.append(sr)

        if not sr["success"]:
            any_failed = True
            log.warning("Crawler [%s] failed — continuing with remaining crawlers.", crawler["key"])

    total_elapsed = time.time() - t0

    # Save crawl summary
    summary_path = PROJECT_ROOT / "crawler" / "crawl_summary.json"
    summary_data = {
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "results": [
            {k: v for k, v in r.items() if k != "error"}
            for r in results
        ],
        "file_counts": count_raw_files(),
        "any_failed": any_failed,
    }
    summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Crawl summary saved → %s", summary_path)

    print_summary(results, total_elapsed)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
