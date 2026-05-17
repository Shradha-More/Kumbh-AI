"""
run_pipeline.py — Master runner for the Kumbh Mela 2027 data pipeline.

Stages:
  1. clean        → knowledge_base/cleaned/
  2. chunk        → knowledge_base/chunked/
  3. deduplicate  → knowledge_base/deduplicated/
  4. translate    → knowledge_base/translated/
  5. qa_generate  → data/synthetic_qa/
  6. paraphrase   → data/synthetic_qa/augmented/

Usage:
  python pipeline/run_pipeline.py [--force] [--steps 1,2,3] [--skip 4]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_LANGUAGES = ["en", "hi", "mr", "gu", "ta", "te", "kn", "ml"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

STAGES: list[dict[str, Any]] = [
    {
        "num": 1, "name": "clean", "description": "Clean raw documents",
        "script": PROJECT_ROOT / "pipeline" / "clean.py",
        "output_dir": PROJECT_ROOT / "knowledge_base" / "cleaned",
        "check_file": None,
    },
    {
        "num": 2, "name": "chunk", "description": "Chunk cleaned documents",
        "script": PROJECT_ROOT / "pipeline" / "chunk.py",
        "output_dir": PROJECT_ROOT / "knowledge_base" / "chunked",
        "check_file": PROJECT_ROOT / "knowledge_base" / "chunked" / "chunked_manifest.json",
    },
    {
        "num": 3, "name": "deduplicate", "description": "Remove duplicate/near-duplicate chunks",
        "script": PROJECT_ROOT / "pipeline" / "deduplicate.py",
        "output_dir": PROJECT_ROOT / "knowledge_base" / "deduplicated",
        "check_file": None,
    },
    {
        "num": 4, "name": "translate", "description": "Translate English chunks to 7 Indic languages",
        "script": PROJECT_ROOT / "translate" / "batch_translate.py",
        "output_dir": PROJECT_ROOT / "knowledge_base" / "translated",
        "check_file": None,
    },
    {
        "num": 5, "name": "qa_generate", "description": "Generate synthetic QA pairs",
        "script": PROJECT_ROOT / "generate" / "qa_generator.py",
        "output_dir": PROJECT_ROOT / "data" / "synthetic_qa",
        "check_file": PROJECT_ROOT / "data" / "synthetic_qa" / "all_languages_combined.jsonl",
    },
    {
        "num": 6, "name": "paraphrase", "description": "Augment QA pairs via paraphrasing",
        "script": PROJECT_ROOT / "generate" / "paraphrase.py",
        "output_dir": PROJECT_ROOT / "data" / "synthetic_qa" / "augmented",
        "check_file": PROJECT_ROOT / "data" / "synthetic_qa" / "augmented" / "all_augmented_combined.jsonl",
    },
]


def output_exists(stage: dict[str, Any]) -> bool:
    check: Path | None = stage.get("check_file")
    if check is not None:
        return check.exists() and check.stat().st_size > 0
    output_dir: Path = stage["output_dir"]
    if not output_dir.exists():
        return False
    return any(output_dir.rglob("*.json")) or any(output_dir.rglob("*.jsonl"))


def run_stage(stage: dict[str, Any]) -> tuple[bool, float]:
    script: Path = stage["script"]
    if not script.exists():
        log.error("Script not found: %s", script)
        return False, 0.0

    log.info("▶  Stage %d/%d — %s", stage["num"], len(STAGES), stage["description"])
    start = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        log.error("Stage %d (%s) FAILED (exit %d, %.1fs)", stage["num"], stage["name"], result.returncode, elapsed)
        return False, elapsed

    log.info("✔  Stage %d (%s) done in %.1fs", stage["num"], stage["name"], elapsed)
    return True, elapsed


def count_json_items(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*.json"):
        if "manifest" in path.name:
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            total += len(data) if isinstance(data, list) else 1
        except Exception:
            pass
    for path in directory.rglob("*.jsonl"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                total += sum(1 for line in fh if line.strip())
        except Exception:
            pass
    return total


def estimate_training_time(total_pairs: int) -> dict[str, str]:
    steps = (total_pairs // 8) * 3  # 3 epochs, batch 8
    secs = steps * 0.15

    def fmt(s: float) -> str:
        return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"

    return {
        "A100 80GB":  fmt(secs),
        "RTX 4090":   fmt(secs * 3),
        "RTX 3090":   fmt(secs * 4),
        "total_steps": str(steps),
    }


def print_summary(stage_results: list[dict[str, Any]], total_elapsed: float) -> None:
    print("\n" + "=" * 70)
    print("  KUMBH MELA 2027 — PIPELINE SUMMARY")
    print("=" * 70)
    print("\n  Stage results:")
    for sr in stage_results:
        status = "PASS" if sr["success"] else "FAIL" if sr["ran"] else "SKIP"
        emoji = {"PASS": "✔", "FAIL": "✘", "SKIP": "—"}[status]
        print(f"    {emoji} Stage {sr['num']} — {sr['name']:<14} {status}  ({sr['elapsed']:.1f}s)")

    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    dirs = {
        "Cleaned docs":    PROJECT_ROOT / "knowledge_base" / "cleaned",
        "Chunks":          PROJECT_ROOT / "knowledge_base" / "chunked",
        "After dedup":     PROJECT_ROOT / "knowledge_base" / "deduplicated",
        "QA pairs":        PROJECT_ROOT / "data" / "synthetic_qa",
        "Augmented pairs": PROJECT_ROOT / "data" / "synthetic_qa" / "augmented",
    }

    print("\n  Dataset statistics:")
    qa_total = 0
    for label, d in dirs.items():
        count = count_json_items(d) if d.exists() else 0
        print(f"    {label:<18} : {count:>8,}")
        if label == "QA pairs":
            qa_total = count

    if qa_total > 0:
        est = estimate_training_time(qa_total)
        print("\n  Estimated fine-tuning time (3 epochs, batch=8, 7B model):")
        for hw, dur in est.items():
            if hw != "total_steps":
                print(f"    {hw:<20} {dur}")
        print(f"    Steps: {est['total_steps']}")

    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kumbh Mela 2027 — Data pipeline runner")
    p.add_argument("--force", action="store_true", help="Re-run all stages even if outputs exist")
    p.add_argument("--steps", default="", help="Comma-separated stage numbers to run (default: all)")
    p.add_argument("--skip",  default="", help="Comma-separated stage numbers to skip")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    only = {int(x) for x in args.steps.split(",") if x.strip().isdigit()}
    skip = {int(x) for x in args.skip.split(",")  if x.strip().isdigit()}

    stages = [s for s in STAGES if (not only or s["num"] in only) and s["num"] not in skip]
    if not stages:
        log.error("No stages selected.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  KUMBH MELA 2027 — STARTING PIPELINE")
    print(f"  Time  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Stages: {', '.join(str(s['num']) for s in stages)}")
    print(f"  Force : {args.force}")
    print("=" * 70 + "\n")

    stage_results: list[dict[str, Any]] = []
    t0 = time.time()
    any_failed = False

    for stage in stages:
        sr = {"num": stage["num"], "name": stage["name"], "ran": False, "success": True, "elapsed": 0.0}

        if not args.force and output_exists(stage):
            log.info("—  Stage %d (%s) already done — skipping (use --force to re-run)", stage["num"], stage["name"])
            stage_results.append(sr)
            continue

        stage["output_dir"].mkdir(parents=True, exist_ok=True)
        sr["ran"] = True
        success, elapsed = run_stage(stage)
        sr["success"] = success
        sr["elapsed"] = elapsed

        if not success:
            any_failed = True
            log.error("Pipeline halted at stage %d.", stage["num"])
            stage_results.append(sr)
            break

        stage_results.append(sr)

    print_summary(stage_results, time.time() - t0)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
