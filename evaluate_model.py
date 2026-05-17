"""
Evaluate the custom Kumbh model in isolation.

Runs the model on a held-out test set and computes:
  - BLEU score (n-gram overlap)
  - ROUGE-L score (longest common subsequence)
  - Keyword recall (domain-specific term coverage)
  - Response quality (non-empty, sufficient length)
  - Per-language breakdown

Usage:
    python evaluate_model.py                          # evaluate with defaults
    python evaluate_model.py --samples 50             # test on 50 samples
    python evaluate_model.py --languages en hi mr     # specific languages only
    python evaluate_model.py --output eval_results.json  # save results
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "synthetic_qa"


# ── Metrics (no extra dependencies) ──────────────────────────────────────────

def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Simplified BLEU (no brevity penalty for simplicity)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(ngrams(ref_tokens, n))
        hyp_ng = Counter(ngrams(hyp_tokens, n))
        overlap = sum((hyp_ng & ref_ng).values())
        total = max(sum(hyp_ng.values()), 1)
        scores.append(overlap / total)

    # Geometric mean
    product = 1.0
    for s in scores:
        product *= max(s, 1e-10)
    return product ** (1.0 / len(scores))


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L using longest common subsequence."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    m, n = len(ref_tokens), len(hyp_tokens)
    # Optimized LCS length (O(m*n) but fine for short texts)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr

    lcs_len = prev[n]
    precision = lcs_len / n if n else 0
    recall = lcs_len / m if m else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_recall(reference: str, hypothesis: str) -> float:
    """Check how many important keywords from reference appear in hypothesis."""
    # Extract keywords (nouns, proper nouns, numbers — simple heuristic)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "of", "in", "to", "for",
        "with", "on", "at", "by", "from", "it", "its", "this", "that",
        "and", "or", "but", "not", "no", "if", "as", "so", "than", "then",
        "also", "very", "just", "about", "more", "most", "some", "any",
    }
    ref_words = set(reference.lower().split()) - stop_words
    hyp_words = set(hypothesis.lower().split()) - stop_words
    # Keep only meaningful words (3+ chars)
    ref_keywords = {w for w in ref_words if len(w) >= 3}
    if not ref_keywords:
        return 1.0
    matched = ref_keywords & hyp_words
    return len(matched) / len(ref_keywords)


# ── Load test data ───────────────────────────────────────────────────────────

def load_test_set(languages: list[str], samples_per_lang: int, seed: int = 42) -> list[dict]:
    """Load a stratified sample from QA pairs as a held-out test set."""
    random.seed(seed)
    test_samples = []

    for lang in languages:
        qa_file = DATA_DIR / f"{lang}_qa_pairs.jsonl"
        if not qa_file.exists():
            log.warning(f"  No QA file for language '{lang}', skipping")
            continue

        pairs = []
        with open(qa_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))

        # Take a random sample (from the end of the file = less likely to be in training)
        if len(pairs) > samples_per_lang:
            # Use last 30% as test pool (simulates held-out split)
            test_pool = pairs[int(len(pairs) * 0.7) :]
            sample = random.sample(test_pool, min(samples_per_lang, len(test_pool)))
        else:
            sample = pairs

        test_samples.extend(sample)
        log.info(f"  Loaded {len(sample)} test samples for '{lang}'")

    return test_samples


# ── Run evaluation ───────────────────────────────────────────────────────────

def evaluate(
    samples: list[dict],
    model_backend: str = "auto",
    show_examples: bool = True,
) -> dict:
    """Run the model on test samples and compute metrics."""

    # Import model
    sys.path.insert(0, str(ROOT))
    os.environ["ENHANCE_RESPONSES"] = "false"  # Disable enhancement for isolated eval
    from api.services.llm import LLMService
    from api.services.rag import RAGService

    llm = LLMService()
    rag = RAGService()

    log.info(f"\n  Model backend: {llm.backend} ({llm.model_name})")
    log.info(f"  Enhancement: DISABLED (isolated evaluation)")
    log.info(f"  Samples: {len(samples)}\n")

    results = {
        "model": llm.model_name,
        "backend": llm.backend,
        "total_samples": len(samples),
        "per_language": defaultdict(lambda: {
            "count": 0, "bleu": [], "rouge_l": [], "keyword_recall": [],
            "non_empty": 0, "avg_length": [],
        }),
        "examples": [],
    }

    for i, sample in enumerate(samples):
        query = sample["instruction"]
        reference = sample["output"]
        language = sample.get("language", "en")
        context_input = sample.get("input", "")

        # Get RAG context (same as production pipeline)
        docs = rag.retrieve(query, language=language, top_k=3)
        context_texts = [d["text"] for d in docs]
        if context_input:
            context_texts = [context_input] + context_texts

        # Generate
        start = time.time()
        try:
            hypothesis = llm.generate(
                query=query,
                context=context_texts,
                language=language,
                max_tokens=200,
            )
        except Exception as e:
            hypothesis = ""
            log.warning(f"  Error on sample {i}: {e}")
        elapsed = time.time() - start

        # Compute metrics
        b = bleu_score(reference, hypothesis)
        r = rouge_l(reference, hypothesis)
        k = keyword_recall(reference, hypothesis)
        is_non_empty = len(hypothesis.strip()) > 10

        lang_results = results["per_language"][language]
        lang_results["count"] += 1
        lang_results["bleu"].append(b)
        lang_results["rouge_l"].append(r)
        lang_results["keyword_recall"].append(k)
        lang_results["non_empty"] += int(is_non_empty)
        lang_results["avg_length"].append(len(hypothesis.split()))

        # Store example
        if len(results["examples"]) < 10:
            results["examples"].append({
                "query": query,
                "reference": reference[:200],
                "hypothesis": hypothesis[:200],
                "language": language,
                "bleu": round(b, 4),
                "rouge_l": round(r, 4),
                "time_s": round(elapsed, 2),
            })

        progress = f"[{i+1}/{len(samples)}]"
        log.info(f"  {progress} ({language}) BLEU={b:.3f} ROUGE-L={r:.3f} KW={k:.3f} | {elapsed:.1f}s")

    # Aggregate
    all_bleu, all_rouge, all_kw = [], [], []
    summary_by_lang = {}

    for lang, lr in results["per_language"].items():
        avg_bleu = sum(lr["bleu"]) / max(len(lr["bleu"]), 1)
        avg_rouge = sum(lr["rouge_l"]) / max(len(lr["rouge_l"]), 1)
        avg_kw = sum(lr["keyword_recall"]) / max(len(lr["keyword_recall"]), 1)
        avg_len = sum(lr["avg_length"]) / max(len(lr["avg_length"]), 1)
        response_rate = lr["non_empty"] / max(lr["count"], 1)

        all_bleu.extend(lr["bleu"])
        all_rouge.extend(lr["rouge_l"])
        all_kw.extend(lr["keyword_recall"])

        summary_by_lang[lang] = {
            "samples": lr["count"],
            "avg_bleu": round(avg_bleu, 4),
            "avg_rouge_l": round(avg_rouge, 4),
            "avg_keyword_recall": round(avg_kw, 4),
            "response_rate": round(response_rate, 4),
            "avg_response_words": round(avg_len, 1),
        }

    results["overall"] = {
        "avg_bleu": round(sum(all_bleu) / max(len(all_bleu), 1), 4),
        "avg_rouge_l": round(sum(all_rouge) / max(len(all_rouge), 1), 4),
        "avg_keyword_recall": round(sum(all_kw) / max(len(all_kw), 1), 4),
        "total_evaluated": len(samples),
    }
    results["per_language"] = summary_by_lang

    return results


def print_report(results: dict):
    """Pretty-print evaluation results."""
    log.info("\n" + "=" * 65)
    log.info("  KUMBH MODEL EVALUATION REPORT")
    log.info("=" * 65)
    log.info(f"  Model    : {results['model']}")
    log.info(f"  Backend  : {results['backend']}")
    log.info(f"  Samples  : {results['total_samples']}")
    log.info("-" * 65)

    log.info("\n  OVERALL METRICS:")
    o = results["overall"]
    log.info(f"    BLEU Score       : {o['avg_bleu']:.4f}")
    log.info(f"    ROUGE-L Score    : {o['avg_rouge_l']:.4f}")
    log.info(f"    Keyword Recall   : {o['avg_keyword_recall']:.4f}")

    log.info("\n  PER-LANGUAGE BREAKDOWN:")
    log.info(f"    {'Lang':<6} {'N':>4} {'BLEU':>8} {'ROUGE-L':>8} {'KW-Recall':>10} {'Resp%':>7} {'AvgWords':>9}")
    log.info(f"    {'-'*6} {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*9}")

    for lang, m in sorted(results["per_language"].items()):
        log.info(
            f"    {lang:<6} {m['samples']:>4} {m['avg_bleu']:>8.4f} {m['avg_rouge_l']:>8.4f} "
            f"{m['avg_keyword_recall']:>10.4f} {m['response_rate']:>6.1%} {m['avg_response_words']:>9.1f}"
        )

    if results.get("examples"):
        log.info("\n  SAMPLE PREDICTIONS:")
        log.info("-" * 65)
        for ex in results["examples"][:5]:
            log.info(f"    Q ({ex['language']}): {ex['query'][:80]}")
            log.info(f"    Expected : {ex['reference'][:80]}...")
            log.info(f"    Got      : {ex['hypothesis'][:80]}...")
            log.info(f"    BLEU={ex['bleu']:.4f}  ROUGE-L={ex['rouge_l']:.4f}  Time={ex['time_s']}s")
            log.info()

    log.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Kumbh model accuracy")
    parser.add_argument("--samples", type=int, default=20, help="Samples per language")
    parser.add_argument("--languages", nargs="+", default=["en", "hi", "mr"],
                        help="Languages to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    log.info("Loading test set...")
    test_set = load_test_set(args.languages, args.samples, args.seed)

    if not test_set:
        log.error("No test samples found!")
        sys.exit(1)

    log.info(f"\nRunning evaluation on {len(test_set)} samples...")
    results = evaluate(test_set)

    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"\nResults saved to {args.output}")
