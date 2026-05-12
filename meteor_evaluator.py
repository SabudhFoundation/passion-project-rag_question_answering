"""
meteor_evaluator.py
===================
Evaluates the RAG pipeline's generated answers using the **METEOR Score**
(Metric for Evaluation of Translation with Explicit ORdering).

METEOR goes beyond simple word overlap (BLEU/ROUGE) by aligning words using:
    - Exact match
    - Stemmed match    (e.g., "running" matches "run")
    - Synonym match    (via WordNet lexical database)

This produces a score between 0.0 and 1.0 that correlates more strongly
with human judgment than lexical n-gram metrics.

INPUT STRATEGY (two modes):
    1. [Default] Read from the most recent BERTScore CSV
       (data/evaluation/generation_eval_summary_*.csv).
       This avoids re-running the expensive RAG pipeline.

    2. [--csv PATH] Explicitly specify a BERTScore CSV file to load.

    3. [--synth]   Re-read directly from the synthetic JSON dataset and
       re-generate answers (only use this if no CSV exists).

USAGE:
    # Score using the latest BERTScore CSV (recommended)
    python meteor_evaluator.py

    # Use a specific CSV
    python meteor_evaluator.py --csv data/evaluation/generation_eval_summary_20260509_210627.csv

    # Limit to first N rows for quick testing
    python meteor_evaluator.py --num 20

OUTPUTS (saved to data/evaluation/):
    - meteor_eval_results_<timestamp>.json   -- Full per-sample scores
    - meteor_eval_summary_<timestamp>.csv    -- Flat table with METEOR + BERTScore columns
    - meteor_eval_report_<timestamp>.md      -- Presentable Markdown report
"""

import os
import sys
import csv
import json
import glob
import argparse
import time
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# ── Project path setup ────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EVAL_DIR = os.path.join(PROJECT_DIR, "data", "evaluation")
OUTPUT_DIR = EVAL_DIR


# ═════════════════════════════════════════════════════════════════════════════
# NLTK / METEOR Bootstrap
# ═════════════════════════════════════════════════════════════════════════════

def _bootstrap_nltk() -> None:
    """
    Ensure the NLTK resources required for METEOR scoring are downloaded.
    METEOR uses WordNet for synonym matching and the punkt tokeniser.
    """
    import nltk

    required = {
        "wordnet":          "corpora/wordnet",
        "punkt":            "tokenizers/punkt",
        "punkt_tab":        "tokenizers/punkt_tab",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "omw-1.4":          "corpora/omw-1.4",
    }

    for name, path in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            nltk.download(name, quiet=True)


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════════════════════

def _find_latest_bert_csv() -> str:
    """
    Return the path to the most recently created BERTScore summary CSV.
    """
    pattern = os.path.join(EVAL_DIR, "generation_eval_summary_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            "No generation_eval_summary_*.csv found in data/evaluation/.\n"
            "Run generation_evaluator.py first, or pass --csv <path>."
        )
    latest = files[-1]
    logger.info("Using BERTScore CSV: %s", os.path.basename(latest))
    return latest


def _load_from_csv(csv_path: str, num: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Read pre-computed generated answers from a BERTScore summary CSV.

    Expected columns (from generation_evaluator.py):
        qid, difficulty, cluster_id, question, reference_answer,
        generated_answer, bert_precision, bert_recall, bert_f1, latency_seconds

    Returns a list of dicts containing the relevant fields.
    """
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if num is not None and i >= num:
                break
            records.append({
                "qid":              row.get("qid", f"q{i}"),
                "question":         row.get("question", ""),
                "reference_answer": row.get("reference_answer", ""),
                "generated_answer": row.get("generated_answer", ""),
                "difficulty":       row.get("difficulty", "unknown").lower(),
                "cluster_id":       row.get("cluster_id", ""),
                # Carry-over BERTScore metrics for the joint report
                "bert_precision":   float(row.get("bert_precision", 0) or 0),
                "bert_recall":      float(row.get("bert_recall", 0) or 0),
                "bert_f1":          float(row.get("bert_f1", 0) or 0),
                "latency_seconds":  float(row.get("latency_seconds", 0) or 0),
            })
    logger.info("Loaded %d records from CSV.", len(records))
    return records


# ═════════════════════════════════════════════════════════════════════════════
# METEOR Scoring
# ═════════════════════════════════════════════════════════════════════════════

def compute_meteor_scores(records: List[Dict[str, Any]]) -> List[float]:
    """
    Compute METEOR scores for every (generated, reference) pair.

    METEOR Algorithm Steps (as implemented by nltk.translate.meteor_score):
        1. Tokenise both the hypothesis (generated) and reference strings.
        2. Build an alignment using: exact match > stem match > synonym match.
        3. Compute Precision_m = |aligned| / |hypothesis_tokens|
                    Recall_m   = |aligned| / |reference_tokens|
        4. Compute F_mean using a weighted harmonic mean (default alpha=0.9).
        5. Apply a fragmentation penalty: Pen = gamma * (chunks/matches)^beta
           to penalise out-of-order alignments.
        6. METEOR = F_mean * (1 - Pen)

    Parameters used in nltk.translate.meteor_score:
        alpha  (default 0.9) -- Weight for Recall over Precision in F_mean.
        beta   (default 3.0) -- Exponent controlling fragmentation penalty steepness.
        gamma  (default 0.5) -- Coefficient scaling the fragmentation penalty.

    Returns:
        List of per-sample METEOR scores (floats between 0.0 and 1.0).
    """
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize

    scores = []
    total = len(records)

    for i, rec in enumerate(records, 1):
        ref_text = str(rec.get("reference_answer", "") or "")
        hyp_text = str(rec.get("generated_answer", "") or "")

        # NLTK meteor_score expects tokenised (list of strings) inputs
        ref_tokens = word_tokenize(ref_text.lower())
        hyp_tokens = word_tokenize(hyp_text.lower())

        if not ref_tokens or not hyp_tokens:
            score = 0.0
        else:
            # Pass reference as a list (meteor_score supports multiple references)
            score = meteor_score([ref_tokens], hyp_tokens)

        scores.append(round(score, 4))

        if i % 50 == 0 or i == total:
            logger.info("  Scored %d / %d samples ...", i, total)

    return scores


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═════════════════════════════════════════════════════════════════════════════

def _aggregate(records: List[Dict[str, Any]], scores: List[float]) -> Dict:
    """
    Compute per-difficulty and overall METEOR statistics.

    Returns a dict with:
        overall:    {mean, min, max, count}
        by_difficulty: {easy|medium|hard: {mean, min, max, count}}
        samples:    list of per-sample dicts (for JSON output)
    """
    difficulty_groups: Dict[str, List[float]] = defaultdict(list)
    samples = []

    for rec, sc in zip(records, scores):
        diff = rec["difficulty"]
        difficulty_groups[diff].append(sc)
        samples.append({**rec, "meteor_score": sc})

    def _stats(vals: List[float]) -> Dict:
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "mean":  round(sum(vals) / len(vals), 4),
            "min":   round(min(vals), 4),
            "max":   round(max(vals), 4),
            "count": len(vals),
        }

    all_scores = [s for s in scores]
    return {
        "overall":        _stats(all_scores),
        "by_difficulty":  {d: _stats(v) for d, v in difficulty_groups.items()},
        "samples":        samples,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Output Writers
# ═════════════════════════════════════════════════════════════════════════════

def _save_json(data: Dict, timestamp: str) -> str:
    path = os.path.join(OUTPUT_DIR, f"meteor_eval_results_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("  JSON results saved -> %s", path)
    return path


def _save_csv(data: Dict, timestamp: str) -> str:
    path = os.path.join(OUTPUT_DIR, f"meteor_eval_summary_{timestamp}.csv")
    fieldnames = [
        "qid", "difficulty", "cluster_id", "question",
        "reference_answer", "generated_answer",
        "meteor_score",
        "bert_precision", "bert_recall", "bert_f1",
        "latency_seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["samples"])
    logger.info("  CSV summary saved  -> %s", path)
    return path


def _save_markdown(data: Dict, timestamp: str, source_csv: str) -> str:
    """
    Generate a clean Markdown report with:
        - Overall METEOR stats
        - Per-difficulty breakdown table
        - Top 3 and Bottom 3 performing examples
        - Brief explanation of METEOR
    """
    path = os.path.join(OUTPUT_DIR, f"meteor_eval_report_{timestamp}.md")

    overall = data["overall"]
    by_diff = data["by_difficulty"]
    samples = data["samples"]

    # Interpret score
    mean_score = overall["mean"]
    if mean_score >= 0.50:
        interpretation = "**Excellent** -- Answers are highly aligned with references (synonyms & stems matched)."
    elif mean_score >= 0.35:
        interpretation = "**Good** -- Answers capture the core meaning with some gaps in synonym coverage."
    elif mean_score >= 0.20:
        interpretation = "**Fair** -- Partial semantic overlap; consider improving retrieval or prompt."
    else:
        interpretation = "**Poor** -- Low alignment. The generated answers diverge significantly from references."

    # Top 3 / Bottom 3
    sorted_samples = sorted(samples, key=lambda x: x["meteor_score"], reverse=True)
    top3    = sorted_samples[:3]
    bottom3 = sorted_samples[-3:]

    lines = [
        "# RAG Generation Evaluation Report -- METEOR Score",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Source CSV:** `{os.path.basename(source_csv)}`",
        f"**Total Questions Evaluated:** {overall['count']}",
        "",
        "## What is METEOR?",
        "",
        "> METEOR (Metric for Evaluation of Translation with Explicit ORdering) scores",
        "> how well a generated answer aligns with a reference answer by matching words",
        "> using **exact match**, **stemming** (run/running), and **WordNet synonyms**",
        "> (car/automobile). It then applies a **fragmentation penalty** to reward",
        "> answers whose matching words appear in the correct order.",
        "",
        "## Overall METEOR Performance",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Mean METEOR Score** | {overall['mean']:.4f} |",
        f"| **Min Score**         | {overall['min']:.4f} |",
        f"| **Max Score**         | {overall['max']:.4f} |",
        f"| **Total Samples**     | {overall['count']} |",
        "",
        "## Interpretation",
        "",
        f"> {interpretation}",
        "",
        "## Performance by Difficulty",
        "",
        "| Difficulty | Mean METEOR | Min | Max | Count |",
        "|---|---|---|---|---|",
    ]

    for diff in ["easy", "medium", "hard"]:
        stats = by_diff.get(diff, {})
        lines.append(
            f"| {diff.capitalize()} "
            f"| {stats.get('mean', 0):.4f} "
            f"| {stats.get('min', 0):.4f} "
            f"| {stats.get('max', 0):.4f} "
            f"| {stats.get('count', 0)} |"
        )

    lines += [
        "",
        "## Comparison with BERTScore",
        "",
        "| Metric | Mean Score |",
        "|---|---|",
        f"| **METEOR**         | {overall['mean']:.4f} |",
    ]

    # Compute overall BERTScore averages if present
    bert_f1_vals = [s.get("bert_f1", 0) for s in samples if s.get("bert_f1")]
    if bert_f1_vals:
        avg_bert_f1 = sum(bert_f1_vals) / len(bert_f1_vals)
        lines.append(f"| **BERTScore F1**    | {avg_bert_f1:.4f} |")

    lines += [
        "",
        "> **Note:** METEOR scores are typically lower than BERTScore F1 because",
        "> BERTScore uses deep contextual embeddings while METEOR uses lexical",
        "> matching with WordNet. Both metrics are complementary.",
        "",
        "## Sample Generations",
        "",
        "### Top Performing Examples",
        "",
    ]

    for rank, s in enumerate(top3, 1):
        lines += [
            f"**{rank}. Question:** {s['question']} *(Difficulty: {s['difficulty']})*",
            "",
            f"- **Reference Answer:** {s['reference_answer']}",
            f"- **Generated Answer:** {s['generated_answer']}",
            f"- **METEOR Score:** {s['meteor_score']:.4f}",
            "",
            "---",
            "",
        ]

    lines += [
        "### Lowest Performing Examples",
        "",
    ]

    for rank, s in enumerate(bottom3, 1):
        lines += [
            f"**{rank}. Question:** {s['question']} *(Difficulty: {s['difficulty']})*",
            "",
            f"- **Reference Answer:** {s['reference_answer']}",
            f"- **Generated Answer:** {s['generated_answer']}",
            f"- **METEOR Score:** {s['meteor_score']:.4f}",
            "",
            "---",
            "",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("  Markdown report saved -> %s", path)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG-generated answers using METEOR score."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to a specific generation_eval_summary_*.csv file. "
             "Defaults to the most recent one in data/evaluation/.",
    )
    parser.add_argument(
        "--num", type=int, default=None,
        help="Limit evaluation to the first N rows (for quick testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("  METEOR EVALUATOR")
    logger.info("  Time: %s", timestamp)
    logger.info("=" * 70)

    # ── Step 1: Bootstrap NLTK ────────────────────────────────────────────────
    logger.info("Checking NLTK resources ...")
    _bootstrap_nltk()
    logger.info("  NLTK resources ready.")

    # ── Step 2: Load data ─────────────────────────────────────────────────────
    source_csv = args.csv if args.csv else _find_latest_bert_csv()
    records = _load_from_csv(source_csv, num=args.num)

    if not records:
        logger.error("No records found in CSV. Exiting.")
        sys.exit(1)

    # Difficulty summary
    diff_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        diff_counts[r["difficulty"]] += 1
    logger.info(
        "  Difficulties: %s",
        {k: diff_counts[k] for k in sorted(diff_counts)}
    )

    # ── Step 3: Compute METEOR scores ─────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Computing METEOR scores for %d samples ...", len(records))
    logger.info("=" * 60)

    t0 = time.time()
    meteor_scores = compute_meteor_scores(records)
    elapsed = time.time() - t0
    logger.info("  METEOR computation completed in %.1f seconds", elapsed)

    # ── Step 4: Aggregate ─────────────────────────────────────────────────────
    data = _aggregate(records, meteor_scores)

    # ── Step 5: Print console summary ─────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("  METEOR EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  %-12s  %10s  %10s  %10s  %6s", "Difficulty", "Mean", "Min", "Max", "Count")
    logger.info("  " + "-" * 56)

    for diff in ["easy", "medium", "hard"]:
        s = data["by_difficulty"].get(diff, {})
        if s:
            logger.info(
                "  %-12s  %10.4f  %10.4f  %10.4f  %6d",
                diff.capitalize(),
                s["mean"], s["min"], s["max"], s["count"],
            )

    logger.info("  " + "-" * 56)
    ov = data["overall"]
    logger.info(
        "  %-12s  %10.4f  %10.4f  %10.4f  %6d",
        "OVERALL", ov["mean"], ov["min"], ov["max"], ov["count"],
    )
    logger.info("=" * 70)

    # Interpretation
    mean = ov["mean"]
    if mean >= 0.50:
        label = "[*] Excellent -- answers are highly aligned with references."
    elif mean >= 0.35:
        label = "[+] Good -- answers capture core meaning with some gaps."
    elif mean >= 0.20:
        label = "[~] Fair -- partial overlap; consider tuning retrieval/prompt."
    else:
        label = "[!] Poor -- answers diverge significantly from references."

    logger.info("")
    logger.info("  %s", label)
    logger.info("")

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = _save_json(data, timestamp)
    csv_path  = _save_csv(data, timestamp)
    md_path   = _save_markdown(data, timestamp, source_csv)

    logger.info("")
    logger.info("  Evaluation complete!")
    logger.info("  JSON results  : %s", json_path)
    logger.info("  CSV summary   : %s", csv_path)
    logger.info("  Markdown report: %s", md_path)


if __name__ == "__main__":
    main()
