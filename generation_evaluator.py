"""
generation_evaluator.py
========================
Evaluates the RAG pipeline's generated answers against the reference
(gold) answers from the synthetic QA dataset using **BERTScore**.

BERTScore leverages contextual embeddings from pre-trained transformers
(default: roberta-large) to compute token-level
similarity between the generated and reference answers, yielding:

    - Precision  -- How much of the generated answer is grounded in the reference.
    - Recall     -- How much of the reference answer is captured by the generation.
    - F1 Score   -- Harmonic mean of precision and recall.

USAGE:
    # Evaluate all questions in the synthetic dataset
    python generation_evaluator.py

    # Evaluate only the first 30 questions
    python generation_evaluator.py --num 30

    # Use a specific BERTScore model
    python generation_evaluator.py --model roberta-large

    # Skip pipeline inference and use cached results
    python generation_evaluator.py --cached results.json

OUTPUTS:
    - Console table with per-difficulty and overall BERTScore metrics
    - JSON results file saved to data/evaluation/generation_eval_results_<timestamp>.json
    - CSV summary saved to data/evaluation/generation_eval_summary_<timestamp>.csv
    - Markdown report saved to data/evaluation/generation_eval_report_<timestamp>.md
"""

import os
import sys
import json
import csv
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# ── Setup Python path ────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SYNTHETIC_DATA_DIR = os.path.join(PROJECT_DIR, "data", "evaluation")
DEFAULT_BERT_MODEL = "roberta-large"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "evaluation")


def _find_synthetic_data_file() -> str:
    """
    Locates the synthetic QA JSON file in data/evaluation/.

    Returns:
        Absolute path to the synthetic data JSON file.

    Raises:
        FileNotFoundError: if no JSON file is found in the directory.
    """
    if not os.path.isdir(SYNTHETIC_DATA_DIR):
        raise FileNotFoundError(
            f"Evaluation data directory not found: {SYNTHETIC_DATA_DIR}"
        )

    json_files = [
        f for f in os.listdir(SYNTHETIC_DATA_DIR)
        if f.endswith(".json") and f.startswith("synthetic_qa")
    ]

    if not json_files:
        raise FileNotFoundError(
            f"No synthetic QA JSON files found in {SYNTHETIC_DATA_DIR}"
        )

    # Use the most recent file if multiple exist
    json_files.sort(reverse=True)
    path = os.path.join(SYNTHETIC_DATA_DIR, json_files[0])
    logger.info("Found synthetic data file: %s", json_files[0])
    return path


def load_synthetic_data(path: str, num_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads synthetic QA records from a JSON file.

    Args:
        path:           absolute path to the synthetic QA JSON file.
        num_questions:  optional limit on the number of records to load.

    Returns:
        List of QA record dicts, each containing at minimum:
          - qid, question, answer, difficulty, cluster_id
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter to only records that were successfully generated
    data = [r for r in data if r.get("generation_ok", True)]

    if num_questions is not None:
        data = data[:num_questions]

    logger.info(
        "Loaded %d synthetic QA records (difficulties: %s)",
        len(data),
        dict(defaultdict(int, {r.get("difficulty", "unknown"): 0 for r in data}).__or__(
            _count_by_key(data, "difficulty")
        )),
    )
    return data


def _count_by_key(records: List[Dict], key: str) -> Dict[str, int]:
    """Counts records grouped by a given key."""
    counts: Dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.get(key, "unknown")] += 1
    return dict(counts)


def generate_answers_via_pipeline(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Runs each question through the RAG QueryPipeline to get generated answers.

    Args:
        records: list of synthetic QA records.

    Returns:
        List of dicts, each with:
          - qid, question, reference_answer, generated_answer,
            difficulty, cluster_id
    """
    from src.pipelines.query import QueryPipeline

    pipeline = QueryPipeline()
    results = []

    total = len(records)
    logger.info("=" * 60)
    logger.info("  Generating answers for %d questions via RAG pipeline", total)
    logger.info("=" * 60)

    for i, record in enumerate(records, 1):
        qid = record.get("qid", f"q_{i}")
        question = record.get("question", "")
        reference_answer = record.get("answer", "")
        difficulty = record.get("difficulty", "unknown")
        cluster_id = record.get("cluster_id", -1)

        if not question:
            logger.warning("  [%d/%d] Skipping empty question (qid=%s)", i, total, qid)
            continue

        logger.info(
            "  [%d/%d] (%s) Q: %s",
            i, total, difficulty, question[:80]
        )

        try:
            start_time = time.time()
            output = pipeline.run(question, verbose=False)
            elapsed = time.time() - start_time

            raw_ans = output.get("answer", "")
            if isinstance(raw_ans, list):
                generated_answer = " ".join([str(x) for x in raw_ans])
            else:
                generated_answer = str(raw_ans)

            results.append({
                "qid": qid,
                "question": question,
                "reference_answer": str(reference_answer),
                "generated_answer": generated_answer,
                "difficulty": difficulty,
                "cluster_id": cluster_id,
                "latency_seconds": round(elapsed, 2),
            })

            logger.info(
                "    -> Generated (%0.1fs): %s",
                elapsed,
                generated_answer[:100] + ("..." if len(generated_answer) > 100 else ""),
            )

        except Exception as e:
            logger.error("    -> FAILED (qid=%s): %s", qid, e)
            results.append({
                "qid": qid,
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": f"[ERROR] {e}",
                "difficulty": difficulty,
                "cluster_id": cluster_id,
                "latency_seconds": -1,
            })

    logger.info("  Generated answers for %d / %d questions", len(results), total)
    return results


def compute_bert_scores(
    generated_answers: List[str],
    reference_answers: List[str],
    model_type: str = DEFAULT_BERT_MODEL,
    batch_size: int = 32,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Computes BERTScore (Precision, Recall, F1) for each generated-reference pair.

    Uses the `bert_score` library which computes token-level contextual
    embeddings and matches them greedily between candidate and reference
    to produce precision, recall, and F1 scores.

    Args:
        generated_answers: list of generated answer strings.
        reference_answers:  list of reference (gold) answer strings.
        model_type:         HuggingFace model name for BERTScore embeddings.
        batch_size:         batch size for BERTScore computation.

    Returns:
        Tuple of (precision_list, recall_list, f1_list), each a list of floats.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.error(
            "bert_score is not installed. Install it with:\n"
            "    pip install bert-score"
        )
        raise

    logger.info("=" * 60)
    logger.info("  Computing BERTScore with model: %s", model_type)
    logger.info("  Pairs to evaluate: %d", len(generated_answers))
    logger.info("=" * 60)

    start_time = time.time()

    P, R, F1 = bert_score_fn(
        cands=generated_answers,
        refs=reference_answers,
        model_type=model_type,
        batch_size=batch_size,
        verbose=True,
        device=None,   # auto-detect GPU/CPU
    )

    elapsed = time.time() - start_time
    logger.info("  BERTScore computation completed in %.1f seconds", elapsed)

    # Convert tensors to Python floats
    precision_list = P.tolist()
    recall_list = R.tolist()
    f1_list = F1.tolist()

    return precision_list, recall_list, f1_list


def aggregate_scores(
    results: List[Dict[str, Any]],
    precision_list: List[float],
    recall_list: List[float],
    f1_list: List[float],
) -> Dict[str, Any]:
    """
    Aggregates BERTScore metrics overall and by difficulty level.

    Args:
        results:         list of result dicts (must contain 'difficulty' key).
        precision_list:  per-sample precision scores.
        recall_list:     per-sample recall scores.
        f1_list:         per-sample F1 scores.

    Returns:
        Dict with structure:
          {
            "overall":  {"precision": ..., "recall": ..., "f1": ..., "count": ...},
            "by_difficulty": {
              "easy":   {"precision": ..., "recall": ..., "f1": ..., "count": ...},
              "medium": {...},
              "hard":   {...},
            },
            "per_sample": [...]
          }
    """
    # ── Attach scores to each result ──────────────────────────────────────────
    for i, result in enumerate(results):
        result["bert_precision"] = round(precision_list[i], 4)
        result["bert_recall"] = round(recall_list[i], 4)
        result["bert_f1"] = round(f1_list[i], 4)

    # ── Overall aggregates ────────────────────────────────────────────────────
    n = len(results)
    overall = {
        "precision": round(sum(precision_list) / n, 4) if n else 0.0,
        "recall": round(sum(recall_list) / n, 4) if n else 0.0,
        "f1": round(sum(f1_list) / n, 4) if n else 0.0,
        "count": n,
    }

    # ── By difficulty ─────────────────────────────────────────────────────────
    by_difficulty: Dict[str, Dict[str, Any]] = {}
    difficulty_groups: Dict[str, List[int]] = defaultdict(list)

    for i, result in enumerate(results):
        diff = result.get("difficulty", "unknown")
        difficulty_groups[diff].append(i)

    for diff, indices in sorted(difficulty_groups.items()):
        p_vals = [precision_list[i] for i in indices]
        r_vals = [recall_list[i] for i in indices]
        f_vals = [f1_list[i] for i in indices]
        cnt = len(indices)

        by_difficulty[diff] = {
            "precision": round(sum(p_vals) / cnt, 4),
            "recall": round(sum(r_vals) / cnt, 4),
            "f1": round(sum(f_vals) / cnt, 4),
            "count": cnt,
        }

    return {
        "overall": overall,
        "by_difficulty": by_difficulty,
        "per_sample": results,
    }


def print_results_table(aggregated: Dict[str, Any]) -> None:
    """
    Prints a formatted table of BERTScore results to the console.

    Args:
        aggregated: dict from aggregate_scores().
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("  GENERATION EVALUATION -- BERTScore Results")
    logger.info("=" * 70)
    logger.info("")
    logger.info(
        "  %-12s  %10s  %10s  %10s  %6s",
        "Difficulty", "Precision", "Recall", "F1", "Count",
    )
    logger.info("  " + "-" * 54)

    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        if diff in aggregated["by_difficulty"]:
            stats = aggregated["by_difficulty"][diff]
            logger.info(
                "  %-12s  %10.4f  %10.4f  %10.4f  %6d",
                diff.capitalize(),
                stats["precision"],
                stats["recall"],
                stats["f1"],
                stats["count"],
            )

    # Any other difficulty levels
    for diff, stats in aggregated["by_difficulty"].items():
        if diff not in ["easy", "medium", "hard"]:
            logger.info(
                "  %-12s  %10.4f  %10.4f  %10.4f  %6d",
                diff.capitalize(),
                stats["precision"],
                stats["recall"],
                stats["f1"],
                stats["count"],
            )

    logger.info("  " + "-" * 54)

    overall = aggregated["overall"]
    logger.info(
        "  %-12s  %10.4f  %10.4f  %10.4f  %6d",
        "OVERALL",
        overall["precision"],
        overall["recall"],
        overall["f1"],
        overall["count"],
    )
    logger.info("=" * 70)

    # ── Interpretation guidance ───────────────────────────────────────────────
    f1 = overall["f1"]
    logger.info("")
    if f1 >= 0.90:
        logger.info("  [*] Excellent -- generated answers are highly aligned with references.")
    elif f1 >= 0.80:
        logger.info("  [+] Good -- generated answers capture most of the reference meaning.")
    elif f1 >= 0.70:
        logger.info("  [~] Fair -- there is moderate semantic overlap; room for improvement.")
    elif f1 >= 0.60:
        logger.info("  [!] Below Average -- significant gaps between generated and reference answers.")
    else:
        logger.info("  [-] Poor -- generated answers diverge substantially from references.")
    logger.info("")


def save_results(
    aggregated: Dict[str, Any],
    bert_model: str,
    timestamp: str,
) -> Tuple[str, str]:
    """
    Saves evaluation results to JSON and CSV files.

    Args:
        aggregated: dict from aggregate_scores().
        bert_model: model name used for BERTScore.
        timestamp:  timestamp string for file naming.

    Returns:
        Tuple of (json_path, csv_path).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── JSON (full results) ───────────────────────────────────────────────────
    json_output = {
        "evaluation_type": "generation_bertscore",
        "bert_model": bert_model,
        "evaluated_at": timestamp,
        "overall": aggregated["overall"],
        "by_difficulty": aggregated["by_difficulty"],
        "per_sample": aggregated["per_sample"],
    }

    json_filename = f"generation_eval_results_{timestamp}.json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    logger.info("  JSON results saved -> %s", json_path)

    # ── CSV (per-sample summary) ──────────────────────────────────────────────
    csv_filename = f"generation_eval_summary_{timestamp}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)

    fieldnames = [
        "qid", "difficulty", "cluster_id",
        "question", "reference_answer", "generated_answer",
        "bert_precision", "bert_recall", "bert_f1",
        "latency_seconds",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in aggregated["per_sample"]:
            writer.writerow(row)

    logger.info("  CSV summary saved  -> %s", csv_path)

    return json_path, csv_path


def save_markdown_report(
    aggregated: Dict[str, Any],
    bert_model: str,
    timestamp: str,
) -> str:
    """
    Saves a presentable Markdown report summarizing the evaluation results.

    Args:
        aggregated: dict from aggregate_scores().
        bert_model: model name used for BERTScore.
        timestamp:  timestamp string for file naming.

    Returns:
        Absolute path to the saved Markdown file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    md_filename = f"generation_eval_report_{timestamp}.md"
    md_path = os.path.join(OUTPUT_DIR, md_filename)

    overall = aggregated["overall"]
    by_diff = aggregated["by_difficulty"]
    
    # Select a few examples (best and worst F1 scores)
    samples = sorted(aggregated["per_sample"], key=lambda x: x["bert_f1"])
    worst_samples = samples[:3] if len(samples) >= 3 else samples
    best_samples = samples[-3:] if len(samples) >= 3 else samples

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RAG Generation Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**BERT Model Used:** `{bert_model}`\n")
        f.write(f"**Total Questions Evaluated:** {overall['count']}\n\n")

        f.write("## Overall Performance\n\n")
        f.write("| Metric | Score |\n")
        f.write("|---|---|\n")
        f.write(f"| **Precision** | {overall['precision']:.4f} |\n")
        f.write(f"| **Recall**    | {overall['recall']:.4f} |\n")
        f.write(f"| **F1 Score**  | {overall['f1']:.4f} |\n\n")

        f.write("## Performance by Difficulty\n\n")
        f.write("| Difficulty | Precision | Recall | F1 Score | Count |\n")
        f.write("|---|---|---|---|---|\n")
        for diff in ["easy", "medium", "hard"]:
            if diff in by_diff:
                stats = by_diff[diff]
                f.write(f"| {diff.capitalize()} | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | {stats['count']} |\n")
        f.write("\n")

        f.write("## Interpretation\n\n")
        f1 = overall["f1"]
        if f1 >= 0.90:
            f.write("> **Excellent** -- Generated answers are highly aligned with references.\n\n")
        elif f1 >= 0.80:
            f.write("> **Good** -- Generated answers capture most of the reference meaning.\n\n")
        elif f1 >= 0.70:
            f.write("> **Fair** -- There is moderate semantic overlap; room for improvement.\n\n")
        elif f1 >= 0.60:
            f.write("> **Below Average** -- Significant gaps between generated and reference answers.\n\n")
        else:
            f.write("> **Poor** -- Generated answers diverge substantially from references.\n\n")

        f.write("## Sample Generations\n\n")
        
        f.write("### Top Performing Examples\n\n")
        for i, s in enumerate(reversed(best_samples), 1):
            f.write(f"**{i}. Question:** {s['question']} (Difficulty: {s['difficulty']})\n\n")
            f.write(f"- **Reference Answer:** {s['reference_answer']}\n")
            f.write(f"- **Generated Answer:** {s['generated_answer']}\n")
            f.write(f"- **Scores:** Precision: {s['bert_precision']:.4f}, Recall: {s['bert_recall']:.4f}, F1: {s['bert_f1']:.4f}\n\n")
            f.write("---\n\n")

        f.write("### Lowest Performing Examples\n\n")
        for i, s in enumerate(worst_samples, 1):
            f.write(f"**{i}. Question:** {s['question']} (Difficulty: {s['difficulty']})\n\n")
            f.write(f"- **Reference Answer:** {s['reference_answer']}\n")
            f.write(f"- **Generated Answer:** {s['generated_answer']}\n")
            f.write(f"- **Scores:** Precision: {s['bert_precision']:.4f}, Recall: {s['bert_recall']:.4f}, F1: {s['bert_f1']:.4f}\n\n")
            f.write("---\n\n")

    logger.info("  Markdown report saved -> %s", md_path)
    return md_path



def run_evaluation(
    num_questions: Optional[int] = None,
    bert_model: str = DEFAULT_BERT_MODEL,
    batch_size: int = 32,
    cached_results_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main evaluation orchestrator.

    Steps:
        1. Load synthetic QA data
        2. Generate answers via RAG pipeline (or load cached results)
        3. Compute BERTScore for each generated-reference pair
        4. Aggregate and display results
        5. Save results to disk

    Args:
        num_questions:       optional limit on number of questions.
        bert_model:          HuggingFace model for BERTScore.
        batch_size:          batch size for BERTScore computation.
        cached_results_path: optional path to pre-computed results JSON.

    Returns:
        Aggregated results dict.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("  GENERATION EVALUATOR -- BERTScore Pipeline")
    logger.info("  Model: %s", bert_model)
    logger.info("  Time:  %s", timestamp)
    logger.info("=" * 70)

    # ── Step 1: Get generated + reference answers ─────────────────────────────
    if cached_results_path:
        logger.info("Loading cached results from: %s", cached_results_path)
        with open(cached_results_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        results = cached.get("per_sample", cached)
        if isinstance(results, dict):
            results = [results]
    else:
        # Load synthetic data
        data_path = _find_synthetic_data_file()
        records = load_synthetic_data(data_path, num_questions)

        # Generate answers via RAG pipeline
        results = generate_answers_via_pipeline(records)

    if not results:
        logger.error("No results to evaluate. Exiting.")
        return {}

    # ── Step 2: Extract answer pairs ──────────────────────────────────────────
    # Filter out error results
    valid_results = [
        r for r in results
        if not str(r.get("generated_answer", "")).startswith("[ERROR]")
    ]

    if not valid_results:
        logger.error("All generation attempts failed. Nothing to score.")
        return {}

    generated_answers = [r["generated_answer"] for r in valid_results]
    reference_answers = [r["reference_answer"] for r in valid_results]

    skipped = len(results) - len(valid_results)
    if skipped > 0:
        logger.warning("  Skipped %d failed generations", skipped)

    # ── Step 3: Compute BERTScore ─────────────────────────────────────────────
    precision_list, recall_list, f1_list = compute_bert_scores(
        generated_answers=generated_answers,
        reference_answers=reference_answers,
        model_type=bert_model,
        batch_size=batch_size,
    )

    # ── Step 4: Aggregate ─────────────────────────────────────────────────────
    aggregated = aggregate_scores(valid_results, precision_list, recall_list, f1_list)

    # ── Step 5: Display ───────────────────────────────────────────────────────
    print_results_table(aggregated)

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    json_path, csv_path = save_results(aggregated, bert_model, timestamp)
    md_path = save_markdown_report(aggregated, bert_model, timestamp)

    logger.info("  Evaluation complete!")
    logger.info("  Full results: %s", json_path)
    logger.info("  CSV summary:  %s", csv_path)
    logger.info("  Markdown report: %s", md_path)

    return aggregated


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the generation evaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG generation quality using BERTScore",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BERT_MODEL,
        help=f"HuggingFace model for BERTScore (default: {DEFAULT_BERT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for BERTScore computation (default: 32)",
    )
    parser.add_argument(
        "--cached",
        type=str,
        default=None,
        help="Path to cached results JSON (skip pipeline inference)",
    )

    args = parser.parse_args()

    run_evaluation(
        num_questions=args.num,
        bert_model=args.model,
        batch_size=args.batch_size,
        cached_results_path=args.cached,
    )


if __name__ == "__main__":
    main()
