"""
src/pipelines/eval.py
======================
Evaluates the retrieval component of the pipeline by calculating 
Precision and Recall at Top K (K = 5, 15, 20).

Usage:
    python src/pipelines/eval.py --num 50
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Set

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from logger import get_logger
from models.retriever import HybridRetriever

logger = get_logger(__name__)

# The ground truth evaluation dataset
EVAL_FILE = os.path.join(os.path.dirname(_SRC_DIR), "data", "evaluation", "synthetic_qa_hotpotqa_20260506_063747.json")


def _normalize_text(text: str) -> str:
    """Normalizes text for substring matching."""
    return " ".join(text.lower().split())


def is_chunk_match(retrieved_text: str, ground_truth_chunks: List[Dict]) -> bool:
    """
    Checks if a retrieved chunk matches any of the ground truth chunks.
    Since chunking strategies may have changed, we use substring matching 
    or high overlap.
    """
    ret_norm = _normalize_text(retrieved_text)
    for gt in ground_truth_chunks:
        gt_norm = _normalize_text(gt.get("text", ""))
        # If one is a substantial substring of the other, consider it a match
        if gt_norm in ret_norm or ret_norm in gt_norm:
            return True
        # Or if they share a lot of words (Jaccard similarity fallback)
        ret_words = set(ret_norm.split())
        gt_words = set(gt_norm.split())
        if not ret_words or not gt_words:
            continue
        intersection = ret_words.intersection(gt_words)
        jaccard = len(intersection) / float(len(ret_words.union(gt_words)))
        if jaccard > 0.4:  # 40% overlap is usually a strong indicator for chunks
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate Retrieval Precision/Recall")
    parser.add_argument("--num", type=int, default=-1, help="Number of questions to evaluate (-1 for all)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  EVALUATING RETRIEVAL: Precision & Recall @ 5, 15, 20")
    logger.info("=" * 60)

    if not os.path.exists(EVAL_FILE):
        logger.error(f"Evaluation file not found: {EVAL_FILE}")
        sys.exit(1)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        if args.num > 0:
            dataset = dataset[:args.num]

    logger.info(f"Loaded {len(dataset)} questions from evaluation set.")

    # Initialize the new Pinecone Hybrid Retriever
    logger.info("Initializing HybridRetriever (Pinecone + ColBERT)...")
    try:
        retriever = HybridRetriever(top_k=20)
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        sys.exit(1)

    # Metrics trackers
    k_values = [5, 15, 20]
    metrics = {k: {"precision": [], "recall": []} for k in k_values}

    for i, entry in enumerate(dataset, 1):
        question = entry.get("question", "")
        ground_truth_chunks = entry.get("supporting_chunks", [])
        total_relevant = len(ground_truth_chunks)

        if not question or total_relevant == 0:
            continue

        logger.info(f"[{i}/{len(dataset)}] Q: {question[:60]}...")
        
        # Retrieve top 20
        retrieved = retriever.retrieve(question)[:20]

        # Evaluate at each K
        for k in k_values:
            retrieved_k = retrieved[:k]
            
            # Count relevant chunks in top K
            relevant_in_k = 0
            # Track which GT chunks were found so we don't double count recall
            found_gt_indices = set()

            for r_chunk in retrieved_k:
                r_text = r_chunk.get("text", "")
                r_norm = _normalize_text(r_text)
                
                is_rel = False
                for gt_idx, gt in enumerate(ground_truth_chunks):
                    gt_norm = _normalize_text(gt.get("text", ""))
                    if gt_norm in r_norm or r_norm in gt_norm:
                        is_rel = True
                        found_gt_indices.add(gt_idx)
                        break
                    
                    # Jaccard fallback
                    ret_words = set(r_norm.split())
                    gt_words = set(gt_norm.split())
                    if ret_words and gt_words:
                        jaccard = len(ret_words.intersection(gt_words)) / len(ret_words.union(gt_words))
                        if jaccard > 0.4:
                            is_rel = True
                            found_gt_indices.add(gt_idx)
                            break
                
                if is_rel:
                    relevant_in_k += 1

            # Calculate Precision and Recall for this query at K
            p_at_k = relevant_in_k / k if k > 0 else 0
            # Recall is based on unique ground truth chunks found
            r_at_k = len(found_gt_indices) / total_relevant if total_relevant > 0 else 0

            metrics[k]["precision"].append(p_at_k)
            metrics[k]["recall"].append(r_at_k)

    # Calculate and print final averages
    logger.info("=" * 60)
    logger.info("  FINAL RETRIEVAL METRICS (Averages)")
    logger.info("=" * 60)
    
    report_lines = [
        "# Retrieval Evaluation Report",
        "",
        f"**Dataset:** {len(dataset)} questions from `synthetic_qa_hotpotqa`",
        "**Pipeline:** Pinecone + BGE-M3 + ColBERT Re-ranker",
        "",
        "## Metrics Summary",
        ""
    ]
    
    for k in k_values:
        avg_p = sum(metrics[k]["precision"]) / len(metrics[k]["precision"]) if metrics[k]["precision"] else 0
        avg_r = sum(metrics[k]["recall"]) / len(metrics[k]["recall"]) if metrics[k]["recall"] else 0
        
        logger.info(f"Top-{k} Metrics:")
        logger.info(f"  - Precision@{k}: {avg_p:.4f} ({(avg_p*100):.1f}%)")
        logger.info(f"  - Recall@{k}:    {avg_r:.4f} ({(avg_r*100):.1f}%)")
        logger.info("-" * 40)
        
        report_lines.extend([
            f"### Top-{k}",
            f"- **Precision@{k}:** {avg_p:.4f} ({(avg_p*100):.1f}%)",
            f"- **Recall@{k}:** {avg_r:.4f} ({(avg_r*100):.1f}%)",
            ""
        ])
        
    # Write report to disk
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(os.path.dirname(_SRC_DIR), "data", "evaluation")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"retrieval_eval_report_{timestamp}.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Report successfully saved to: {report_path}")

if __name__ == "__main__":
    main()
