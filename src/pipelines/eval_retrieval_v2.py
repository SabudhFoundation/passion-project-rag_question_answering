"""
src/pipelines/eval_retrieval_v2.py
==================================
Evaluation pipeline for the current HybridRetriever using the synthetic dataset.
Computes Hit@K, Precision@K, Recall@K, MRR@K, and NDCG@K for K in [1, 3, 5, 10].
Optionally evaluates robustness to paraphrasing using Groq.

Usage:
    ./venv/bin/python src/pipelines/eval_retrieval_v2.py --num 50
"""

import os
import sys
import json
import time
import math
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from groq import Groq

# Insert src directory to path
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from models.retriever import HybridRetriever

logger = get_logger(__name__)

# Ground-truth evaluation file path
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
EVAL_FILE = os.path.join(
    _PROJECT_ROOT, "data", "evaluation",
    "synthetic_qa_hotpotqa_20260506_063747.json"
)
EVAL_OUT_DIR = os.path.join(_PROJECT_ROOT, "data", "evaluation")


# ── Text Normalization and Matching ──────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Normalizes text for substring matching."""
    return " ".join(text.lower().split())


def is_chunk_match(retrieved_text: str, ground_truth_chunks: list) -> tuple:
    """
    Checks if a retrieved chunk matches any of the ground truth chunks.
    Returns (is_match, matched_gt_idx).
    """
    ret_norm = _normalize_text(retrieved_text)
    ret_words = set(ret_norm.split())
    
    for idx, gt in enumerate(ground_truth_chunks):
        gt_norm = _normalize_text(gt.get("text", ""))
        gt_words = set(gt_norm.split())
        
        # Substring containment
        if gt_norm in ret_norm or ret_norm in gt_norm:
            return True, idx
            
        # Jaccard similarity fallback
        if ret_words and gt_words:
            jaccard = len(ret_words.intersection(gt_words)) / len(ret_words.union(gt_words))
            if jaccard >= 0.4:
                return True, idx
                
    return False, -1


# ── Paraphrasing logic using Groq ─────────────────────────────────────────────

def paraphrase_question(groq_client: Groq, question: str) -> str:
    """Uses Groq to paraphrase a question for robustness checks."""
    prompt = (
        "Rewrite the following question in different words but keep the exact same meaning. "
        "Return ONLY the rewritten question, nothing else.\n\n"
        f"Question: {question}"
    )
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=150,
            )
            result = resp.choices[0].message.content.strip()
            if result and result.lower() != question.lower():
                return result
        except Exception as e:
            logger.warning(f"Paraphrase attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return question


# ── Metric Computation ────────────────────────────────────────────────────────

def compute_metrics(retrieved_chunks: list, ground_truth_chunks: list, ks=(1, 3, 5, 10)) -> dict:
    """
    Computes retrieval metrics at various K.
    """
    total_relevant = len(ground_truth_chunks)
    out = {}
    
    for k in ks:
        # 1. Hit Rate
        hit = 0
        for i in range(min(k, len(retrieved_chunks))):
            r_text = retrieved_chunks[i].get("text", "")
            matched, _ = is_chunk_match(r_text, ground_truth_chunks)
            if matched:
                hit = 1
                break
        out[f"hit@{k}"] = hit
        
        # 2. Precision
        relevant_retrieved_in_k = 0
        for i in range(min(k, len(retrieved_chunks))):
            r_text = retrieved_chunks[i].get("text", "")
            matched, _ = is_chunk_match(r_text, ground_truth_chunks)
            if matched:
                relevant_retrieved_in_k += 1
        out[f"prec@{k}"] = relevant_retrieved_in_k / k if k > 0 else 0.0
        
        # 3. Recall (unique gold chunks retrieved)
        found_gt_in_k = set()
        for i in range(min(k, len(retrieved_chunks))):
            r_text = retrieved_chunks[i].get("text", "")
            matched, gt_idx = is_chunk_match(r_text, ground_truth_chunks)
            if matched:
                found_gt_in_k.add(gt_idx)
        out[f"recall@{k}"] = len(found_gt_in_k) / total_relevant if total_relevant > 0 else 0.0
        
        # 4. MRR
        mrr = 0.0
        for rank, i in enumerate(range(min(k, len(retrieved_chunks))), 1):
            r_text = retrieved_chunks[i].get("text", "")
            matched, _ = is_chunk_match(r_text, ground_truth_chunks)
            if matched:
                mrr = 1.0 / rank
                break
        out[f"mrr@{k}"] = mrr
        
        # 5. NDCG
        dcg = 0.0
        for i in range(min(k, len(retrieved_chunks))):
            r_text = retrieved_chunks[i].get("text", "")
            matched, _ = is_chunk_match(r_text, ground_truth_chunks)
            if matched:
                dcg += 1.0 / math.log2(i + 2)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(total_relevant, k)))
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
        
    return out


def aggregate(all_m: list) -> dict:
    """Averages a list of metric dicts."""
    if not all_m:
        return {}
    return {k: round(float(np.mean([m[k] for m in all_m])), 4) for k in all_m[0]}


# ── Main Run Loop ─────────────────────────────────────────────────────────────

def run_eval(samples: list, retriever: HybridRetriever, groq_client: Groq, eval_samples: int):
    """Runs evaluation over a list of samples."""
    samples = samples[:eval_samples]
    
    res = {
        "hybrid": {"original": [], "paraphrase": []}
    }
    
    print(f"\n[INFO] Evaluating {len(samples)} questions using active HybridRetriever...")
    
    for sample in tqdm(samples):
        orig = sample["question"]
        gt_chunks = sample.get("supporting_chunks", [])
        
        if not orig or not gt_chunks:
            continue
            
        # Get paraphrase
        if groq_client:
            para = paraphrase_question(groq_client, orig)
        else:
            para = orig
            
        # Evaluate original question
        retrieved_orig = retriever.retrieve(orig)
        res["hybrid"]["original"].append(compute_metrics(retrieved_orig, gt_chunks))
        
        # Evaluate paraphrased question
        if groq_client:
            retrieved_para = retriever.retrieve(para)
            res["hybrid"]["paraphrase"].append(compute_metrics(retrieved_para, gt_chunks))
        else:
            res["hybrid"]["paraphrase"].append(res["hybrid"]["original"][-1])
            
    return {
        "hybrid": {
            v: aggregate(res["hybrid"][v]) for v in res["hybrid"]
        }
    }


def print_results(summary: dict, n: int):
    """Prints results formatted identically to step2_eval_retrieval.py."""
    print("\n" + "="*72)
    print("  RETRIEVAL EVALUATION RESULTS (ACTIVE PIPELINE)")
    print(f"  Dataset: HotpotQA | Questions evaluated: {n} | Variants: original + paraphrase")
    print("="*72)

    method = "hybrid"
    print(f"\n  ── {method.upper()} (Pinecone Hybrid + ColBERT Rerank) ─────────────────────")
    for variant in ["original", "paraphrase"]:
        m = summary[method][variant]
        print(f"\n    [{variant.upper()}]")
        for k in [1, 3, 5, 10]:
            hit_val   = m.get(f"hit@{k}", 0)
            hit_count = int(round(hit_val * n))
            rec_pct   = m.get(f"recall@{k}", 0) * 100
            prec_pct  = m.get(f"prec@{k}", 0) * 100
            mrr       = m.get(f"mrr@{k}", 0)
            ndcg      = m.get(f"ndcg@{k}", 0)
            print(f"      K={k:<3}  "
                  f"Hit={hit_val:.3f} [{hit_count}/{n} questions]  "
                  f"Recall={rec_pct:5.1f}%  "
                  f"Precision={prec_pct:5.1f}%  "
                  f"MRR={mrr:.3f}  "
                  f"NDCG={ndcg:.3f}")

    print("\n\n" + "="*72)
    print("  PARAPHRASE ROBUSTNESS SUMMARY")
    print("  Shows how much performance drops when questions are reworded")
    print("="*72)
    print(f"\n  {'Method':<10} | {'Hit@5':>8} {'Para':>8} {'Drop':>7} | "
          f"{'Recall@5':>10} {'Para':>8} {'Drop':>7}")
    print(f"  {'-'*68}")
    
    oh  = summary[method]["original"].get("hit@5", 0)
    ph  = summary[method]["paraphrase"].get("hit@5", 0)
    or_ = summary[method]["original"].get("recall@5", 0)
    pr  = summary[method]["paraphrase"].get("recall@5", 0)
    print(f"  {method:<10} | {oh:>8.3f} {ph:>8.3f} {oh-ph:>+7.3f} | "
          f"{or_:>10.3f} {pr:>8.3f} {or_-pr:>+7.3f}")

    print("\n\n" + "="*72)
    print("  METRIC REFERENCE")
    print("="*72)
    print("""
  Hit@K      : Binary — did ANY relevant doc appear in top-K?
               1 = yes (at least 1 gold doc found), 0 = no
               Hit@5=0.990 means 99/100 questions had a gold doc in top-5

  Recall@K   : What fraction of ALL relevant docs were found in top-K?
               HotpotQA has 1-2 gold docs per question.
               Recall@5=0.82 → avg 1.64 out of 2 gold docs found in top-5
               THIS is the most important metric for multi-hop QA.

  Precision@K: Of K docs retrieved, what fraction were actually relevant?
               Precision@5=20% means 1 of 5 retrieved docs is gold.

  MRR@K      : Mean Reciprocal Rank — how high is the FIRST relevant doc?
               Rank1→1.0, Rank2→0.5, Rank3→0.33. Higher = better.

  NDCG@K     : Considers both presence AND rank of relevant docs.
               Best overall ranking quality metric.
""")

def write_markdown_report(summary: dict, n: int, timestamp: str, filepath: str):
    """Generates and writes a beautiful markdown report to disk."""
    method = "hybrid"
    orig = summary[method]["original"]
    para = summary[method]["paraphrase"]
    
    oh5 = orig.get("hit@5", 0)
    ph5 = para.get("hit@5", 0)
    or5 = orig.get("recall@5", 0)
    pr5 = para.get("recall@5", 0)
    
    lines = [
        f"# Retrieval Evaluation Report ({timestamp})",
        "",
        f"**Dataset:** HotpotQA (Synthetic Ground-Truth Set)",
        f"**Retriever:** `HybridRetriever` (Pinecone Hybrid + BGE-M3 + ColBERT Reranker)",
        f"**Questions Evaluated:** {n}",
        "",
        "## Metrics Summary",
        "",
        "### [ORIGINAL]",
        "| Metric | K=1 | K=3 | K=5 | K=10 |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Hit Rate** | {orig.get('hit@1', 0):.3f} | {orig.get('hit@3', 0):.3f} | {orig.get('hit@5', 0):.3f} | {orig.get('hit@10', 0):.3f} |",
        f"| **Recall** | {orig.get('recall@1', 0)*100:.1f}% | {orig.get('recall@3', 0)*100:.1f}% | {orig.get('recall@5', 0)*100:.1f}% | {orig.get('recall@10', 0)*100:.1f}% |",
        f"| **Precision** | {orig.get('prec@1', 0)*100:.1f}% | {orig.get('prec@3', 0)*100:.1f}% | {orig.get('prec@5', 0)*100:.1f}% | {orig.get('prec@10', 0)*100:.1f}% |",
        f"| **MRR** | {orig.get('mrr@1', 0):.3f} | {orig.get('mrr@3', 0):.3f} | {orig.get('mrr@5', 0):.3f} | {orig.get('mrr@10', 0):.3f} |",
        f"| **NDCG** | {orig.get('ndcg@1', 0):.3f} | {orig.get('ndcg@3', 0):.3f} | {orig.get('ndcg@5', 0):.3f} | {orig.get('ndcg@10', 0):.3f} |",
        "",
        "### [PARAPHRASE]",
        "| Metric | K=1 | K=3 | K=5 | K=10 |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Hit Rate** | {para.get('hit@1', 0):.3f} | {para.get('hit@3', 0):.3f} | {para.get('hit@5', 0):.3f} | {para.get('hit@10', 0):.3f} |",
        f"| **Recall** | {para.get('recall@1', 0)*100:.1f}% | {para.get('recall@3', 0)*100:.1f}% | {para.get('recall@5', 0)*100:.1f}% | {para.get('recall@10', 0)*100:.1f}% |",
        f"| **Precision** | {para.get('prec@1', 0)*100:.1f}% | {para.get('prec@3', 0)*100:.1f}% | {para.get('prec@5', 0)*100:.1f}% | {para.get('prec@10', 0)*100:.1f}% |",
        f"| **MRR** | {para.get('mrr@1', 0):.3f} | {para.get('mrr@3', 0):.3f} | {para.get('mrr@5', 0):.3f} | {para.get('mrr@10', 0):.3f} |",
        f"| **NDCG** | {para.get('ndcg@1', 0):.3f} | {para.get('ndcg@3', 0):.3f} | {para.get('ndcg@5', 0):.3f} | {para.get('ndcg@10', 0):.3f} |",
        "",
        "## Paraphrase Robustness Summary",
        "Shows how much performance drops when questions are reworded.",
        "",
        "| Metric | Original | Paraphrased | Drop |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Hit@5** | {oh5:.3f} | {ph5:.3f} | {oh5-ph5:+.3f} |",
        f"| **Recall@5** | {or5:.3f} | {pr5:.3f} | {or5-pr5:+.3f} |",
        "",
        "## Metric Definitions",
        "*   **Hit@K**: Binary — did at least one relevant (gold) document appear in the top-K retrieved chunks?",
        "*   **Recall@K**: What fraction of all relevant (gold) chunks (usually 2 in HotpotQA) were retrieved in the top-K?",
        "*   **Precision@K**: What fraction of the retrieved top-K chunks were actually relevant (gold)?",
        "*   **MRR@K (Mean Reciprocal Rank)**: How high up did the first relevant chunk appear? (e.g. Rank 1 = 1.0, Rank 2 = 0.5).",
        "*   **NDCG@K (Normalized Discounted Cumulative Gain)**: Measures the overall ranking quality, considering both presence and position of relevant documents."
    ]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Saved Markdown report to: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Active HybridRetriever Pipeline")
    parser.add_argument("--num", type=int, default=50, help="Number of questions to evaluate")
    parser.add_argument("--no-paraphrase", action="store_true", help="Skip paraphrase evaluation")
    args = parser.parse_args()

    # Load synthetic dataset
    if not os.path.exists(EVAL_FILE):
        print(f"[ERROR] Synthetic eval file not found at: {EVAL_FILE}")
        sys.exit(1)
        
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    print(f"[INFO] Loaded {len(samples)} questions from evaluation dataset.")

    # Initialize active HybridRetriever
    print("[INFO] Initializing HybridRetriever (Pinecone + BGE-M3 + ColBERT)...")
    retriever = HybridRetriever(top_k=10)

    # Initialize Groq for paraphrase robustness
    groq_client = None
    if not args.no_paraphrase and config.GROQ_API_KEY:
        print("[INFO] Initializing Groq client for paraphrase robustness check...")
        groq_client = Groq(api_key=config.GROQ_API_KEY)
    else:
        print("[INFO] Paraphrase checks disabled or GROQ_API_KEY not set.")

    # Run evaluation
    num_eval = min(len(samples), args.num)
    summary = run_eval(samples, retriever, groq_client, num_eval)

    # Print results
    print_results(summary, num_eval)

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON summary
    json_path = os.path.join(EVAL_OUT_DIR, f"retrieval_eval_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Saved results JSON to: {json_path}")
    
    # Save Markdown report
    md_path = os.path.join(EVAL_OUT_DIR, f"retrieval_eval_report_{timestamp}.md")
    write_markdown_report(summary, num_eval, timestamp, md_path)

