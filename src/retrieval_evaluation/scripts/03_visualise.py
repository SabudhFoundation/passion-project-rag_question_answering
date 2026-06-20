"""
STEP 3: VISUALISE & REPORT
===========================
Loads eval_results.json and prints full formatted report.
Saves eval_report.txt and eval_report.md

Run: python scripts/03_visualise.py
"""

import json
import os

RESULTS_DIR = "results"

PIPELINE_LABELS = {
    "dense":           "Dense (BGE-M3)",
    "bm25":            "BM25 (Lexical)",
    "hybrid":          "Hybrid (a=0.6)",
    "hybrid_reranked": "Hybrid+Reranked",
    "query_expansion": "Query Expansion",
    "query_splitting": "Query Splitting",
}

K_VALUES = [3, 5, 10]


def load_results():
    path = f"{RESULTS_DIR}/eval_results.json"
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run 02_evaluate.py first.")
        exit(1)
    with open(path) as f:
        return json.load(f)


def section_table(results, mode_label):
    lines = []
    lines.append(f"\n  +-- {mode_label} " + "-"*80)
    hdr = f"  | {'Pipeline':<22}"
    for k in K_VALUES:
        hdr += f"  {'P@'+str(k):>7} {'R@'+str(k):>7} {'H@'+str(k):>7} {'N@'+str(k):>7} {'M@'+str(k):>7}"
    lines.append(hdr)
    lines.append("  | " + "-"*100)
    for p, m in results.items():
        label = PIPELINE_LABELS.get(p, p)
        row   = f"  | {label:<22}"
        for k in K_VALUES:
            row += (f"  {m[f'precision@{k}']:>7.4f}"
                    f" {m[f'recall@{k}']:>7.4f}"
                    f" {m[f'hit@{k}']:>7.4f}"
                    f" {m[f'ndcg@{k}']:>7.4f}"
                    f" {m[f'mrr@{k}']:>7.4f}")
        lines.append(row)
    lines.append("  +" + "-"*103)
    return lines


def best_pipeline_summary(results, mode_label):
    lines = []
    lines.append(f"\n  BEST PIPELINE — {mode_label}")
    lines.append(f"  {'Metric':<16}  {'Best Pipeline':<22}  {'Score':>8}")
    lines.append("  " + "-"*52)
    key_metrics = (
        [f"recall@{k}"    for k in K_VALUES] +
        [f"precision@{k}" for k in K_VALUES] +
        [f"hit@{k}"       for k in K_VALUES]
    )
    for metric in key_metrics:
        best_p = max(results.keys(), key=lambda p: results[p].get(metric, 0))
        val    = results[best_p].get(metric, 0)
        label  = PIPELINE_LABELS.get(best_p, best_p)
        lines.append(f"  {metric.upper():<16}  {label:<22}  {val:>8.4f}")
    return lines


def improvement_over_baseline(results, baseline="hybrid_reranked", mode_label="ORIGINAL"):
    """Show how new pipelines compare to hybrid_reranked baseline."""
    lines = []
    lines.append(f"\n  QUERY EXPANSION & SPLITTING vs HYBRID+RERANKED — {mode_label}")
    lines.append(f"  {'Pipeline':<22}  {'Metric':<16}  {'Baseline':>9}  {'New':>9}  {'Change':>8}")
    lines.append("  " + "-"*72)
    base = results.get(baseline, {})
    for p in ["query_expansion", "query_splitting"]:
        if p not in results:
            continue
        label = PIPELINE_LABELS.get(p, p)
        for metric in [f"recall@{k}" for k in K_VALUES] + [f"precision@{k}" for k in K_VALUES]:
            b     = base.get(metric, 0)
            v     = results[p].get(metric, 0)
            delta = v - b
            arrow = "▲" if delta >= 0 else "▼"
            lines.append(f"  {label:<22}  {metric.upper():<16}  {b:>9.4f}  {v:>9.4f}  {arrow} {abs(delta):.4f}")
        lines.append("")
    return lines


def robustness_section(orig, para):
    lines = []
    lines.append(f"\n  ROBUSTNESS — ORIGINAL vs PARAPHRASE (negative = performance drop)")
    lines.append(f"  {'Pipeline':<22}  {'Metric':<16}  {'Original':>9}  {'Paraphrase':>10}  {'Drop':>8}")
    lines.append("  " + "-"*75)
    for p in orig:
        label = PIPELINE_LABELS.get(p, p)
        for metric in [f"recall@{k}" for k in K_VALUES] + [f"precision@{k}" for k in K_VALUES]:
            o = orig[p].get(metric, 0)
            q = para[p].get(metric, 0) if p in para else 0
            d = q - o
            lines.append(f"  {label:<22}  {metric.upper():<16}  {o:>9.4f}  {q:>10.4f}  {d:>+8.4f}")
    return lines


def pipeline_descriptions():
    lines = []
    lines.append("\n  PIPELINE DESCRIPTIONS")
    lines.append("  " + "-"*80)
    descs = [
        ("Dense (BGE-M3)",  "BGE-M3 query vector → Pinecone ANN search (cosine similarity)"),
        ("BM25 (Lexical)",  "BM25Okapi keyword matching — no semantic understanding"),
        ("Hybrid (a=0.6)",  "0.6*dense_norm + 0.4*bm25_norm after min-max normalization"),
        ("Hybrid+Reranked", "Hybrid top-20 → ms-marco-MiniLM-L-6-v2 cross-encoder rerank"),
        ("Query Expansion",  "Groq LLM expands query → retrieve on both → RRF merge → rerank"),
        ("Query Splitting",  "Groq LLM splits query into 2-3 sub-queries → retrieve each → RRF merge → rerank"),
    ]
    for name, desc in descs:
        lines.append(f"  {name:<22}  {desc}")
    return lines


def build_full_report(results):
    orig = results.get("original", {})
    para = results.get("paraphrase", {})

    lines = []
    lines.append("=" * 110)
    lines.append("  RAG EVALUATION REPORT — BAAI/bge-m3")
    lines.append("  Model    : BAAI/bge-m3  (Dense 1024-dim, fp16)")
    lines.append("  Dataset  : HotpotQA (distractor)")
    lines.append("  Pipelines: Dense | BM25 | Hybrid | Hybrid+Reranked | Query Expansion | Query Splitting")
    lines.append("  Metrics  : Precision | Recall | Hit | NDCG | MRR  @  K = 3, 5, 10")
    lines.append("  Modes    : Original Queries  +  Paraphrased Queries (Groq llama-3.1-8b-instant)")
    lines.append("=" * 110)

    # Pipeline descriptions
    lines += pipeline_descriptions()

    # Tables
    lines.append("\n\n  [P=Precision  R=Recall  H=Hit  N=NDCG  M=MRR]")
    lines += section_table(orig, "ORIGINAL QUERIES")
    lines += section_table(para, "PARAPHRASED QUERIES")

    # Best pipeline
    lines.append("\n" + "-"*110)
    lines += best_pipeline_summary(orig, "ORIGINAL QUERIES")
    lines += best_pipeline_summary(para, "PARAPHRASED QUERIES")

    # Query Expansion & Splitting vs baseline
    lines.append("\n" + "-"*110)
    lines += improvement_over_baseline(orig, mode_label="ORIGINAL")
    lines += improvement_over_baseline(para, mode_label="PARAPHRASE")

    # Robustness
    lines.append("\n" + "-"*110)
    lines += robustness_section(orig, para)

    lines.append("\n" + "=" * 110)
    return lines


def build_markdown(results):
    orig = results.get("original", {})
    para = results.get("paraphrase", {})
    lines = []
    lines.append("# RAG Evaluation Report — BAAI/bge-m3\n")
    lines.append("**Model:** `BAAI/bge-m3` | **Dataset:** HotpotQA | **Metrics:** Precision, Recall, Hit, NDCG, MRR @ K=3,5,10\n")
    lines.append("**Pipelines:** Dense | BM25 | Hybrid (α=0.6) | Hybrid+Reranked | Query Expansion | Query Splitting\n")

    for mode_label, res in [("Original Queries", orig), ("Paraphrased Queries", para)]:
        lines.append(f"\n## {mode_label}\n")
        header = "| Pipeline | " + " | ".join(
            f"P@{k} | R@{k} | H@{k} | N@{k} | M@{k}" for k in K_VALUES
        ) + " |"
        sep = "|---|" + "|---|" * (5 * len(K_VALUES))
        lines.append(header)
        lines.append(sep)
        for p, m in res.items():
            label = PIPELINE_LABELS.get(p, p)
            row   = f"| **{label}** |"
            for k in K_VALUES:
                row += (f" {m[f'precision@{k}']:.4f} |"
                        f" {m[f'recall@{k}']:.4f} |"
                        f" {m[f'hit@{k}']:.4f} |"
                        f" {m[f'ndcg@{k}']:.4f} |"
                        f" {m[f'mrr@{k}']:.4f} |")
            lines.append(row)
    return "\n".join(lines)


def main():
    results      = load_results()
    report_lines = build_full_report(results)
    report_text  = "\n".join(report_lines)

    # Print to console
    print(report_text)

    # Save .txt
    txt_path = f"{RESULTS_DIR}/eval_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n✅  Report saved  → {txt_path}")

    # Save .md
    md_path = f"{RESULTS_DIR}/eval_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(results))
    print(f"✅  Markdown saved → {md_path}")


if __name__ == "__main__":
    main()
