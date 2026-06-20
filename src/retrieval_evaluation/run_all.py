#!/usr/bin/env python3
"""
RUN_ALL.PY — Master runner
===========================
Runs Step 1 (ingest) → Step 2 (evaluate) → Step 3 (visualise) in sequence.

Usage:
    python run_all.py
    python run_all.py --skip-ingest     # If index already populated
    python run_all.py --skip-ingest --eval-only
"""

import subprocess
import sys
import argparse
import time

def run_step(script: str, label: str):
    print("\n" + "▓" * 65)
    print(f"  ▶  {label}")
    print("▓" * 65)
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FATAL] {label} failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  ✔  {label} completed in {elapsed/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion (index already populated)")
    args = parser.parse_args()

    print("█" * 65)
    print("  RAG EVALUATION PIPELINE — BAAI/bge-m3")
    print("  4 Pipelines: Dense | BM25 | Hybrid | Hybrid+Reranked")
    print("█" * 65)

    if not args.skip_ingest:
        run_step("scripts/01_ingest.py",    "STEP 1: Ingest HotpotQA → Pinecone (bge-m3)")
    else:
        print("\n  [SKIP] Ingestion skipped (--skip-ingest flag set)")

    run_step("scripts/02_evaluate.py", "STEP 2: Evaluate 4 retrieval pipelines")
    run_step("scripts/03_visualise.py","STEP 3: Generate report & markdown")

    print("\n" + "█" * 65)
    print("  🎉  ALL STEPS COMPLETE")
    print("  Results in:  results/eval_results.json")
    print("               results/eval_results.csv")
    print("               results/eval_report.md")
    print("█" * 65)


if __name__ == "__main__":
    main()
