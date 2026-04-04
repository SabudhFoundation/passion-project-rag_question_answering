"""
src/main.py
===========
MAIN SCRIPT — Runs the complete RAG ingestion pipeline.

WHAT THIS FILE DOES:
  Connects all modules in the correct order:
    1. DataDownloader  → downloads raw data to src/data/raw/
    2. Preprocessor    → cleans data, saves to src/data/processed/
    3. Chunker         → splits into chunks, saves chunks.jsonl
    4. Embedder        → creates vectors, uploads to Pinecone

HOW TO RUN:

  Full ingestion (run once to populate Pinecone):
    cd passion-project-rag_question_answering
    python src/main.py --ingest

  Test retrieval (after ingestion):
    python src/main.py --query "Where was Albert Einstein born?"

  Evaluate system:
    python src/main.py --evaluate --num 10

TEAM STRUCTURE:
  Ayan     → ingestion  (pre-processing.py + build_features.py)
  Teammate → retrieval  (retriever.py — plug in here)
  Teammate → generation (generator.py — plug in here)
"""

import os
import sys
import json
import argparse

# ── Make sure src/ is on the Python path ─────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(SRC_DIR))

import config

# ── Import ingestion classes (YOUR code) ─────────────────────────────────────
from preprocessing_data.pre_processing   import DataDownloader, Preprocessor
from feature_engineering.build_features  import Chunker, Embedder


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

def run_ingestion():
    """
    Runs the full ingestion pipeline end-to-end.

    DATA FLOW:
      DataDownloader.download_all()
          ↓  (file paths)
      Preprocessor.process_all()
          ↓  (clean records list)
      Chunker.chunk_records()
          ↓  (chunk dicts list)
      Embedder.embed_and_upload()
          ↓  (vectors stored in Pinecone)

    Each class does ONE job and passes its output to the next.
    This is called a PIPELINE pattern.
    """
    print("\n" + "═" * 55)
    print("  RAG INGESTION PIPELINE")
    print("  HotpotQA + Wikipedia → Pinecone")
    print("═" * 55)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    downloader = DataDownloader()
    paths      = downloader.download_all()

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    preprocessor = Preprocessor()
    records      = preprocessor.process_all(
        hotpotqa_path  = paths.get("hotpotqa"),
        wikipedia_path = paths.get("wikipedia"),
    )

    if not records:
        print("❌ No records after preprocessing. Check data files.")
        return

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    chunker = Chunker()
    chunks  = chunker.chunk_records(records)

    if not chunks:
        print("❌ No chunks produced. Check preprocessing step.")
        return

    # ── Step 4: Embed + Upload ────────────────────────────────────────────────
    embedder = Embedder()
    total    = embedder.embed_and_upload(chunks)

    # ── Final Summary ─────────────────────────────────────────────────────────
    hpqa = sum(1 for r in records if r["source"] == "hotpotqa")
    wiki = sum(1 for r in records if r["source"] == "wikipedia")

    print("\n" + "═" * 55)
    print("  ✅  INGESTION COMPLETE")
    print("═" * 55)
    print(f"  HotpotQA records  : {hpqa:,}")
    print(f"  Wikipedia records : {wiki:,}")
    print(f"  Total chunks      : {len(chunks):,}")
    print(f"  Vectors in Pinecone: {total:,}")
    print(f"  Index             : {config.PINECONE_INDEX}")
    print(f"  Namespaces        : hotpotqa | wikipedia")
    print("\n  Ingestion done! Retrieval team can now use embed_query()")
    print("  and get_index() from the Embedder class.")


# =============================================================================
# QUERY PIPELINE (retrieval + generation — teammates plug in here)
# =============================================================================

def run_query(query: str) -> dict:
    """
    Runs retrieval + generation for one question.

    ════════════════════════════════════════════════════════
    INTEGRATION POINT: This is where ingestion connects
    to the retrieval and generation modules.

    Embedder exposes two methods for the retrieval team:
      embed_query(query) → 384-dim vector of the question
      get_index()        → live Pinecone index for search
    ════════════════════════════════════════════════════════
    """
    print(f"\n{'═'*55}")
    print(f"  Query: {query}")
    print(f"{'═'*55}")

    embedder = Embedder()

    # ── RETRIEVAL (teammates implement Retriever class) ───────────────────────
    # For now: basic retrieval using Embedder directly
    # When teammates provide retrieval/retriever.py, replace this block:
    #   from retrieval.retriever import Retriever
    #   retriever = Retriever()
    #   chunks = retriever.search(query)

    print("\n🔍 Retrieving relevant chunks...")
    query_vector = embedder.embed_query(query)
    index        = embedder.get_index()

    result = index.query(
        vector          = query_vector,
        top_k           = config.TOP_K,
        namespace       = config.NAMESPACE_WIKI,
        include_metadata= True,
    )

    chunks = [
        {
            "score": round(m["score"], 4),
            "text":  m["metadata"].get("original_text", ""),
            "title": m["metadata"].get("title", ""),
        }
        for m in result.get("matches", [])
    ]

    print(f"  Found {len(chunks)} relevant chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] score={c['score']:.4f} | {c['title'][:50]}")

    # ── GENERATION (teammates implement Generator class) ──────────────────────
    # When teammates provide generation/generator.py, replace this block:
    #   from generation.generator import Generator
    #   generator = Generator()
    #   result = generator.generate(query, chunks)

    if chunks:
        print(f"\n  📝 Top result: {chunks[0]['text'][:300]}...")
    else:
        print("\n  ❌ No relevant context found.")

    return {"query": query, "chunks": chunks}


# =============================================================================
# EVALUATION
# =============================================================================

def run_evaluation(num: int = 10):
    """Evaluates the system on HotpotQA questions using Exact Match."""
    print(f"\n{'═'*55}")
    print(f"  EVALUATION — {num} questions")
    print(f"{'═'*55}")

    if not os.path.exists(config.HOTPOTQA_FILE):
        print("❌ HotpotQA not found. Run --ingest first.")
        return

    with open(config.HOTPOTQA_FILE) as f:
        data = json.load(f)[:num]

    embedder = Embedder()
    index    = embedder.get_index()
    correct  = 0

    for i, entry in enumerate(data, 1):
        q    = entry.get("question", "")
        gold = entry.get("answer", "").lower().strip()
        if not q:
            continue

        vec    = embedder.embed_query(q)
        result = index.query(vector=vec, top_k=3,
                             namespace=config.NAMESPACE_WIKI,
                             include_metadata=True)
        top_text = result["matches"][0]["metadata"].get(
            "original_text", "") if result["matches"] else ""

        pred = top_text[:200].lower()
        ok   = gold in pred
        if ok:
            correct += 1

        print(f"  [{i:2d}] {'✅' if ok else '❌'} "
              f"Q: {q[:45]} | Gold: {gold[:25]}")

    em = correct / len(data) * 100 if data else 0
    print(f"\n  Exact Match: {em:.1f}% ({correct}/{len(data)})")
    print("  Baseline EM of 20-35% is normal at this stage.")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG Question Answering System")
    parser.add_argument("--ingest",   action="store_true",
                        help="Run full ingestion pipeline")
    parser.add_argument("--query",    type=str, default=None,
                        help="Ask a question")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on HotpotQA")
    parser.add_argument("--num",      type=int, default=10,
                        help="Number of questions to evaluate")
    args = parser.parse_args()

    if args.ingest:
        run_ingestion()
    elif args.query:
        run_query(args.query)
    elif args.evaluate:
        run_evaluation(args.num)
    else:
        print("\n" + "═" * 55)
        print("  RAG Question Answering System")
        print("═" * 55)
        print("\n  Commands:")
        print("    python src/main.py --ingest")
        print("    python src/main.py --query \"Your question here\"")
        print("    python src/main.py --evaluate --num 20")
        print("\n  First time? Run --ingest to populate Pinecone.")


if __name__ == "__main__":
    main()
