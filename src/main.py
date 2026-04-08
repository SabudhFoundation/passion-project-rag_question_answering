"""
src/main.py
===========
THE MAIN ORCHESTRATOR — connects all 4 classes in a clean pipeline.

WHAT THIS FILE DOES:
  Runs the complete ingestion pipeline by calling each class in order:

    DataDownloader → Preprocessor → Chunker → Embedder

  Each class does its job and passes its output to the next.
  This file does NOT contain any logic itself — it only orchestrates.

TEAM OWNERSHIP:
  Ingestion (your module)   → DataDownloader, Preprocessor, Chunker, Embedder
  Retrieval (teammates)     → Retriever class (plug in below — clearly marked)
  Generation (teammates)    → Generator class (plug in below — clearly marked)

HOW TO RUN:

  Full ingestion (run once to populate Pinecone):
    python src/main.py --ingest

  Ask a question (after ingestion is done):
    python src/main.py --query "Who directed the film that starred Leonardo DiCaprio in Titanic?"

  Evaluate on HotpotQA questions:
    python src/main.py --evaluate --num 10

  Interactive mode (keep asking questions):
    python src/main.py --interactive
"""

import os
import sys
import json
import argparse

# ── Set up Python path so imports work correctly ──────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(SRC_DIR))

import config

# ── Import YOUR ingestion classes ─────────────────────────────────────────────
from preprocessing_data.pre_processing  import DataDownloader, Preprocessor
from feature_engineering.build_features import Chunker, Embedder


# ═════════════════════════════════════════════════════════════════════════════
# INGESTION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_ingestion():
    """
    Runs the full 4-step ingestion pipeline.

    DATA FLOW (each step feeds the next):

      Step 1: DataDownloader.download()
              → Downloads hotpot_train_v1.1.json to src/data/raw/
              → Returns: file path (string)

      Step 2: Preprocessor.process(file_path)
              → Cleans text, filters to supporting paragraphs only
              → Returns: list of clean record dicts

      Step 3: Chunker.chunk_records(records)
              → Splits each record into 512-char overlapping chunks
              → Returns: list of chunk dicts with full metadata

      Step 4: Embedder.embed_and_upload(chunks)
              → Converts chunks to 384-dim vectors
              → Uploads vectors + metadata to Pinecone
              → Returns: count of vectors uploaded

    This pattern is called a PIPELINE — output of each step is
    the input of the next step. Each class is independent.
    """
    print("\n" + "═" * 55)
    print("  RAG INGESTION PIPELINE")
    print("  Dataset: HotpotQA only")
    print("═" * 55)

    # ── STEP 1: Download ──────────────────────────────────────────────────────
    print("\n[STEP 1/4] Download")
    downloader = DataDownloader()
    file_path  = downloader.download()

    if not file_path:
        print("❌ Download failed. Cannot continue.")
        return

    # ── STEP 2: Preprocess ────────────────────────────────────────────────────
    print("\n[STEP 2/4] Preprocess")
    preprocessor = Preprocessor()
    records      = preprocessor.process(file_path)

    if not records:
        print("❌ Preprocessing produced no records. Cannot continue.")
        return

    # ── STEP 3: Chunk ─────────────────────────────────────────────────────────
    print("\n[STEP 3/4] Chunk")
    chunker = Chunker()
    chunks  = chunker.chunk_records(records)

    if not chunks:
        print("❌ Chunking produced no chunks. Cannot continue.")
        return

    # ── STEP 4: Embed + Upload ────────────────────────────────────────────────
    print("\n[STEP 4/4] Embed and Upload to Pinecone")
    embedder = Embedder()
    total    = embedder.embed_and_upload(chunks)

    # ── Final Summary ─────────────────────────────────────────────────────────
    bridge_count = sum(1 for c in chunks if c.get("is_bridge"))
    easy   = sum(1 for r in records if r.get("level") == "easy")
    medium = sum(1 for r in records if r.get("level") == "medium")
    hard   = sum(1 for r in records if r.get("level") == "hard")

    print("\n" + "═" * 55)
    print("  ✅  INGESTION COMPLETE")
    print("═" * 55)
    print(f"  Records preprocessed  : {len(records):,}")
    print(f"  Chunks created        : {len(chunks):,}")
    print(f"    ↳ Bridge chunks     : {bridge_count:,}")
    print(f"  Difficulty breakdown  : easy={easy} | medium={medium} | hard={hard}")
    print(f"  Vectors in Pinecone   : {total:,}")
    print(f"  Index                 : {config.PINECONE_INDEX}")
    print(f"  Namespace             : {config.NAMESPACE_HOTPOT}")
    print(f"\n  Ingestion done! Retrieval team can now use:")
    print(f"    embedder.embed_query(query) → query vector")
    print(f"    embedder.get_index()        → Pinecone index")


# ═════════════════════════════════════════════════════════════════════════════
# QUERY PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_query(query: str, verbose: bool = True) -> dict:
    """
    Retrieves relevant chunks for a query and (optionally) generates an answer.

    ════════════════════════════════════════════════════════════════
    INTEGRATION POINT — WHERE INGESTION MEETS RETRIEVAL

    Your ingestion module provides:
      embedder.embed_query(query)  → 384-dim query vector
      embedder.get_index()         → live Pinecone index

    Retrieval team uses these two methods to build their Retriever.
    They do NOT need to touch DataDownloader, Preprocessor, or Chunker.

    TO PLUG IN RETRIEVAL TEAM'S CODE:
      Replace the "Basic retrieval" block below with:
        from retrieval.retriever import Retriever
        retriever = Retriever()
        chunks    = retriever.search(query)
    ════════════════════════════════════════════════════════════════

    Args:
        query:   user's natural language question (plain text)
        verbose: if True, prints step-by-step output

    Returns:
        dict with query, retrieved_chunks, and answer
    """
    if verbose:
        print(f"\n{'═'*55}")
        print(f"  Query: {query}")
        print(f"{'═'*55}")

    # ── Instantiate Embedder (the bridge between ingestion and retrieval) ─────
    embedder = Embedder()

    # ────────────────────────────────────────────────────────────────────────
    # RETRIEVAL — currently uses basic Pinecone query
    # TEAMMATES: Replace this entire block with your Retriever class
    # ────────────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n🔍 Retrieving top-{config.TOP_K} relevant chunks...")

    # Step 1: Embed the query using the same model as ingestion
    query_vector = embedder.embed_query(query)

    # Step 2: Search Pinecone
    index  = embedder.get_index()
    result = index.query(
        vector          = query_vector,
        top_k           = config.TOP_K,
        namespace       = config.NAMESPACE_HOTPOT,
        include_metadata= True,
    )

    # Step 3: Format results
    retrieved_chunks = [
        {
            "score":    round(match["score"], 4),
            "text":     match["metadata"].get("original_text", ""),
            "title":    match["metadata"].get("title", ""),
            "question": match["metadata"].get("question", ""),
            "answer":   match["metadata"].get("answer", ""),
            "is_bridge":match["metadata"].get("is_bridge", False),
        }
        for match in result.get("matches", [])
    ]

    if verbose:
        print(f"  Found {len(retrieved_chunks)} chunks:")
        for i, c in enumerate(retrieved_chunks, 1):
            print(f"  [{i}] score={c['score']:.4f} | "
                  f"bridge={c['is_bridge']} | {c['title'][:50]}")
    # ────────────────────────────────────────────────────────────────────────
    # END RETRIEVAL BLOCK
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # GENERATION — currently shows top retrieved chunk as answer
    # TEAMMATES: Replace this block with your Generator class
    # ────────────────────────────────────────────────────────────────────────
    if retrieved_chunks:
        answer = retrieved_chunks[0]["text"][:300]
    else:
        answer = "No relevant context found."

    if verbose:
        print(f"\n  {'─'*53}")
        print(f"  Answer: {answer}")
        print(f"  {'─'*53}")
    # ────────────────────────────────────────────────────────────────────────
    # END GENERATION BLOCK
    # ────────────────────────────────────────────────────────────────────────

    return {
        "query":            query,
        "retrieved_chunks": retrieved_chunks,
        "answer":           answer,
    }


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation(num_questions: int = 10):
    """
    Evaluates system accuracy on HotpotQA questions.

    METRIC: Exact Match (EM)
      EM = 1 if the gold answer appears in the retrieved text
      EM = 0 otherwise
      Final score = (correct / total) × 100%

    For a baseline RAG system, EM of 20-35% is normal.
    Improvements come from better embeddings, re-ranking, or larger LLMs.
    """
    print(f"\n{'═'*55}")
    print(f"  EVALUATION — {num_questions} HotpotQA questions")
    print(f"{'═'*55}")

    if not os.path.exists(config.HOTPOTQA_FILE):
        print("❌ HotpotQA file not found. Run --ingest first.")
        return

    with open(config.HOTPOTQA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)[:num_questions]

    correct = 0
    results = []

    for i, entry in enumerate(data, 1):
        question    = entry.get("question", "")
        gold_answer = entry.get("answer", "").lower().strip()
        q_type      = entry.get("type", "")
        level       = entry.get("level", "")

        if not question:
            continue

        output = run_query(question, verbose=False)
        pred   = output["answer"].lower()

        is_correct = gold_answer in pred
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"  [{i:2d}] {status} [{level}/{q_type}]")
        print(f"        Q: {question[:55]}")
        print(f"        Gold: {gold_answer[:35]} | Pred: {pred[:35]}")
        results.append(is_correct)

    total = len(results)
    em    = correct / total * 100 if total > 0 else 0

    print(f"\n{'─'*55}")
    print(f"  Exact Match Score: {em:.1f}% ({correct}/{total})")
    print(f"{'─'*55}")
    print("  Baseline 20-35% is expected. To improve:")
    print("  - Use a larger embedding model (e5-large)")
    print("  - Add re-ranking (cross-encoder)")
    print("  - Use a better LLM (Mistral-7B, GPT-3.5)")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAG Question Answering — HotpotQA Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run full ingestion: download → preprocess → chunk → embed → upload"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question, e.g.:\n"
             "--query \"Who directed the film that ...\""
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate system on HotpotQA questions"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of questions for --evaluate (default: 10)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive question-answering mode"
    )

    args = parser.parse_args()

    if args.ingest:
        run_ingestion()

    elif args.query:
        run_query(args.query)

    elif args.evaluate:
        run_evaluation(args.num)

    elif args.interactive:
        print("\n" + "═" * 55)
        print("  Interactive Mode — type 'quit' to exit")
        print("═" * 55)
        while True:
            try:
                question = input("\n  Your question: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    print("  Goodbye!")
                    break
                if question:
                    run_query(question)
            except KeyboardInterrupt:
                print("\n  Goodbye!")
                break

    else:
        # Show help when run with no arguments
        print("\n" + "═" * 55)
        print("  RAG Question Answering — HotpotQA Pipeline")
        print("═" * 55)
        print("\n  Available commands:")
        print("  ─────────────────────────────────────────")
        print("  python src/main.py --ingest")
        print("    → Downloads HotpotQA, preprocesses, chunks,")
        print("      embeds, and uploads to Pinecone")
        print()
        print("  python src/main.py --query \"Your question here\"")
        print("    → Retrieves relevant chunks for your question")
        print()
        print("  python src/main.py --evaluate --num 20")
        print("    → Evaluates system on 20 HotpotQA questions")
        print()
        print("  python src/main.py --interactive")
        print("    → Ask questions interactively")
        print()
        print("  First time? Start with:")
        print("  python src/main.py --ingest")


if __name__ == "__main__":
    main()
