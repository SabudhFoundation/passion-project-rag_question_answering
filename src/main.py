"""
src/main.py
============
Main pipeline orchestrator.

Calls all 4 classes in correct order:
  DataDownloader → Preprocessor → Chunker → Embedder


COMMANDS:
  python src/main.py --ingest
  python src/main.py --query "Who directed Titanic?"
  python src/main.py --evaluate --num 10
  python src/main.py --interactive
"""

import os
import sys
import json
import argparse

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(SRC_DIR))

import config
from utils import get_logger, print_section, print_summary, format_retrieval_result

from preprocessing_data.data_downloader import DataDownloader
from preprocessing_data.preprocessor    import Preprocessor
from feature_engineering.chunker        import Chunker
from feature_engineering.embedder       import Embedder

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# INGESTION
# ═════════════════════════════════════════════════════════════════════════════

def run_ingestion() -> None:
    """
    Runs the full 4-step ingestion pipeline with exception handling.
    Each step raises a specific exception if it fails — caught here
    with a clear message telling you exactly what to fix.
    """
    print_section("RAG INGESTION PIPELINE — HotpotQA")

    try:
        # Step 1: Download
        downloader = DataDownloader()
        file_path  = downloader.download()
        if not file_path:
            raise RuntimeError(
                "Download returned no file. Check network and disk space."
            )

        # Step 2: Preprocess
        preprocessor = Preprocessor()
        records      = preprocessor.process(file_path)
        if not records:
            raise RuntimeError(
                "Preprocessing returned 0 records. Check HotpotQA file."
            )

        # Step 3: Chunk
        chunker = Chunker()
        chunks  = chunker.chunk_records(records)
        if not chunks:
            raise RuntimeError(
                "Chunking returned 0 chunks. Check preprocessed records."
            )

        # Step 4: Embed + Upload
        embedder = Embedder()
        total    = embedder.embed_and_upload(chunks)

        bridge = sum(1 for c in chunks if c.get("is_bridge"))
        print_summary("✅ INGESTION COMPLETE", {
            "Records preprocessed":  len(records),
            "Chunks created":        len(chunks),
            "Bridge chunks":         bridge,
            "Vectors in Pinecone":   total,
            "Index":                 config.PINECONE_INDEX,
            "Namespace":             config.NAMESPACE_HOTPOT,
        })

    except (OSError, FileNotFoundError) as e:
        logger.error(f"File system error:\n{e}")
    except (ValueError, ImportError) as e:
        logger.error(f"Configuration or dependency error:\n{e}")
    except RuntimeError as e:
        logger.error(f"Pipeline error:\n{e}")
    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user (Ctrl+C)")


# ═════════════════════════════════════════════════════════════════════════════
# QUERY
# ═════════════════════════════════════════════════════════════════════════════

def run_query(query: str, verbose: bool = True) -> dict:
    """
    Retrieves relevant chunks for a user question.

    ════════════════════════════════════════════════════════
    RETRIEVAL TEAM: Replace the marked block with:
      from retrieval.retriever import Retriever
      retriever = Retriever()
      chunks    = retriever.search(query)
    ════════════════════════════════════════════════════════
    """
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")

    if verbose:
        print(f"\n{'═'*55}")
        print(f"  Query: {query}")
        print(f"{'═'*55}")

    try:
        embedder = Embedder()

        # ── RETRIEVAL BLOCK (replace with teammate's Retriever) ───────────────
        query_vector = embedder.embed_query(query)
        index        = embedder.get_index()
        result       = index.query(
            vector           = query_vector,
            top_k            = config.TOP_K,
            namespace        = config.NAMESPACE_HOTPOT,
            include_metadata = True,
        )
        chunks = [
            format_retrieval_result(m)
            for m in result.get("matches", [])
        ]
        # ── END RETRIEVAL BLOCK ───────────────────────────────────────────────

        if verbose:
            logger.info(f"Retrieved {len(chunks)} chunks")
            for i, c in enumerate(chunks, 1):
                print(f"  [{i}] score={c['score']:.4f} | "
                      f"bridge={c['is_bridge']} | {c['title'][:50]}")
            if chunks:
                print(f"\n  Top result:\n  {chunks[0]['text'][:300]}...")

        return {"query": query, "chunks": chunks}

    except ValueError as e:
        logger.error(f"Query error: {e}")
        return {"query": query, "chunks": []}
    except RuntimeError as e:
        logger.error(f"Retrieval failed: {e}")
        return {"query": query, "chunks": []}


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation(num: int = 10) -> None:
    """Evaluates on HotpotQA questions using Exact Match."""
    print_section(f"EVALUATION — {num} HotpotQA questions")

    if not os.path.exists(config.HOTPOTQA_FILE):
        logger.error(
            f"HotpotQA file not found: {config.HOTPOTQA_FILE}\n"
            f"Run --ingest first."
        )
        return

    try:
        with open(config.HOTPOTQA_FILE) as f:
            data = json.load(f)[:num]
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Could not load HotpotQA file: {e}")
        return

    correct = 0
    for i, entry in enumerate(data, 1):
        q    = entry.get("question", "")
        gold = entry.get("answer", "").lower().strip()
        if not q:
            continue
        try:
            out  = run_query(q, verbose=False)
            pred = out["chunks"][0]["text"][:300].lower() \
                   if out["chunks"] else ""
            ok   = gold in pred
            if ok:
                correct += 1
            print(f"  [{i:2d}] {'✅' if ok else '❌'} Q: {q[:50]}")
            print(f"        Gold: {gold[:30]}")
        except Exception as e:
            logger.warning(f"Evaluation failed for question {i}: {e}")

    em = correct / len(data) * 100 if data else 0
    print_summary("Evaluation Results", {
        "Exact Match": f"{em:.1f}%",
        "Correct":     correct,
        "Total":       len(data),
    })
    print("  Baseline 20-35% EM is expected at this stage.")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — HotpotQA"
    )
    parser.add_argument("--ingest",      action="store_true",
                        help="Run full ingestion pipeline")
    parser.add_argument("--query",       type=str, default=None,
                        help="Ask a question")
    parser.add_argument("--evaluate",    action="store_true",
                        help="Evaluate on HotpotQA")
    parser.add_argument("--num",         type=int, default=10,
                        help="Questions for --evaluate")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive Q&A mode")
    args = parser.parse_args()

    if args.ingest:
        run_ingestion()
    elif args.query:
        try:
            run_query(args.query)
        except ValueError as e:
            logger.error(e)
    elif args.evaluate:
        run_evaluation(args.num)
    elif args.interactive:
        print("\nInteractive mode — type 'quit' to exit")
        while True:
            try:
                q = input("\nQuestion: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if q:
                    run_query(q)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        print("\n  Commands:")
        print("  python src/main.py --ingest")
        print("  python src/main.py --query \"Your question\"")
        print("  python src/main.py --evaluate --num 10")
        print("  python src/main.py --interactive")
        print("\n  First time? Run: python src/main.py --ingest")


if __name__ == "__main__":
    main()
