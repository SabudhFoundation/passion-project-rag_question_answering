"""
src/main.py
===========
THE MAIN ENTRY POINT — thin CLI that delegates to pipeline classes.

This file contains ZERO domain logic. It only:
  1. Parses command-line arguments
  2. Instantiates the correct pipeline class
  3. Calls pipeline.run()

All actual logic lives in the pipeline classes:
  - pipelines/ingestion.py   → IngestionPipeline
  - pipelines/query.py       → QueryPipeline
  - pipelines/evaluation.py  → EvaluationPipeline

HOW TO RUN:

  Full ingestion (run once to populate Pinecone):
    python src/main.py --ingest

  Ask a question (after ingestion is done):
    python src/main.py --query "Who directed the film that starred Leonardo DiCaprio?"

  Evaluate on HotpotQA questions:
    python src/main.py --evaluate --num 10

  Interactive mode (keep asking questions):
    python src/main.py --interactive
    
  Launch Chainlit Dashboard:
    python src/main.py --app
"""

import os
import sys
import argparse

# ── Set up Python path so imports work correctly ──────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(SRC_DIR))

from logger import get_logger
from exceptions import RAGPipelineError

logger = get_logger(__name__)


def main() -> None:
    """Parses CLI arguments and delegates to the appropriate pipeline."""

    parser = argparse.ArgumentParser(
        description="RAG Question Answering — HotpotQA Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run full ingestion: download -> preprocess -> chunk -> embed -> upload",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help='Ask a single question, e.g.:\n'
             '--query "Who directed the film that ..."',
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate system on HotpotQA questions",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of questions for --evaluate (default: 10)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive question-answering mode",
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Launch the Chainlit Web application dashboard",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run end-to-end: Full Ingestion -> Test Query -> Launch App",
    )

    args = parser.parse_args()

    # ── Route to the correct pipeline ─────────────────────────────────────────

    if args.all:
        from pipelines.ingestion import IngestionPipeline
        from pipelines.query import QueryPipeline
        import subprocess
        try:
            logger.info("Starting end-to-end pipeline execution...")
            # 1. Ingestion
            logger.info("=== Phase 1: Ingestion ===")
            ingest_pipeline = IngestionPipeline()
            ingest_pipeline.run()
            
            # 2. Test Query (exercises Retriever + Generator)
            logger.info("=== Phase 2: Test Query (Retriever & Generator) ===")
            query_pipeline = QueryPipeline()
            query_pipeline.run("What is the main topic of the ingested dataset?")
            
            # 3. Launch Chainlit App for UI
            logger.info("=== Phase 3: Launching Chainlit App ===")
            app_path = os.path.join(SRC_DIR, "visualization", "app.py")
            subprocess.run(
                [sys.executable, "-m", "chainlit", "run", app_path,
                 "--host", "0.0.0.0", "--port", "8000"],
                cwd=os.path.dirname(SRC_DIR),
            )
        except RAGPipelineError as e:
            logger.error("End-to-end pipeline failed: %s", e)
            sys.exit(1)

    elif args.ingest:
        from pipelines.ingestion import IngestionPipeline
        try:
            pipeline = IngestionPipeline()
            pipeline.run()
        except RAGPipelineError as e:
            logger.error("Ingestion failed: %s", e)
            sys.exit(1)

    elif args.query:
        from pipelines.query import QueryPipeline
        pipeline = QueryPipeline()
        pipeline.run(args.query)

    elif args.evaluate:
        from pipelines.evaluation import EvaluationPipeline
        pipeline = EvaluationPipeline()
        pipeline.run(num_questions=args.num)

    elif args.interactive:
        from pipelines.query import QueryPipeline
        pipeline = QueryPipeline()

        logger.info("=" * 55)
        logger.info("  Interactive Mode — type 'quit' to exit")
        logger.info("=" * 55)
        while True:
            try:
                question = input("\n  Your question: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    logger.info("  Goodbye!")
                    break
                if question:
                    pipeline.run(question)
            except KeyboardInterrupt:
                logger.info("\n  Goodbye!")
                break
                
    elif args.app:
        import subprocess
        logger.info("Launching Chainlit dashboard...")
        app_path = os.path.join(SRC_DIR, "visualization", "app.py")
        subprocess.run(
            [sys.executable, "-m", "chainlit", "run", app_path,
             "--host", "0.0.0.0", "--port", "8000"],
            cwd=os.path.dirname(SRC_DIR),
        )

    else:
        _print_usage()


def _print_usage() -> None:
    """Prints the available commands when no arguments are provided."""
    logger.info("=" * 55)
    logger.info("  RAG Question Answering — HotpotQA Pipeline")
    logger.info("=" * 55)
    logger.info("")
    logger.info("  Available commands:")
    logger.info("  python src/main.py --ingest")
    logger.info("  python src/main.py --query \"Your question here\"")
    logger.info("  python src/main.py --evaluate --num 20")
    logger.info("  python src/main.py --interactive")
    logger.info("  python src/main.py --app")
    logger.info("  python src/main.py --all")
    logger.info("")
    logger.info("  First time? Start with: python src/main.py --all")


if __name__ == "__main__":
    main()
