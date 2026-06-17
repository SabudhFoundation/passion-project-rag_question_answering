"""
src/pipelines/ingestion.py
==========================
IngestionPipeline — orchestrates the full data ingestion process.

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

    Step 4: Embedder.embed_texts(chunks)
            → Converts chunks to 384-dim vectors
            → Returns: chunks with 'embedding' field

    Step 5: VectorStoreManager.upsert_chunks(chunks)
            → Uploads vectors + metadata to Pinecone
            → Returns: count of vectors uploaded

USAGE:
    pipeline = IngestionPipeline()
    pipeline.run()
"""

import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from exceptions import RAGPipelineError
from preprocessing_data.pre_processing import DataDownloader, Preprocessor, Chunker
from feature_engineering.build_features import Embedder, VectorStoreManager

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full 5-step ingestion pipeline.

    Connects: DataDownloader → Preprocessor → Chunker → Embedder → VectorStoreManager

    Each class does its job and passes output to the next.
    This class does NOT contain domain logic — it only orchestrates.

    USAGE:
        pipeline = IngestionPipeline()
        pipeline.run()
    """

    def __init__(self) -> None:
        """Initializes all pipeline components."""
        self._downloader = DataDownloader()
        self._preprocessor = Preprocessor()
        self._chunker = Chunker()
        self._embedder = Embedder()
        self._vector_store = VectorStoreManager()
        logger.info("IngestionPipeline initialised — all components ready")

    def __repr__(self) -> str:
        return "IngestionPipeline(steps=5)"

    def run(self) -> None:
        """
        Executes the full ingestion pipeline.

        Raises:
            RAGPipelineError — if any pipeline stage fails.
        """
        logger.info("=" * 55)
        logger.info("  RAG INGESTION PIPELINE")
        logger.info("  Dataset: HotpotQA only")
        logger.info("=" * 55)

        try:
            # ── STEP 1: Download ──────────────────────────────────────────
            logger.info("[STEP 1/5] Download")
            file_path = self._downloader.download()

            # ── STEP 2: Preprocess ────────────────────────────────────────
            logger.info("[STEP 2/5] Preprocess")
            records = self._preprocessor.process(file_path)

            if not records:
                logger.error("Preprocessing produced no records. Cannot continue.")
                return

            # ── STEP 3: Chunk ─────────────────────────────────────────────
            logger.info("[STEP 3/5] Chunk")
            chunks = self._chunker.chunk_records(records)

            if not chunks:
                logger.error("Chunking produced no chunks. Cannot continue.")
                return

            # ── STEP 4: Embed ─────────────────────────────────────────────
            logger.info("[STEP 4/5] Embed")
            chunks = self._embedder.embed_texts(chunks)

            # ── STEP 5: Upload to Pinecone ────────────────────────────────
            logger.info("[STEP 5/5] Upload to Pinecone")
            total = self._vector_store.upsert_chunks(chunks)

            # ── Final Summary ─────────────────────────────────────────────
            self._print_summary(records, chunks, total)

        except RAGPipelineError as e:
            logger.error("Pipeline failed: %s", e)
            raise

    def _print_summary(self, records: list, chunks: list, total: int) -> None:
        """Prints the final ingestion summary."""
        bridge_count = sum(1 for c in chunks if c.get("is_bridge"))
        easy   = sum(1 for r in records if r.get("level") == "easy")
        medium = sum(1 for r in records if r.get("level") == "medium")
        hard   = sum(1 for r in records if r.get("level") == "hard")

        logger.info("=" * 55)
        logger.info("  INGESTION COMPLETE")
        logger.info("=" * 55)
        logger.info("  Records preprocessed  : %d", len(records))
        logger.info("  Chunks created        : %d", len(chunks))
        logger.info("    Bridge chunks       : %d", bridge_count)
        logger.info(
            "  Difficulty breakdown  : easy=%d | medium=%d | hard=%d",
            easy, medium, hard,
        )
        logger.info("  Vectors in Pinecone   : %d", total)
        logger.info("  Index                 : %s", config.PINECONE_INDEX)
        logger.info("  Namespace             : %s", config.NAMESPACE_HOTPOT)
