"""
src/pipelines/query.py
======================
QueryPipeline — handles user question answering via hybrid retrieval.

DATA FLOW:
    User question (string)
        ↓
    Load chunks from disk → convert to LangChain Documents
        ↓
    LangChainVectorStore.connect_existing() → vector retriever
        ↓
    HybridRetriever(documents, store) → BM25 + Vector ensemble
        ↓
    HybridRetriever.retrieve(question) → ranked chunks
        ↓
    Generate answer from top chunks
        ↓
    Return dict with query, retrieved_chunks, answer

USAGE:
    pipeline = QueryPipeline()
    result   = pipeline.run("Who directed Titanic?")
"""

import os
import sys
from typing import Dict, Any, List, Optional

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from exceptions import RAGPipelineError, RetrievalError
from preprocessing_data.pre_processing import Chunker
from models.train_model import LangChainVectorStore
from models.retriever import HybridRetriever
from models.predict_model import Generator

logger = get_logger(__name__)


class QueryPipeline:
    """
    Retrieves relevant chunks using hybrid search and generates an answer.

    RETRIEVAL STRATEGY:
      Combines BM25 (keyword) + Vector (semantic) search via the
      HybridRetriever. This catches both exact entity matches and
      semantic similarities.

    COMPONENTS:
      - Chunker:              loads chunks from disk → LangChain Documents
      - LangChainVectorStore: connects to existing Pinecone index
      - HybridRetriever:      BM25 + Vector ensemble search

    USAGE:
        pipeline = QueryPipeline()
        result   = pipeline.run("Who directed Titanic?")
    """

    def __init__(self) -> None:
        """
        Initializes the query pipeline.

        Loads chunks and builds the hybrid retriever on first init.
        This takes a few seconds but only happens once.
        """
        self._retriever: Optional[HybridRetriever] = None
        self._generator = Generator()
        self._initialized = False
        logger.info("QueryPipeline initialised (retriever builds on first query)")

    def __repr__(self) -> str:
        return f"QueryPipeline(top_k={config.TOP_K}, initialized={self._initialized})"

    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Retrieves relevant chunks and generates an answer for a query.

        Args:
            query:   user's natural language question (plain text).
            verbose: if True, logs step-by-step output.

        Returns:
            dict with keys: query, retrieved_chunks, answer.
        """
        if verbose:
            logger.info("=" * 55)
            logger.info("  Query: %s", query)
            logger.info("=" * 55)

        try:
            # ── Ensure retriever is built ─────────────────────────────────
            self._ensure_initialized()

            # ── Retrieval ─────────────────────────────────────────────────
            retrieved_chunks = self._retrieve(query, verbose)

            # ── Generation ────────────────────────────────────────────────
            answer = self._generate(query, retrieved_chunks)

            if verbose:
                logger.info("  %s", "─" * 53)
                logger.info("  Answer: %s", answer)
                logger.info("  %s", "─" * 53)

            return {
                "query":            query,
                "retrieved_chunks": retrieved_chunks,
                "answer":           answer,
            }

        except RAGPipelineError as e:
            logger.error("Query failed: %s", e)
            return {
                "query":            query,
                "retrieved_chunks": [],
                "answer":           f"Error: {e}",
            }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """
        Lazily builds the hybrid retriever on first use.

        Steps:
          1. Load chunks from disk (saved during ingestion)
          2. Convert to LangChain Documents
          3. Connect to existing Pinecone index via LangChain
          4. Build HybridRetriever (BM25 + Vector)
        """
        if self._initialized:
            return

        logger.info("Building hybrid retriever (first query — one-time setup)...")

        try:
            # Step 1: Load chunks from disk
            chunker = Chunker()
            chunks = chunker.load_chunks_from_disk()

            if not chunks:
                raise RetrievalError(
                    "No chunks found on disk. Run --ingest first."
                )

            # Step 2: Convert to LangChain Documents
            documents = Chunker.to_langchain_documents(chunks)

            # Step 3: Connect to existing Pinecone index
            store = LangChainVectorStore()
            store.connect_existing()

            # Step 4: Build HybridRetriever
            self._retriever = HybridRetriever(
                documents=documents,
                langchain_store=store,
            )

            self._initialized = True
            logger.info("Hybrid retriever ready")

        except RAGPipelineError:
            raise
        except Exception as e:
            raise RetrievalError(
                f"Failed to initialize retriever: {e}"
            ) from e

    def _retrieve(self, query: str, verbose: bool) -> List[Dict[str, Any]]:
        """
        Runs hybrid retrieval (BM25 + Vector) for the given query.

        Args:
            query:   user's question text.
            verbose: if True, logs retrieval details.

        Returns:
            list of chunk dicts with score, text, title, etc.
        """
        if verbose:
            logger.info("Retrieving via hybrid search (BM25 + Vector)...")

        retrieved_chunks = self._retriever.retrieve(query)

        if verbose:
            logger.info("  Found %d chunks:", len(retrieved_chunks))
            for i, c in enumerate(retrieved_chunks, 1):
                logger.info(
                    "  [%d] score=%.4f | bridge=%s | %s",
                    i, c.get("score", 0), c.get("is_bridge", False),
                    c.get("title", "")[:50],
                )

        return retrieved_chunks

        return retrieved_chunks

    def _generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the Groq LLM via the Generator class.

        Args:
            query: user's natural language question.
            retrieved_chunks: list of chunk dicts from retrieval.

        Returns:
            str — the generated answer.
        """
        if not retrieved_chunks:
            return "No relevant context found."
            
        result = self._generator.generate(query=query, chunks=retrieved_chunks)
        return result.get("answer", "Failed to generate an answer.")

