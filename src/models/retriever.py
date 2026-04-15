"""
src/models/retriever.py
=======================
HybridRetriever — combines BM25 keyword search with Vector semantic search.

WHY HYBRID RETRIEVAL?
  Pure vector search (our previous approach) embeds the question and
  finds semantically similar chunks. But it can miss:
    - Exact names ("Christopher Nolan")
    - Specific IDs or dates ("1997")

  BM25 is a keyword-based algorithm that catches EXACT term matches.
  By combining both, we get the best of both worlds:
    - BM25 catches specific names/dates that embeddings miss
    - Vector (MMR) catches semantic meaning that keywords miss

WHAT IS MMR (Maximal Marginal Relevance)?
  Standard vector search might return 5 chunks that all say the same thing.
  MMR penalizes redundancy: after selecting the best match, it picks
  the next chunk that is BOTH relevant AND different from what was
  already selected. This gives more diverse, useful context.

WHAT IS RECIPROCAL RANK FUSION (RRF)?
  Each retriever returns ranked results. RRF merges them by scoring:
    score(doc) = sum( weight / (rank + k) ) across all retrievers
  where k=60 is a smoothing constant.
  This produces a single ranked list that balances both retrievers.

DATA FLOW:
    User question (string)
        |
    HybridRetriever.retrieve(question)
        |-- BM25Retriever.invoke(question)     -> keyword matches
        |-- VectorRetriever.invoke(question)   -> semantic matches
        | merged via Reciprocal Rank Fusion
    List of LangChain Documents (ranked)

USAGE:
    from models.retriever import HybridRetriever
    retriever = HybridRetriever(documents, langchain_store)
    results   = retriever.retrieve("Who directed Titanic?")
"""

import os
import sys
from typing import List, Dict, Any, Tuple

from langchain_community.retrievers import BM25Retriever

# ── Project imports ───────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from exceptions import RetrievalError

logger = get_logger(__name__)


class HybridRetriever:
    """
    Combines BM25 (keyword) and Vector (semantic) retrieval.

    SINGLE RESPONSIBILITY: Search for relevant chunks. Nothing else.

    COMPONENTS:
      1. BM25Retriever   -- keyword-based search (catches exact names/IDs)
      2. VectorRetriever -- MMR-based semantic search (catches meaning)
      3. Reciprocal Rank Fusion -- merges both with configurable weights

    NOTE: We implement RRF manually instead of using LangChain's
    EnsembleRetriever, which was removed in LangChain v1.2+.

    USAGE:
        retriever = HybridRetriever(documents, langchain_store)
        results   = retriever.retrieve("Who directed Titanic?")
    """

    def __init__(
        self,
        documents: list,
        langchain_store,
        bm25_k: int = None,
        ensemble_weights: list = None,
    ) -> None:
        """
        Initializes the hybrid retriever.

        Args:
            documents:        LangChain Document objects (for BM25 index).
            langchain_store:  LangChainVectorStore instance (for vector search).
            bm25_k:           top-K for BM25 (default: config.BM25_K).
            ensemble_weights: [BM25_weight, Vector_weight] (default: config.ENSEMBLE_WEIGHTS).

        Raises:
            RetrievalError -- if initialization fails.
        """
        if not documents:
            raise RetrievalError("No documents provided for BM25 index.")

        self._documents = documents
        self._langchain_store = langchain_store
        self._bm25_k = bm25_k or config.BM25_K
        self._ensemble_weights = ensemble_weights or config.ENSEMBLE_WEIGHTS

        self._bm25_retriever = None
        self._vector_retriever = None

        try:
            self._build()
            logger.info("HybridRetriever initialised")
            logger.info(
                "  BM25 k=%d | Weights=%s | Documents=%d",
                self._bm25_k, self._ensemble_weights, len(documents),
            )
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(
                f"Failed to initialize HybridRetriever: {e}"
            ) from e

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(bm25_k={self._bm25_k}, "
            f"weights={self._ensemble_weights}, "
            f"docs={len(self._documents)})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves relevant chunks using hybrid BM25 + Vector search.

        Args:
            query: user's natural language question.

        Returns:
            list of dicts with text, title, score, metadata.

        Raises:
            RetrievalError -- if retrieval fails.
        """
        if not query or not isinstance(query, str):
            raise RetrievalError(
                f"Query must be a non-empty string, got: {type(query).__name__}"
            )

        try:
            logger.info("Hybrid retrieval for: '%s'", query[:60])

            # Run both retrievers
            bm25_results = self._bm25_retriever.invoke(query)
            vector_results = self._vector_retriever.invoke(query)

            logger.info(
                "  BM25: %d results | Vector: %d results",
                len(bm25_results), len(vector_results),
            )

            # Merge via Reciprocal Rank Fusion
            merged = self._reciprocal_rank_fusion(
                results_list=[bm25_results, vector_results],
                weights=self._ensemble_weights,
            )

            # Convert LangChain Documents to our standard dict format
            retrieved_chunks = []
            for rank, (doc, score) in enumerate(merged, 1):
                retrieved_chunks.append({
                    "text":        doc.page_content,
                    "title":       doc.metadata.get("title", ""),
                    "source":      doc.metadata.get("source", ""),
                    "doc_id":      doc.metadata.get("doc_id", ""),
                    "question":    doc.metadata.get("question", ""),
                    "answer":      doc.metadata.get("answer", ""),
                    "is_bridge":   doc.metadata.get("is_bridge", False),
                    "is_multihop": doc.metadata.get("is_multihop", False),
                    "score":       round(score, 4),
                    "rank":        rank,
                })

            logger.info("  Merged: %d unique chunks", len(retrieved_chunks))
            return retrieved_chunks

        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(
                f"Hybrid retrieval failed: {e}"
            ) from e

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        """Builds the BM25 and Vector retrievers."""
        self._bm25_retriever = self._build_bm25()
        self._vector_retriever = self._build_vector()
        logger.info("  Both retrievers built (BM25 + Vector)")

    def _build_bm25(self) -> BM25Retriever:
        """
        Builds the BM25 keyword retriever from documents.

        BM25 (Best Matching 25) is a ranking function that scores
        documents based on term frequency. It excels at finding
        exact keyword matches that embedding models might miss.
        """
        try:
            bm25 = BM25Retriever.from_documents(self._documents)
            bm25.k = self._bm25_k
            logger.info("  BM25 retriever built (k=%d)", self._bm25_k)
            return bm25
        except Exception as e:
            raise RetrievalError(
                f"Failed to build BM25 retriever: {e}"
            ) from e

    def _build_vector(self):
        """
        Builds the vector retriever with MMR from the LangChain store.

        MMR (Maximal Marginal Relevance) selects chunks that are
        BOTH relevant to the query AND diverse from each other.
        This prevents returning 5 chunks that all say the same thing.
        """
        try:
            retriever = self._langchain_store.as_retriever()
            logger.info("  Vector retriever built (MMR, top_k=%d)", config.TOP_K)
            return retriever
        except Exception as e:
            raise RetrievalError(
                f"Failed to build vector retriever: {e}"
            ) from e

    @staticmethod
    def _reciprocal_rank_fusion(
        results_list: List[list],
        weights: List[float],
        k: int = 60,
        top_n: int = None,
    ) -> List[Tuple]:
        """
        Merges multiple ranked result lists using Reciprocal Rank Fusion.

        FORMULA:
          score(doc) = sum( weight_i / (rank_i + k) ) for each retriever i

        Args:
            results_list: list of result lists (one per retriever).
            weights:      weight for each retriever.
            k:            smoothing constant (default: 60).
            top_n:        max results to return (default: config.TOP_K).

        Returns:
            list of (Document, score) tuples, sorted by score descending.
        """
        if top_n is None:
            top_n = config.TOP_K

        # Score each document
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}

        for results, weight in zip(results_list, weights):
            for rank, doc in enumerate(results, 1):
                # Use page_content as unique key
                doc_key = doc.page_content[:200]

                if doc_key not in doc_map:
                    doc_map[doc_key] = doc
                    doc_scores[doc_key] = 0.0

                doc_scores[doc_key] += weight / (rank + k)

        # Sort by score descending
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top-N as (Document, score) tuples
        return [
            (doc_map[key], score)
            for key, score in sorted_docs[:top_n]
        ]
