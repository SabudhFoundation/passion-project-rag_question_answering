"""
src/models/train_model.py
=========================
LangChain-based Vector Store wrapper for retrieval.

WHY THIS FILE EXISTS:
  Our existing VectorStoreManager (in build_features.py) uses the raw
  Pinecone client for maximum control during ingestion (progress bars,
  batch upsert, stable IDs).

  But the HybridRetriever needs a LangChain-compatible vector store
  that supports `.as_retriever()` for ensemble search.

  This class bridges that gap: it wraps PineconeVectorStore from
  LangChain and provides the `.as_retriever()` interface.

DATA FLOW:
    LangChain Documents (from Chunker.to_langchain_documents())
        ↓
    LangChainVectorStore.create_from_documents(documents)
        ↓ creates PineconeVectorStore with embeddings
    LangChainVectorStore.as_retriever()
        ↓ returns a LangChain retriever for hybrid search

USAGE:
    from models.train_model import LangChainVectorStore
    store = LangChainVectorStore()
    store.create_from_documents(documents)
    retriever = store.as_retriever()
"""

import os
import sys
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ── Project imports ───────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from exceptions import VectorStoreError

logger = get_logger(__name__)


class LangChainVectorStore:
    """
    LangChain-compatible wrapper around PineconeVectorStore.

    SINGLE RESPONSIBILITY: Provide a LangChain retriever interface
    on top of our existing Pinecone index.

    WHY NOT JUST USE VectorStoreManager?
      VectorStoreManager (in build_features.py) is optimized for
      INGESTION: batch upsert, stable IDs, progress bars.
      This class is optimized for RETRIEVAL: it wraps the same
      Pinecone index with LangChain's retriever API, enabling
      `.as_retriever()` which the HybridRetriever needs.

    FEATURES:
      - Uses HuggingFaceEmbeddings (same model as ingestion)
      - Connects to existing Pinecone index (created during ingestion)
      - Provides MMR (Maximal Marginal Relevance) search
      - Can also create a new index from LangChain Documents

    USAGE:
        store = LangChainVectorStore()
        store.connect_existing()
        retriever = store.as_retriever()
    """

    def __init__(
        self,
        index_name: str = None,
        model_name: str = None,
    ) -> None:
        """
        Initializes the LangChain vector store wrapper.

        Args:
            index_name: Pinecone index name (default: config.PINECONE_INDEX).
            model_name: HuggingFace embedding model (default: config.EMBEDDING_MODEL).
        """
        self._index_name = index_name or config.PINECONE_INDEX
        self._model_name = model_name or config.EMBEDDING_MODEL
        self._vectorstore: Optional[PineconeVectorStore] = None
        self._embedding = None

        logger.info("LangChainVectorStore initialised")
        logger.info("  Index: %s | Model: %s", self._index_name, self._model_name)

    def __repr__(self) -> str:
        connected = self._vectorstore is not None
        return (
            f"LangChainVectorStore(index='{self._index_name}', "
            f"connected={connected})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def connect_existing(self) -> "LangChainVectorStore":
        """
        Connects to an existing Pinecone index (created during ingestion).

        This is the primary method for retrieval — it wraps the
        existing index with LangChain's PineconeVectorStore.

        Returns:
            self (for method chaining).

        Raises:
            VectorStoreError — if connection fails.
        """
        try:
            self._validate_api_key()
            embedding = self._get_embedding()

            self._ensure_index_exists()

            self._vectorstore = PineconeVectorStore(
                index_name=self._index_name,
                embedding=embedding,
                namespace=config.NAMESPACE_HOTPOT,
                text_key="original_text",
            )
            logger.info("  Connected to existing Pinecone index via LangChain")
            return self

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Failed to connect LangChain to Pinecone: {e}"
            ) from e

    def create_from_documents(self, documents: list) -> "LangChainVectorStore":
        """
        Creates a PineconeVectorStore from LangChain Documents.

        This handles embedding + indexing in one step via LangChain.
        Use this as an ALTERNATIVE to the manual Embedder + VectorStoreManager path.

        Args:
            documents: list of LangChain Document objects.

        Returns:
            self (for method chaining).

        Raises:
            VectorStoreError — if creation fails.
        """
        if not documents:
            raise VectorStoreError("No documents provided for vector store creation.")

        try:
            self._validate_api_key()
            embedding = self._get_embedding()

            self._ensure_index_exists()

            logger.info("Creating PineconeVectorStore from %d documents...", len(documents))
            self._vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embedding,
                index_name=self._index_name,
                namespace=config.NAMESPACE_HOTPOT,
                text_key="original_text",
            )
            logger.info("  Vector store created successfully")
            return self

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Failed to create vector store from documents: {e}"
            ) from e

    def as_retriever(self, **kwargs):
        """
        Returns a LangChain retriever from the vector store.

        Uses MMR (Maximal Marginal Relevance) by default to
        reduce redundancy in retrieved chunks.

        Returns:
            LangChain retriever object.

        Raises:
            VectorStoreError — if vector store is not connected.
        """
        if self._vectorstore is None:
            raise VectorStoreError(
                "Vector store not connected. "
                "Call connect_existing() or create_from_documents() first."
            )

        search_kwargs = kwargs.get("search_kwargs", {
            "k":           config.TOP_K,
            "fetch_k":     config.VECTOR_FETCH_K,
            "lambda_mult": config.MMR_LAMBDA,
        })

        return self._vectorstore.as_retriever(
            search_type=kwargs.get("search_type", "mmr"),
            search_kwargs=search_kwargs,
        )

    def get_vectorstore(self) -> PineconeVectorStore:
        """
        Returns the underlying PineconeVectorStore object.

        Raises:
            VectorStoreError — if not connected.
        """
        if self._vectorstore is None:
            raise VectorStoreError("Vector store not connected.")
        return self._vectorstore

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_embedding(self) -> HuggingFaceEmbeddings:
        """Lazy-loads the HuggingFace embedding model."""
        if self._embedding is None:
            try:
                logger.info("  Loading HuggingFace embedding: %s", self._model_name)
                self._embedding = HuggingFaceEmbeddings(
                    model_name=self._model_name,
                )
                logger.info("  Embedding model loaded")
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to load HuggingFace embedding: {e}"
                ) from e
        return self._embedding

    def _ensure_index_exists(self) -> None:
        """Creates the Pinecone index if it doesn't exist yet."""
        try:
            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            existing = [idx.name for idx in pc.list_indexes()]

            if self._index_name not in existing:
                logger.info("  Creating Pinecone index '%s'...", self._index_name)
                pc.create_index(
                    name=self._index_name,
                    dimension=config.EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=config.PINECONE_REGION,
                    ),
                )
                logger.info("  Index created")
            else:
                logger.info("  Index '%s' already exists", self._index_name)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to ensure index exists: {e}"
            ) from e

    @staticmethod
    def _validate_api_key() -> None:
        """Validates that the Pinecone API key is set."""
        if (
            not config.PINECONE_API_KEY
            or config.PINECONE_API_KEY == "PASTE_YOUR_KEY_HERE"
        ):
            raise VectorStoreError(
                "Pinecone API key not set!\n"
                "Run this in your terminal first:\n"
                "  set PINECONE_API_KEY=your-actual-key-here"
            )
