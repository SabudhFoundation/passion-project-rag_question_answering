"""
src/feature_engineering/build_features.py
==========================================
Embedding + Vector DB module.

Contains TWO classes, each with a single responsibility:
  1. Embedder           — converts text chunks into embedding vectors
  2. VectorStoreManager — manages Pinecone connection, upsert, and queries

DATA FLOW:
    Chunker returns → list of chunk dicts
        ↓
    Embedder.embed_texts(chunks)
        ↓ returns chunks with 'embedding' field attached
    VectorStoreManager.upsert_chunks(chunks)
        ↓ uploads vectors + metadata to Pinecone

WHAT ARE "FEATURES" IN NLP/RAG?
  In traditional ML, features are numbers extracted from raw data.
  In NLP/RAG, the "features" are embedding vectors — numbers that
  represent the MEANING of each text chunk.
  Embedding + Vector DB = Feature Engineering for RAG.

DESIGN PRINCIPLES:
  - Single Responsibility: Embedder handles ONLY embedding,
    VectorStoreManager handles ONLY Pinecone operations.
  - Fail-fast: Custom exceptions with clear error messages.
  - Lazy loading: Model and index only load when first needed.
  - Logging: All output uses Python's logging module.
"""

import os
import sys
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ── Project imports ───────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from exceptions import EmbeddingError, VectorStoreError

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 1: Embedder
# ═════════════════════════════════════════════════════════════════════════════

class Embedder:
    """
    Converts text into embedding vectors using SentenceTransformer models.

    SINGLE RESPONSIBILITY: Text → Vector conversion. Nothing else.

    WHAT IS AN EMBEDDING?
      A computer cannot compare the MEANING of two sentences directly.
      An embedding model converts text into a list of 384 numbers where:
        - Similar sentences → similar numbers → close in vector space
        - Different sentences → different numbers → far in vector space

    FEATURES:
      - Lazy model loading (model downloads only on first use)
      - Batch encoding with configurable batch size
      - Normalized embeddings for cosine similarity

    INTERFACE FOR RETRIEVAL TEAM:
        embed_query(query)  → converts user question to 384-dim vector
        embed_texts(chunks) → adds 'embedding' field to each chunk dict

    USAGE:
        embedder = Embedder()
        vector   = embedder.embed_query("Who directed Titanic?")
        chunks   = embedder.embed_texts(chunks)
    """

    def __init__(
        self,
        model_name: str = None,
        batch_size: int = None,
    ) -> None:
        """
        Initializes the Embedder with configurable model and batch size.

        Args:
            model_name: HuggingFace model name (default: config.EMBEDDING_MODEL)
            batch_size: texts per forward pass (default: config.EMBED_BATCH_SIZE)
        """
        self._model_name = model_name or config.EMBEDDING_MODEL
        self._batch_size = batch_size or config.EMBED_BATCH_SIZE
        self._model: Optional[SentenceTransformer] = None

        logger.info("Embedder initialised (model loads on first use)")
        logger.info("  Model: %s", self._model_name)

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"Embedder(model='{self._model_name}', "
            f"batch_size={self._batch_size}, loaded={loaded})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> List[float]:
        """
        Converts a single user question into a 384-dim embedding vector.

        ════════════════════════════════════════════════════
        HANDOFF METHOD FOR THE RETRIEVAL TEAM.
        Teammates import Embedder and call this method.
        ════════════════════════════════════════════════════

        Args:
            query: user's natural language question (plain text).

        Returns:
            list of 384 floats — the normalized embedding vector.

        Raises:
            EmbeddingError — if query is invalid or model fails.
        """
        if not query or not isinstance(query, str):
            raise EmbeddingError(
                f"Query must be a non-empty string, got: {type(query).__name__}"
            )

        try:
            model = self._get_model()
            vector = model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vector.tolist()
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed query: {e}"
            ) from e

    def embed_texts(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates 384-dim embedding vectors for all chunk texts.

        Each chunk dict gets an 'embedding' field added containing
        a list of 384 floats.

        Args:
            chunks: list of chunk dicts (must have 'text' key).

        Returns:
            same list of chunks, each with 'embedding' field added.

        Raises:
            EmbeddingError — if encoding fails.
        """
        if not chunks:
            raise EmbeddingError("No chunks provided for embedding.")

        logger.info("Embedding %d chunks...", len(chunks))

        try:
            model = self._get_model()
            texts = [chunk["text"] for chunk in chunks]

            embeddings = model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            # Attach each embedding to its chunk
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding.tolist()

            logger.info("  Embedded %d chunks successfully", len(chunks))
            return chunks

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings: {e}"
            ) from e

    def embed_and_upload(
        self,
        chunks: List[Dict[str, Any]],
        vector_store: "VectorStoreManager" = None,
    ) -> int:
        """
        Convenience method: embed all chunks and upload to Pinecone.

        This is the main ingestion entry point that combines embedding
        (this class) with vector upload (VectorStoreManager).

        Args:
            chunks:       list of chunk dicts from Chunker.
            vector_store: optional VectorStoreManager instance
                          (creates one if not provided).

        Returns:
            int — total number of vectors successfully uploaded.

        Raises:
            EmbeddingError  — if embedding fails.
            VectorStoreError — if upload fails.
        """
        logger.info("=" * 55)
        logger.info("  Embedder: Generating vectors + uploading to Pinecone")
        logger.info("=" * 55)

        if not chunks:
            logger.warning("No chunks to embed")
            return 0

        # Step 1: Generate embedding vectors
        chunks = self.embed_texts(chunks)

        # Step 2: Upload to Pinecone via VectorStoreManager
        if vector_store is None:
            vector_store = VectorStoreManager()

        total = vector_store.upsert_chunks(chunks)

        # Step 3: Report final state
        try:
            stats = vector_store.get_stats()
            total_vectors = stats.get("total_vector_count", 0)
            logger.info("  Upload complete!")
            logger.info("    Vectors uploaded this run : %d", total)
            logger.info("    Total vectors in Pinecone : %d", total_vectors)
            logger.info("    Index name                : %s", config.PINECONE_INDEX)
            logger.info("    Namespace                 : %s", config.NAMESPACE_HOTPOT)
        except VectorStoreError:
            logger.warning("  Could not fetch index stats after upload")

        return total

    # ─────────────────────────────────────────────────────────────────────────
    # RETRIEVAL TEAM INTERFACE (backward-compatible)
    # ─────────────────────────────────────────────────────────────────────────

    def get_index(self):
        """
        Returns the live Pinecone index object.

        ════════════════════════════════════════════════════
        HANDOFF METHOD FOR THE RETRIEVAL TEAM.
        Delegates to VectorStoreManager.
        ════════════════════════════════════════════════════
        """
        vs = VectorStoreManager()
        return vs.get_index()

    def get_index_stats(self) -> dict:
        """Returns statistics about the Pinecone index."""
        vs = VectorStoreManager()
        return vs.get_stats()

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """
        Lazy-loads the SentenceTransformer model.
        Downloads ~80MB on first run, then cached locally.

        Raises:
            EmbeddingError — if model cannot be loaded.
        """
        if self._model is None:
            try:
                logger.info(
                    "Loading embedding model: %s", self._model_name
                )
                logger.info("  (First run downloads ~80 MB — this is normal)")
                self._model = SentenceTransformer(self._model_name)
                dim = self._model.get_sentence_embedding_dimension()
                logger.info(
                    "  Model loaded — outputs %d-dimensional vectors", dim
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self._model_name}': {e}"
                ) from e
        return self._model


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 2: VectorStoreManager
# ═════════════════════════════════════════════════════════════════════════════

class VectorStoreManager:
    """
    Manages all Pinecone vector database operations.

    SINGLE RESPONSIBILITY: Vector DB connection and CRUD. Nothing else.

    FEATURES:
      - Lazy index connection (only connects when first needed)
      - Automatic index creation if it doesn't exist
      - Batch upsert with error recovery (continues on batch failure)
      - API key validation before connection attempt

    INDEX CONFIGURATION:
      dimension = 384   → must match MiniLM model output
      metric    = cosine → measures angle between vectors (best for text)
      ServerlessSpec    → managed cloud index, no infrastructure needed

    USAGE:
        vs = VectorStoreManager()
        vs.upsert_chunks(chunks)         # upload during ingestion
        index = vs.get_index()           # used by retrieval team
        stats = vs.get_stats()           # check vector counts
    """

    def __init__(self) -> None:
        """Initializes the VectorStoreManager (connects lazily)."""
        self._index = None
        logger.info("VectorStoreManager initialised (connects on first use)")

    def __repr__(self) -> str:
        connected = self._index is not None
        return (
            f"VectorStoreManager(index='{config.PINECONE_INDEX}', "
            f"connected={connected})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def get_index(self):
        """
        Returns the live Pinecone index object (lazy-connected).

        ════════════════════════════════════════════════════
        HANDOFF METHOD FOR THE RETRIEVAL TEAM.
        Teammates use this to run Pinecone queries.
        ════════════════════════════════════════════════════

        Returns:
            Pinecone Index object.

        Raises:
            VectorStoreError — if connection fails.
        """
        if self._index is None:
            self._index = self._connect()
        return self._index

    def get_stats(self) -> dict:
        """
        Returns statistics about the Pinecone index (vector counts etc.).

        Raises:
            VectorStoreError — if stats query fails.
        """
        try:
            index = self.get_index()
            return index.describe_index_stats()
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Failed to get index stats: {e}"
            ) from e

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Uploads all chunks to Pinecone in batches.

        WHY UPSERT (not insert)?
          UPSERT = UPDATE if ID exists, INSERT if new.
          Since we use stable IDs (same chunk → same ID), re-running
          the pipeline UPDATES existing vectors instead of duplicating.

        Args:
            chunks: list of chunk dicts (must have 'embedding' and 'chunk_id').

        Returns:
            int — total number of vectors successfully uploaded.

        Raises:
            VectorStoreError — if connection fails (individual batch
                               failures are logged and skipped).
        """
        if not chunks:
            logger.warning("No chunks to upsert")
            return 0

        index = self.get_index()

        # Build batches
        batches = [
            chunks[i: i + config.UPSERT_BATCH_SIZE]
            for i in range(0, len(chunks), config.UPSERT_BATCH_SIZE)
        ]

        logger.info("Uploading to namespace '%s':", config.NAMESPACE_HOTPOT)
        logger.info("  %d vectors | %d batches", len(chunks), len(batches))

        total = 0
        failed_batches = 0

        for batch in tqdm(batches, desc=f"  [{config.NAMESPACE_HOTPOT}]"):
            try:
                vectors = self._build_vector_tuples(batch)
                if not vectors:
                    continue

                response = index.upsert(
                    vectors=vectors,
                    namespace=config.NAMESPACE_HOTPOT,
                )
                total += response.get("upserted_count", len(vectors))

            except Exception as e:
                failed_batches += 1
                logger.warning("  Batch failed: %s — skipping", str(e))

        logger.info("  Uploaded %d vectors", total)
        if failed_batches > 0:
            logger.warning("  %d batches failed (see warnings above)", failed_batches)

        return total

    def query(
        self,
        vector: List[float],
        top_k: int = None,
        namespace: str = None,
        include_metadata: bool = True,
    ) -> dict:
        """
        Queries Pinecone for the most similar vectors.

        Args:
            vector:           query embedding (list of 384 floats).
            top_k:            number of results to return (default: config.TOP_K).
            namespace:        Pinecone namespace (default: config.NAMESPACE_HOTPOT).
            include_metadata: whether to include metadata in results.

        Returns:
            Pinecone query result dict with 'matches' key.

        Raises:
            VectorStoreError — if query fails.
        """
        try:
            index = self.get_index()
            return index.query(
                vector=vector,
                top_k=top_k or config.TOP_K,
                namespace=namespace or config.NAMESPACE_HOTPOT,
                include_metadata=include_metadata,
            )
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Pinecone query failed: {e}"
            ) from e

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _connect(self):
        """
        Connects to Pinecone and creates the index if it doesn't exist.

        Raises:
            VectorStoreError — if API key is missing or connection fails.
        """
        # ── Validate API key ──────────────────────────────────────────────
        self._validate_api_key()

        try:
            logger.info("Connecting to Pinecone...")
            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            existing = [idx.name for idx in pc.list_indexes()]

            if config.PINECONE_INDEX not in existing:
                logger.info(
                    "  Creating index '%s'...", config.PINECONE_INDEX
                )
                pc.create_index(
                    name=config.PINECONE_INDEX,
                    dimension=config.EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=config.PINECONE_REGION,
                    ),
                )
                logger.info("  Index created")
            else:
                logger.info(
                    "  Index '%s' already exists", config.PINECONE_INDEX
                )

            index = pc.Index(config.PINECONE_INDEX)
            stats = index.describe_index_stats()
            logger.info(
                "  Current vector count: %d",
                stats.get("total_vector_count", 0),
            )
            return index

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Failed to connect to Pinecone: {e}"
            ) from e

    @staticmethod
    def _validate_api_key() -> None:
        """
        Validates that the Pinecone API key is set.

        Raises:
            VectorStoreError — if key is missing or placeholder.
        """
        if (
            not config.PINECONE_API_KEY
            or config.PINECONE_API_KEY == "PASTE_YOUR_KEY_HERE"
        ):
            raise VectorStoreError(
                "Pinecone API key not set!\n"
                "Run this in your terminal first:\n"
                "  set PINECONE_API_KEY=your-actual-key-here"
            )

    @staticmethod
    def _build_vector_tuples(
        batch: List[Dict[str, Any]],
    ) -> List[tuple]:
        """
        Converts a batch of chunk dicts into Pinecone vector tuples.

        Each tuple = (id, vector, metadata).

        Returns:
            list of (str, list[float], dict) tuples.
        """
        vectors = []
        for chunk in batch:
            if "embedding" not in chunk:
                continue

            vectors.append((
                chunk["chunk_id"],
                chunk["embedding"],
                {
                    # Text stored so no second DB needed
                    "original_text": chunk["text"][:1000],

                    # Dataset and document info
                    "source":    chunk["source"],
                    "doc_id":    chunk["doc_id"],
                    "chunk_idx": chunk["chunk_idx"],
                    "title":     chunk["title"][:200],

                    # Ground truth for evaluation
                    "question":  chunk.get("question", "")[:300],
                    "answer":    chunk.get("answer", "")[:200],

                    # Retrieval strategy flags
                    "is_multihop": chunk.get("is_multihop", False),
                    "is_bridge":   chunk.get("is_bridge", False),

                    # HotpotQA metadata
                    "type":  chunk.get("type", ""),
                    "level": chunk.get("level", ""),
                },
            ))
        return vectors


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logger.info("Testing Embedder and VectorStoreManager...\n")

    # Try loading existing chunks first
    chunks = []
    if os.path.exists(config.CHUNKS_FILE):
        with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunks.append(json.loads(line.strip()))
                except Exception:
                    pass
        logger.info("Loaded %d chunks from disk", len(chunks))
    else:
        logger.error(
            "No chunks file found at %s. "
            "Run pre_processing.py first.",
            config.CHUNKS_FILE,
        )
        exit(1)

    if chunks:
        logger.info("Got %d chunks", len(chunks))
        logger.info("Sample chunk:")
        s = chunks[0]
        logger.info("  chunk_id  : %s", s["chunk_id"])
        logger.info("  source    : %s", s["source"])
        logger.info("  is_bridge : %s", s["is_bridge"])
        logger.info("  text[:80] : %s...", s["text"][:80])

        # Embed and upload to Pinecone
        try:
            embedder = Embedder()
            total = embedder.embed_and_upload(chunks)
            logger.info("Test complete. %d vectors in Pinecone.", total)
        except (EmbeddingError, VectorStoreError) as e:
            logger.error("Test failed: %s", e)
            exit(1)
