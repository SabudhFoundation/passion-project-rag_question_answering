"""
src/exceptions.py
=================
Centralized custom exception hierarchy for the RAG pipeline.

WHY CUSTOM EXCEPTIONS?
  - Generic `Exception` tells you NOTHING about what went wrong.
  - Custom exceptions let callers catch SPECIFIC failure types
    and handle each differently.
  - Makes debugging 10x faster — the exception name itself tells
    you which pipeline stage failed.

HIERARCHY:
    RAGPipelineError          ← base (catch-all for any pipeline error)
    ├── DataDownloadError     ← network / file download failures
    ├── PreprocessingError    ← data cleaning / validation failures
    ├── ChunkingError         ← text splitting failures
    ├── EmbeddingError        ← model loading / encoding failures
    ├── VectorStoreError      ← Pinecone connection / upsert failures
    ├── RetrievalError        ← hybrid retrieval failures (BM25/vector)
    └── GenerationError       ← LLM generation failures (Groq API)

USAGE:
    from exceptions import DataDownloadError

    try:
        downloader.download()
    except DataDownloadError as e:
        logger.error(f"Download failed: {e}")
"""


class RAGPipelineError(Exception):
    """
    Base exception for all RAG pipeline errors.

    Catch this to handle ANY pipeline failure in one place:
        try:
            run_ingestion()
        except RAGPipelineError as e:
            logger.error(f"Pipeline failed: {e}")
    """
    pass


class DataDownloadError(RAGPipelineError):
    """
    Raised when dataset download fails.

    Common causes:
      - No internet connection
      - Server (CMU) is down
      - File is corrupted (too small after download)
      - Disk is full
    """
    pass


class PreprocessingError(RAGPipelineError):
    """
    Raised when data cleaning or validation fails.

    Common causes:
      - JSON file is malformed or truncated
      - Required fields missing from dataset entries
      - File not found (forgot to run download first)
    """
    pass


class ChunkingError(RAGPipelineError):
    """
    Raised when text splitting fails.

    Common causes:
      - Empty input records
      - Text splitter misconfigured (invalid chunk_size)
      - Disk full when saving chunks to JSONL
    """
    pass


class EmbeddingError(RAGPipelineError):
    """
    Raised when embedding model loading or text encoding fails.

    Common causes:
      - Model download fails (first run, no internet)
      - Out of memory (model is ~80 MB)
      - Input text is None or wrong type
    """
    pass


class VectorStoreError(RAGPipelineError):
    """
    Raised when Pinecone operations fail.

    Common causes:
      - Invalid or missing API key
      - Network timeout during upsert
      - Index dimension mismatch (384 expected)
      - Rate limiting (too many requests)
    """
    pass


class RetrievalError(RAGPipelineError):
    """
    Raised when hybrid retrieval (BM25 + vector search) fails.

    Common causes:
      - Vector store not initialized (forgot to run ingestion)
      - BM25 index build fails (empty document list)
      - Ensemble retriever misconfigured
      - Query returns no results
    """
    pass


class GenerationError(RAGPipelineError):
    """
    Raised when LLM generation fails.

    Common causes:
      - Invalid or missing Groq API key
      - LLM returned invalid JSON response
      - Model rate limiting or timeout
      - Empty context provided to generator
    """
    pass
