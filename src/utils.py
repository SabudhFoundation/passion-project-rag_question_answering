"""
src/utils.py
=============
Shared utility functions used across the entire project.

WHY utils.py EXISTS:
  Many classes need the same small helpers — like making a unique ID,
  reading a JSONL file, or setting up a logger.

  Without utils.py:
    Every class copy-pastes the same 10 lines. Bug in one? Fix in 10 places.

  With utils.py:
    Write once here. All classes import and reuse it.
    Fix a bug here once → all classes get the fix automatically.

  This is the DRY principle: Don't Repeat Yourself.

WHAT IS IN THIS FILE:
  - Logger setup       → consistent logging across all classes
  - File helpers       → read/write JSONL, ensure folders exist
  - ID generation      → stable unique IDs for Pinecone vectors
  - Text helpers       → clean text, check minimum length
  - Pinecone helpers   → format vectors for upsert

HOW TO IMPORT:
  from utils import get_logger, make_stable_id, read_jsonl, clean_text
"""

import os
import re
import json
import hashlib
import logging

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


# ─────────────────────────────────────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a named logger with consistent formatting.

    WHY USE A LOGGER instead of print()?
      - Logger has levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
      - You can turn off all INFO messages in production with one line
      - Log messages include timestamp and which class sent them
      - print() has none of these features

    USAGE IN ANY CLASS:
        from utils import get_logger
        logger = get_logger(__name__)   # __name__ = the file's module name
        logger.info("Starting download...")
        logger.warning("File already exists, skipping")
        logger.error("Download failed: connection refused")

    Args:
        name: usually pass __name__ — Python fills in the module name

    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if not logger.handlers:
        handler   = logging.StreamHandler()
        formatter = logging.Formatter(config.LOG_FORMAT, datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# FILE & FOLDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """
    Creates a directory if it doesn't already exist.
    Does nothing if it already exists (safe to call multiple times).

    USAGE:
        ensure_dir(config.RAW_DIR)
        ensure_dir(config.PROCESSED_DIR)
    """
    os.makedirs(path, exist_ok=True)


def read_jsonl(filepath: str) -> list:
    """
    Reads a JSONL file and returns a list of dicts.

    WHAT IS JSONL?
      JSON Lines = one complete JSON object per line.
      Regular JSON: one big list/object in the whole file.
      JSONL: {"id":1, "text":"..."}\n{"id":2, "text":"..."}\n...

      JSONL is better for large datasets because you can read
      one line at a time without loading the whole file into memory.

    Args:
        filepath: path to a .jsonl file

    Returns:
        list of dicts, one per line

    Raises:
        FileNotFoundError: if file does not exist
    """
    logger = get_logger(__name__)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"JSONL file not found: {filepath}\n"
            f"Make sure the previous pipeline step ran successfully."
        )

    records = []
    bad_lines = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_lines += 1
                logger.warning(f"Skipping bad JSON on line {line_num}: {e}")

    if bad_lines > 0:
        logger.warning(f"Skipped {bad_lines} malformed lines in {filepath}")

    logger.info(f"Loaded {len(records):,} records from {filepath}")
    return records


def write_jsonl(records: list, filepath: str) -> None:
    """
    Writes a list of dicts to a JSONL file (one dict per line).

    Args:
        records:  list of dicts to write
        filepath: output file path (creates parent dirs if needed)

    Raises:
        IOError: if file cannot be written
    """
    logger = get_logger(__name__)

    try:
        ensure_dir(os.path.dirname(filepath))
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(records):,} records to {filepath}")
    except IOError as e:
        raise IOError(
            f"Could not write to {filepath}: {e}\n"
            f"Check that the folder exists and you have write permission."
        )


def file_exists_and_valid(filepath: str, min_mb: float = 0,
                          min_lines: int = 0) -> bool:
    """
    Returns True if file exists and meets minimum size/line requirements.

    Used by DataDownloader to skip re-downloading large files.

    Args:
        filepath: path to check
        min_mb:   minimum file size in megabytes (0 = no check)
        min_lines: minimum number of lines (0 = no check)

    Returns:
        bool
    """
    if not os.path.exists(filepath):
        return False

    if min_mb > 0:
        actual_mb = os.path.getsize(filepath) / (1024 * 1024)
        if actual_mb < min_mb:
            return False

    if min_lines > 0:
        with open(filepath, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        if count < min_lines:
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# TEXT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Cleans a text string for embedding.

    WHAT WE REMOVE:
      - Zero-width characters (\u200b etc.) → invisible but waste tokens
      - Multiple spaces/newlines → collapse to single space
      - Leading and trailing whitespace

    Args:
        text: raw text string

    Returns:
        cleaned text string (empty string if input is invalid)
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove invisible zero-width characters common in scraped text
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

    # Collapse multiple whitespace (spaces, tabs, newlines) to single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def is_valid_text(text: str, min_length: int = 30) -> bool:
    """
    Returns True if text is long enough to be meaningful.

    WHY CHECK LENGTH?
      Very short texts (< 30 chars) provide no useful context.
      Embedding them just wastes Pinecone storage and API quota.

    Args:
        text:       string to check
        min_length: minimum character count (default 30)

    Returns:
        bool
    """
    return isinstance(text, str) and len(text.strip()) >= min_length


# ─────────────────────────────────────────────────────────────────────────────
# ID GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_stable_id(doc_id: str, chunk_idx: int, text: str) -> str:
    """
    Creates a stable, unique ID for a Pinecone vector.

    WHY STABLE IDs?
      Same chunk → same ID on every pipeline run.
      Pinecone UPSERT = UPDATE if ID exists, INSERT if new.
      Without stable IDs: run pipeline twice → double your vector count.
      With stable IDs: run pipeline twice → same vectors updated in place.

    HOW IT WORKS:
      MD5 hash of (doc_id + chunk_idx + first 30 chars of text)
      → deterministic → always the same for the same input
      → first 20 hex chars is enough for uniqueness

    Args:
        doc_id:    unique document identifier
        chunk_idx: position of chunk within document
        text:      the chunk text (first 30 chars used)

    Returns:
        20-character hex string
    """
    raw    = f"{doc_id}_{chunk_idx}_{text[:30]}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return digest[:20]


# ─────────────────────────────────────────────────────────────────────────────
# PINECONE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_pinecone_vector(chunk: dict) -> tuple:
    """
    Formats one chunk dict into the (id, vector, metadata) tuple
    that Pinecone's upsert() method expects.

    WHY A HELPER?
      The metadata schema must be identical in DataDownloader and in
      Retriever (for filtering). Having it in one place prevents drift.
      Change the schema here → both ingestion and retrieval see the change.

    PINECONE METADATA SCHEMA:
      original_text → raw chunk text (no second DB needed)
      source        → dataset name (for namespace filtering)
      doc_id        → parent document (for full-doc reconstruction)
      chunk_idx     → reading order position
      title         → document title
      question      → associated question (for question-aware retrieval)
      answer        → gold answer (for evaluation)
      is_multihop   → True for HotpotQA (retrieval strategy flag)
      is_bridge     → True if chunk contains bridge entity
      type          → "bridge" or "comparison" (HotpotQA question type)
      level         → "easy", "medium", "hard" (difficulty)

    Args:
        chunk: dict with 'chunk_id', 'embedding', and all metadata fields

    Returns:
        (id, vector, metadata) tuple for Pinecone upsert

    Raises:
        ValueError: if chunk is missing required 'embedding' field
    """
    if "embedding" not in chunk:
        raise ValueError(
            f"Chunk '{chunk.get('chunk_id', 'unknown')}' has no embedding.\n"
            f"Make sure Embedder._generate_embeddings() ran successfully."
        )

    metadata = {
        "original_text": chunk.get("text", "")[:1000],
        "source":        chunk.get("source", ""),
        "doc_id":        chunk.get("doc_id", ""),
        "chunk_idx":     chunk.get("chunk_idx", 0),
        "title":         chunk.get("title", "")[:200],
        "question":      chunk.get("question", "")[:300],
        "answer":        chunk.get("answer", "")[:200],
        "is_multihop":   chunk.get("is_multihop", False),
        "is_bridge":     chunk.get("is_bridge", False),
        "type":          chunk.get("type", ""),
        "level":         chunk.get("level", ""),
    }

    return (chunk["chunk_id"], chunk["embedding"], metadata)


def format_retrieval_result(match: dict) -> dict:
    """
    Formats a raw Pinecone query match into a clean result dict.
    Used by the retrieval team to standardise results.

    Args:
        match: raw match dict from Pinecone query response

    Returns:
        clean result dict with score and all metadata fields
    """
    meta = match.get("metadata", {})
    return {
        "score":      round(match.get("score", 0.0), 4),
        "chunk_id":   match.get("id", ""),
        "text":       meta.get("original_text", ""),
        "source":     meta.get("source", ""),
        "doc_id":     meta.get("doc_id", ""),
        "chunk_idx":  meta.get("chunk_idx", 0),
        "title":      meta.get("title", ""),
        "question":   meta.get("question", ""),
        "answer":     meta.get("answer", ""),
        "is_bridge":  meta.get("is_bridge", False),
        "is_multihop":meta.get("is_multihop", False),
        "type":       meta.get("type", ""),
        "level":      meta.get("level", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str, width: int = 55) -> None:
    """Prints a formatted section header to the terminal."""
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_summary(title: str, stats: dict) -> None:
    """Prints a formatted summary table to the terminal."""
    print(f"\n  {'─'*50}")
    print(f"  {title}")
    print(f"  {'─'*50}")
    for key, val in stats.items():
        if isinstance(val, int):
            print(f"  {key:<30}: {val:,}")
        else:
            print(f"  {key:<30}: {val}")
    print(f"  {'─'*50}")
