"""
src/preprocessing_data/pre_processing.py
=========================================
Data Ingestion + Chunking module.

Contains THREE classes, each with a single responsibility:
  1. DataDownloader  — downloads HotpotQA from the internet
  2. Preprocessor    — cleans and structures the raw data
  3. Chunker         — splits clean records into overlapping text chunks

DATA FLOW:
    DataDownloader.download()
        ↓ returns file path (string)
    Preprocessor.process(file_path)
        ↓ returns list of clean record dicts
    Chunker.chunk_records(records)
        ↓ returns list of chunk dicts (ready for embedding)

DESIGN PRINCIPLES:
  - Single Responsibility: Each class does exactly ONE job.
  - Fail-fast: Custom exceptions are raised immediately on errors.
  - Logging: All output uses the `logging` module (no print).
  - Defensive: Every public method validates its inputs first.
"""

import os
import re
import json
import time
import hashlib
import urllib.request
from typing import List, Optional, Dict, Any

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Project imports ───────────────────────────────────────────────────────────
import sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from exceptions import (
    DataDownloadError,
    PreprocessingError,
    ChunkingError,
)

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 1: DataDownloader
# ═════════════════════════════════════════════════════════════════════════════

class DataDownloader:
    """
    Downloads the HotpotQA dataset from CMU official servers.

    SINGLE RESPONSIBILITY: Only downloads. Nothing else.

    WHAT IS HotpotQA?
      - 112,779 multi-hop question-answer pairs
      - Each question needs TWO Wikipedia paragraphs to answer
      - Fields: _id, question, answer, context, supporting_facts, type, level
      - Source: https://hotpotqa.github.io/

    FEATURES:
      - Automatic retry with exponential backoff (max 3 attempts)
      - Skip download if file already exists and is valid
      - Custom exception on failure (DataDownloadError)

    USAGE:
        downloader = DataDownloader()
        file_path  = downloader.download()
    """

    # ── Class constants ───────────────────────────────────────────────────────
    _DOWNLOAD_URL = (
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
    )
    _MIN_FILE_SIZE_MB = 100        # files smaller than this are corrupted
    _MAX_RETRIES = 3               # download retry attempts
    _RETRY_BACKOFF_SECONDS = 5     # initial wait between retries

    def __init__(self) -> None:
        """Initializes the downloader and creates the raw data directory."""
        try:
            os.makedirs(config.RAW_DIR, exist_ok=True)
            logger.info("DataDownloader initialised")
            logger.info("  Data directory: %s", config.RAW_DIR)
        except OSError as e:
            raise DataDownloadError(
                f"Cannot create data directory '{config.RAW_DIR}': {e}"
            ) from e

    def __repr__(self) -> str:
        return (
            f"DataDownloader(url='{self._DOWNLOAD_URL}', "
            f"target='{config.HOTPOTQA_FILE}')"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def download(self) -> str:
        """
        Downloads HotpotQA training set with retry logic.

        Returns:
            str — local file path where the dataset was saved.

        Raises:
            DataDownloadError — if download fails after all retries.
        """
        logger.info("=" * 55)
        logger.info("  DataDownloader: Downloading HotpotQA")
        logger.info("=" * 55)

        # Skip if file already exists and is valid
        if self._already_downloaded():
            return config.HOTPOTQA_FILE

        logger.info("Downloading HotpotQA training set...")
        logger.info("  Size: ~540 MB | Source: CMU servers")
        logger.info("  This may take 5-15 minutes depending on your internet.")

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                logger.info(
                    "  Download attempt %d/%d", attempt, self._MAX_RETRIES
                )
                urllib.request.urlretrieve(
                    self._DOWNLOAD_URL,
                    config.HOTPOTQA_FILE,
                    self._show_progress,
                )
                mb = os.path.getsize(config.HOTPOTQA_FILE) / (1024 * 1024)
                logger.info("  Downloaded successfully (%.0f MB)", mb)
                logger.info("  Saved to: %s", config.HOTPOTQA_FILE)
                return config.HOTPOTQA_FILE

            except Exception as e:
                last_error = e
                logger.warning(
                    "  Attempt %d failed: %s", attempt, str(e)
                )
                if attempt < self._MAX_RETRIES:
                    wait = self._RETRY_BACKOFF_SECONDS * attempt
                    logger.info("  Retrying in %d seconds...", wait)
                    time.sleep(wait)

        # All retries exhausted
        raise DataDownloadError(
            f"Download failed after {self._MAX_RETRIES} attempts. "
            f"Last error: {last_error}\n"
            f"Try downloading manually from: {self._DOWNLOAD_URL}\n"
            f"Save to: {config.HOTPOTQA_FILE}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _already_downloaded(self) -> bool:
        """
        Returns True if the file already exists and is big enough.
        Prevents re-downloading 540MB every time you run the script.
        """
        if not os.path.exists(config.HOTPOTQA_FILE):
            return False

        mb = os.path.getsize(config.HOTPOTQA_FILE) / (1024 * 1024)
        if mb < self._MIN_FILE_SIZE_MB:
            logger.warning(
                "  File exists but too small (%.0f MB) — re-downloading", mb
            )
            return False

        logger.info(
            "  HotpotQA already downloaded (%.0f MB) — skipping", mb
        )
        return True

    def _show_progress(
        self, block_num: int, block_size: int, total_size: int
    ) -> None:
        """Callback that shows download progress percentage."""
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100)
            mb = downloaded / (1024 * 1024)
            print(
                f"\r   {percent:.1f}% — {mb:.0f} MB downloaded",
                end="", flush=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 2: Preprocessor
# ═════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    """
    Reads raw HotpotQA JSON and produces clean, structured records.

    SINGLE RESPONSIBILITY: Only cleans and structures data. Nothing else.

    WHAT DOES THIS CLASS DO?
      1. Opens the raw HotpotQA JSON file
      2. For each entry, extracts the question, answer, and context
      3. Filters context to ONLY supporting paragraphs (skips distractors)
      4. Cleans text (removes extra spaces, invisible characters)
      5. Returns a list of clean record dicts in a unified schema

    WHY FILTER TO SUPPORTING PARAGRAPHS?
      HotpotQA distractor setting includes 10 paragraphs per question:
        - 2 gold (supporting) paragraphs → contain the actual answer
        - 8 distractor paragraphs → noise, unrelated to the answer
      We only use the 2 gold paragraphs. Embedding distractors would
      add noise to Pinecone and make retrieval less accurate.

    OUTPUT SCHEMA:
      {
        "doc_id":          unique ID for this QA pair,
        "title":           titles of supporting paragraphs joined,
        "text":            fused supporting paragraph text,
        "source":          always "hotpotqa",
        "question":        the multi-hop question,
        "answer":          the gold answer,
        "is_multihop":     always True for HotpotQA,
        "type":            "bridge" or "comparison",
        "level":           "easy", "medium", or "hard",
        "bridge_entities": list of bridge entity titles,
      }

    USAGE:
        preprocessor = Preprocessor()
        records      = preprocessor.process("path/to/hotpot_train.json")
    """

    # Compiled regex patterns (module-level compilation for performance)
    _RE_ZERO_WIDTH = re.compile(r'[\u200b\u200c\u200d\ufeff]')
    _RE_WHITESPACE = re.compile(r'\s+')

    def __init__(self) -> None:
        """Initializes the preprocessor and creates the processed data directory."""
        try:
            os.makedirs(config.PROCESSED_DIR, exist_ok=True)
            logger.info("Preprocessor initialised")
        except OSError as e:
            raise PreprocessingError(
                f"Cannot create processed directory '{config.PROCESSED_DIR}': {e}"
            ) from e

    def __repr__(self) -> str:
        return (
            f"Preprocessor(processed_dir='{config.PROCESSED_DIR}', "
            f"max_records={config.MAX_HOTPOT_RECORDS})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Reads HotpotQA file and returns list of clean record dicts.

        Args:
            file_path: path to hotpot_train_v1.1.json

        Returns:
            list of clean record dicts (see schema in class docstring).

        Raises:
            PreprocessingError — if file cannot be read or parsed.
        """
        logger.info("=" * 55)
        logger.info("  Preprocessor: Cleaning HotpotQA data")
        logger.info("=" * 55)

        # ── Input validation ──────────────────────────────────────────────
        self._validate_file_path(file_path)

        # ── Load raw JSON ─────────────────────────────────────────────────
        raw_data = self._load_json(file_path)

        # ── Apply record limit ────────────────────────────────────────────
        if config.MAX_HOTPOT_RECORDS:
            raw_data = raw_data[:config.MAX_HOTPOT_RECORDS]
            logger.info(
                "  Using first %d records (MAX_HOTPOT_RECORDS=%d)",
                len(raw_data), config.MAX_HOTPOT_RECORDS,
            )

        # ── Process each entry ────────────────────────────────────────────
        records: List[Dict[str, Any]] = []
        skipped = 0

        for entry in tqdm(raw_data, desc="  Cleaning"):
            try:
                record = self._process_one_entry(entry)
                if record:
                    records.append(record)
                else:
                    skipped += 1
            except Exception as e:
                logger.debug(
                    "  Skipping entry '%s': %s",
                    entry.get("_id", "unknown"), str(e),
                )
                skipped += 1

        # ── Save to disk ──────────────────────────────────────────────────
        self._save_to_disk(records)

        logger.info("  Preprocessor complete:")
        logger.info("    Clean records : %d", len(records))
        logger.info("    Skipped       : %d", skipped)

        return records

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_file_path(self, file_path: str) -> None:
        """
        Validates that the file path is a non-empty string pointing to
        an existing file. Raises PreprocessingError otherwise.
        """
        if not file_path or not isinstance(file_path, str):
            raise PreprocessingError(
                "file_path must be a non-empty string. "
                "Run DataDownloader.download() first."
            )
        if not os.path.exists(file_path):
            raise PreprocessingError(
                f"File not found: '{file_path}'. "
                "Run DataDownloader.download() first."
            )

    def _load_json(self, file_path: str) -> list:
        """
        Loads and parses the raw JSON file.

        Raises:
            PreprocessingError — if file cannot be read or is not valid JSON.
        """
        try:
            logger.info("  Loading: %s", file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            logger.info("  Total entries in file: %d", len(raw_data))
            return raw_data
        except json.JSONDecodeError as e:
            raise PreprocessingError(
                f"Invalid JSON in '{file_path}': {e}"
            ) from e
        except IOError as e:
            raise PreprocessingError(
                f"Cannot read file '{file_path}': {e}"
            ) from e

    def _process_one_entry(self, entry: dict) -> Optional[Dict[str, Any]]:
        """
        Processes one HotpotQA entry → one clean record dict.
        Returns None if the entry is invalid (missing required fields).

        HotpotQA entry structure:
          {
            "_id": "abc123",
            "question": "Who directed the film that ...",
            "answer": "Christopher Nolan",
            "context": [
              ["Film Title", ["sentence 0", "sentence 1", ...]],
              ["Director Name", ["sentence 0", ...]],
              ...8 more distractor paragraphs...
            ],
            "supporting_facts": [["Film Title", 0], ["Director Name", 1]],
            "type": "bridge",
            "level": "hard"
          }
        """
        _id       = entry.get("_id", "")
        question  = self._clean_text(entry.get("question", ""))
        answer    = self._clean_text(entry.get("answer", ""))
        context   = entry.get("context", [])
        sup_facts = entry.get("supporting_facts", [])
        q_type    = entry.get("type", "bridge")
        level     = entry.get("level", "medium")

        # ── Validate required fields ─────────────────────────────────────
        if not self._validate_entry(question, context):
            return None

        # ── Build title → paragraph map ──────────────────────────────────
        para_map = {}
        for title, sentences in context:
            para_map[title] = self._clean_text(" ".join(sentences))

        # ── Extract supporting titles (2 gold paragraphs) ────────────────
        supporting_titles = list(dict.fromkeys(
            title for title, _ in sup_facts
        ))

        # Fallback: if supporting_facts is empty, use all context titles
        if not supporting_titles:
            supporting_titles = [title for title, _ in context]

        # ── Identify bridge entities ─────────────────────────────────────
        bridge_entities = (
            supporting_titles[1:] if len(supporting_titles) > 1 else []
        )

        # ── Fuse ONLY supporting paragraphs ──────────────────────────────
        fused_parts = [
            f"[{title}] {para_map[title]}"
            for title in supporting_titles
            if title in para_map and len(para_map.get(title, "")) > 30
        ]
        fused_text = " ".join(fused_parts)

        # Skip if context is too short to be meaningful
        if len(fused_text) < 50:
            return None

        return {
            "doc_id":          _id,
            "title":           " | ".join(supporting_titles),
            "text":            fused_text,
            "source":          "hotpotqa",
            "question":        question,
            "answer":          answer,
            "is_multihop":     True,
            "type":            q_type,
            "level":           level,
            "bridge_entities": bridge_entities,
        }

    @staticmethod
    def _validate_entry(question: str, context: list) -> bool:
        """
        Validates that an entry has the minimum required fields
        to produce a useful record.
        """
        if not question:
            return False
        if not context or not isinstance(context, list):
            return False
        return True

    def _clean_text(self, text: str) -> str:
        """
        Removes extra whitespace and invisible characters from text.

        WHY CLEAN TEXT?
          Raw text often has:
            - Multiple spaces between words
            - Invisible zero-width characters (common in scraped web text)
            - Leading/trailing whitespace
          These waste embedding tokens and add noise to vectors.
        """
        if not text or not isinstance(text, str):
            return ""
        # Remove zero-width characters
        text = self._RE_ZERO_WIDTH.sub('', text)
        # Collapse multiple whitespace into single space
        text = self._RE_WHITESPACE.sub(' ', text)
        return text.strip()

    def _save_to_disk(self, records: List[Dict[str, Any]]) -> None:
        """
        Saves processed records to src/data/processed/ as JSONL.

        Raises:
            PreprocessingError — if disk write fails.
        """
        path = os.path.join(config.PROCESSED_DIR, "hotpotqa_processed.jsonl")
        try:
            with open(path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("  Saved processed records to: %s", path)
        except IOError as e:
            raise PreprocessingError(
                f"Cannot save processed records to '{path}': {e}"
            ) from e


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 3: Chunker (moved from build_features.py)
# ═════════════════════════════════════════════════════════════════════════════

class Chunker:
    """
    Splits preprocessed records into overlapping text chunks.

    SINGLE RESPONSIBILITY: Only splits text. Nothing else.

    WHY DO WE NEED CHUNKING?
      The embedding model (MiniLM) can read at most ~512 tokens at once.
      A HotpotQA context can be 500-1000+ tokens.
      So we cut each context into 512-character pieces.
      Each piece gets its own embedding vector in Pinecone.

    WHY OVERLAP (50 characters)?
      For HotpotQA bridge questions (88.4% of dataset), the bridge
      entity (linking word between two hops) must appear in at least
      one chunk. Overlap guarantees this.

    WHY RecursiveCharacterTextSplitter?
      It tries to split at natural boundaries, in this order:
        1. Paragraph break (\\n\\n) → best, keeps ideas together
        2. Line break (\\n)
        3. Sentence end (". ")
        4. Space (" ") → last resort
      It NEVER cuts mid-word.

    USAGE:
        chunker = Chunker()
        chunks  = chunker.chunk_records(records)
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> None:
        """
        Initializes the chunker with configurable chunk size and overlap.

        Args:
            chunk_size:    characters per chunk (default: config.CHUNK_SIZE)
            chunk_overlap: overlap between chunks (default: config.CHUNK_OVERLAP)

        Raises:
            ChunkingError — if chunk_size or chunk_overlap are invalid.
        """
        self._chunk_size = chunk_size or config.CHUNK_SIZE
        self._chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        # Validate parameters
        if self._chunk_size <= 0:
            raise ChunkingError(
                f"chunk_size must be positive, got {self._chunk_size}"
            )
        if self._chunk_overlap < 0:
            raise ChunkingError(
                f"chunk_overlap must be non-negative, got {self._chunk_overlap}"
            )
        if self._chunk_overlap >= self._chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({self._chunk_overlap}) must be less than "
                f"chunk_size ({self._chunk_size})"
            )

        try:
            self._splitter = RecursiveCharacterTextSplitter(
                separators      = ["\n\n", "\n", ". ", " "],
                chunk_size      = self._chunk_size,
                chunk_overlap   = self._chunk_overlap,
                length_function = len,
            )
        except Exception as e:
            raise ChunkingError(
                f"Failed to initialize text splitter: {e}"
            ) from e

        logger.info(
            "Chunker initialised (size=%d, overlap=%d)",
            self._chunk_size, self._chunk_overlap,
        )

    def __repr__(self) -> str:
        return (
            f"Chunker(chunk_size={self._chunk_size}, "
            f"chunk_overlap={self._chunk_overlap})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def chunk_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits all records into chunks and saves them to disk.

        Args:
            records: list of clean record dicts from Preprocessor.

        Returns:
            list of chunk dicts, each ready for embedding.

        Raises:
            ChunkingError — if input is invalid or chunking fails.
        """
        logger.info("=" * 55)
        logger.info("  Chunker: Splitting records into chunks")
        logger.info("=" * 55)

        if not records:
            raise ChunkingError("No records provided for chunking.")

        if not isinstance(records, list):
            raise ChunkingError(
                f"Expected list of records, got {type(records).__name__}"
            )

        all_chunks: List[Dict[str, Any]] = []

        for record in tqdm(records, desc="  Chunking"):
            try:
                chunks = self._chunk_one_record(record)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(
                    "  Failed to chunk record '%s': %s",
                    record.get("doc_id", "unknown"), str(e),
                )

        # Save to disk
        self._save_to_disk(all_chunks)

        # Print summary
        bridge_chunks = [c for c in all_chunks if c.get("is_bridge")]
        logger.info("  Chunking complete:")
        logger.info("    Records processed : %d", len(records))
        logger.info("    Total chunks      : %d", len(all_chunks))
        logger.info("    Bridge chunks     : %d", len(bridge_chunks))
        logger.info(
            "    Avg chunks/record : %d",
            len(all_chunks) // max(len(records), 1),
        )

        return all_chunks

    def load_chunks_from_disk(self) -> List[Dict[str, Any]]:
        """
        Loads previously saved chunks from disk.

        Returns:
            list of chunk dicts (empty list if file not found).
        """
        if not os.path.exists(config.CHUNKS_FILE):
            logger.warning("No chunks file found. Run chunk_records() first.")
            return []

        chunks: List[Dict[str, Any]] = []
        try:
            with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chunks.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except IOError as e:
            raise ChunkingError(
                f"Cannot read chunks file '{config.CHUNKS_FILE}': {e}"
            ) from e

        logger.info("Loaded %d chunks from disk", len(chunks))
        return chunks

    @staticmethod
    def to_langchain_documents(chunks: List[Dict[str, Any]]) -> list:
        """
        Converts chunk dicts into LangChain Document objects.

        WHY LANGCHAIN DOCUMENTS?
          The HybridRetriever (BM25 + Vector ensemble) requires
          LangChain Document objects, not raw dicts.
          Each Document has:
            - page_content: the chunk text
            - metadata: all other fields (source, title, doc_id, etc.)

        Args:
            chunks: list of chunk dicts from chunk_records().

        Returns:
            list of LangChain Document objects.
        """
        from langchain_core.documents import Document

        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "source":      chunk.get("source", ""),
                    "title":       chunk.get("title", ""),
                    "doc_id":      chunk.get("doc_id", ""),
                    "chunk_id":    chunk.get("chunk_id", ""),
                    "chunk_idx":   chunk.get("chunk_idx", 0),
                    "question":    chunk.get("question", ""),
                    "answer":      chunk.get("answer", ""),
                    "is_multihop": chunk.get("is_multihop", False),
                    "is_bridge":   chunk.get("is_bridge", False),
                    "type":        chunk.get("type", ""),
                    "level":       chunk.get("level", ""),
                },
            )
            documents.append(doc)

        logger.info("Converted %d chunks to LangChain Documents", len(documents))
        return documents

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _chunk_one_record(self, record: dict) -> List[Dict[str, Any]]:
        """
        Splits one record's text into overlapping chunks.
        Attaches full metadata to each chunk.

        QUESTION-AWARE CHUNKING:
          We prepend the question before the context text.
          This shifts the chunk's embedding vector toward the
          question's meaning → better retrieval precision.
        """
        # Prepend question for question-aware embedding
        if record.get("question"):
            text_to_split = (
                f"Q: {record['question']}\n"
                f"Context: {record['text']}"
            )
        else:
            text_to_split = record["text"]

        chunk_texts     = self._splitter.split_text(text_to_split)
        bridge_entities = record.get("bridge_entities", [])
        chunks: List[Dict[str, Any]] = []

        for idx, chunk_text in enumerate(chunk_texts):
            # Skip chunks that are too short to be meaningful
            if len(chunk_text.strip()) < 30:
                continue

            # Bridge detection
            chunk_lower = chunk_text.lower()
            is_bridge = any(
                entity.lower() in chunk_lower
                for entity in bridge_entities
            ) if bridge_entities else False

            chunk = {
                "chunk_id":    self._make_stable_id(
                                   record["doc_id"], idx, chunk_text),
                "doc_id":      record["doc_id"],
                "chunk_idx":   idx,
                "text":        chunk_text,
                "source":      record["source"],
                "title":       record["title"],
                "question":    record.get("question", ""),
                "answer":      record.get("answer", ""),
                "is_multihop": record.get("is_multihop", False),
                "is_bridge":   is_bridge,
                "type":        record.get("type", ""),
                "level":       record.get("level", ""),
            }
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _make_stable_id(doc_id: str, idx: int, text: str) -> str:
        """
        Creates a stable unique ID for each chunk using MD5 hash.

        WHY STABLE?
          If you run ingestion twice, the same chunk gets the same ID.
          Pinecone's UPSERT will UPDATE instead of creating a duplicate.
        """
        raw    = f"{doc_id}_{idx}_{text[:30]}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:20]
        return digest

    def _save_to_disk(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Saves all chunks to src/data/processed/chunks.jsonl.

        Raises:
            ChunkingError — if disk write fails.
        """
        try:
            os.makedirs(config.PROCESSED_DIR, exist_ok=True)
            with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            logger.info("  Chunks saved to: %s", config.CHUNKS_FILE)
        except IOError as e:
            raise ChunkingError(
                f"Cannot save chunks to '{config.CHUNKS_FILE}': {e}"
            ) from e


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Testing DataDownloader, Preprocessor, and Chunker...\n")

    # ── Test DataDownloader ───────────────────────────────────────────────
    try:
        downloader = DataDownloader()
        path = downloader.download()
    except DataDownloadError as e:
        logger.error("Download failed: %s", e)
        exit(1)

    # ── Test Preprocessor ─────────────────────────────────────────────────
    try:
        preprocessor = Preprocessor()
        records = preprocessor.process(path)
        logger.info("Got %d clean records.", len(records))
    except PreprocessingError as e:
        logger.error("Preprocessing failed: %s", e)
        exit(1)

    # ── Test Chunker ──────────────────────────────────────────────────────
    if records:
        try:
            chunker = Chunker()
            chunks = chunker.chunk_records(records)
            logger.info("Got %d chunks.", len(chunks))
            if chunks:
                sample = chunks[0]
                logger.info("Sample chunk:")
                logger.info("  chunk_id  : %s", sample["chunk_id"])
                logger.info("  source    : %s", sample["source"])
                logger.info("  is_bridge : %s", sample["is_bridge"])
                logger.info("  text[:80] : %s...", sample["text"][:80])
        except ChunkingError as e:
            logger.error("Chunking failed: %s", e)
            exit(1)
