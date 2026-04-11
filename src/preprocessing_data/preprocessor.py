"""
src/preprocessing_data/preprocessor.py
========================================
CLASS: Preprocessor

ONE CLASS — ONE FILE.
SINGLE RESPONSIBILITY: Clean and structure raw HotpotQA JSON. Nothing else.

EXCEPTION HANDLING:
  File not found, corrupted JSON, and missing fields all handled with
  clear error messages explaining what went wrong and how to fix it.
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import (
    get_logger, ensure_dir, write_jsonl,
    clean_text, is_valid_text, print_section, print_summary,
)

logger = get_logger(__name__)


class Preprocessor:
    """
    Reads raw HotpotQA JSON → returns clean, unified record dicts.

    CLEAN RECORD SCHEMA (what this class outputs):
      doc_id, title, text, source, question, answer,
      is_multihop, type, level, bridge_entities

    USAGE:
        preprocessor = Preprocessor()
        records      = preprocessor.process("path/to/hotpot.json")
    """

    def __init__(self):
        try:
            ensure_dir(config.PROCESSED_DIR)
            logger.info("Preprocessor ready")
        except OSError as e:
            raise OSError(
                f"Cannot create processed data directory: {config.PROCESSED_DIR}\n"
                f"Reason: {e}"
            )

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def process(self, file_path: str) -> list:
        """
        Main method: reads HotpotQA file → list of clean record dicts.

        Args:
            file_path: path to hotpot_train_v1.1.json

        Returns:
            list of clean record dicts

        Raises:
            FileNotFoundError: if file_path does not exist
            ValueError:        if file cannot be parsed as JSON
        """
        print_section("STEP 2/4 — Preprocessor")

        # ── Validate input ────────────────────────────────────────────────────
        if not file_path:
            raise ValueError(
                "file_path cannot be empty.\n"
                "Pass the path returned by DataDownloader.download()."
            )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"HotpotQA file not found: {file_path}\n"
                f"Run DataDownloader.download() first."
            )

        # ── Load raw JSON ─────────────────────────────────────────────────────
        logger.info(f"Loading: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"HotpotQA file is not valid JSON: {e}\n"
                f"File may be corrupted. Delete it and re-run DataDownloader."
            )
        except OSError as e:
            raise OSError(
                f"Cannot read file {file_path}: {e}\n"
                f"Check file permissions."
            )

        logger.info(f"Total entries in file: {len(raw_data):,}")

        # Apply record limit (for testing)
        if config.MAX_HOTPOT_RECORDS:
            raw_data = raw_data[:config.MAX_HOTPOT_RECORDS]
            logger.info(
                f"Using first {len(raw_data):,} records "
                f"(MAX_HOTPOT_RECORDS={config.MAX_HOTPOT_RECORDS})"
            )

        # ── Process each entry ────────────────────────────────────────────────
        from tqdm import tqdm
        records  = []
        skipped  = 0
        errors   = 0

        for entry in tqdm(raw_data, desc="  Preprocessing"):
            try:
                record = self._process_entry(entry)
                if record:
                    records.append(record)
                else:
                    skipped += 1
            except Exception as e:
                # Log individual entry errors but keep going
                errors += 1
                logger.debug(f"Skipping entry {entry.get('_id','?')}: {e}")

        # ── Save to disk ──────────────────────────────────────────────────────
        save_path = os.path.join(config.PROCESSED_DIR, "hotpotqa_processed.jsonl")
        try:
            write_jsonl(records, save_path)
        except IOError as e:
            # Non-fatal: log warning but still return records
            logger.warning(f"Could not save processed records: {e}")

        print_summary("Preprocessor Complete", {
            "Clean records": len(records),
            "Skipped (invalid)": skipped,
            "Errors": errors,
        })

        return records

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _process_entry(self, entry: dict):
        """
        Processes one HotpotQA entry → one clean record dict.
        Returns None if entry is unusable.
        """
        _id      = entry.get("_id", "")
        question = clean_text(entry.get("question", ""))
        answer   = clean_text(entry.get("answer", ""))
        context  = entry.get("context", [])
        sup_facts= entry.get("supporting_facts", [])
        q_type   = entry.get("type", "bridge")
        level    = entry.get("level", "medium")

        if not question or not context:
            return None

        # Build paragraph map: title → cleaned text
        para_map = {}
        for item in context:
            if not isinstance(item, list) or len(item) < 2:
                continue
            title, sentences = item[0], item[1]
            if isinstance(sentences, list):
                para_map[title] = clean_text(" ".join(sentences))

        # Get supporting titles (gold paragraphs only)
        supporting_titles = list(dict.fromkeys(
            t for t, _ in sup_facts if isinstance(t, str)
        )) or list(para_map.keys())

        bridge_entities = supporting_titles[1:] if len(supporting_titles) > 1 else []

        # Fuse ONLY supporting paragraphs (skip distractors)
        fused_parts = [
            f"[{t}] {para_map[t]}"
            for t in supporting_titles
            if t in para_map and is_valid_text(para_map.get(t, ""))
        ]
        fused_text = " ".join(fused_parts)

        if not is_valid_text(fused_text, min_length=50):
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


# ── STANDALONE TEST ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        p = Preprocessor()
        records = p.process(config.HOTPOTQA_FILE)
        print(f"\n✅ Got {len(records)} clean records")
        if records:
            r = records[0]
            print(f"  Sample: Q={r['question'][:60]} | A={r['answer']}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ {e}")
