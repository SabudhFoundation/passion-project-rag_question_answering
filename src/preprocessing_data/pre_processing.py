"""
src/preprocessing_data/pre_processing.py
=========================================
Contains TWO classes:
  1. DataDownloader  — downloads HotpotQA from the internet
  2. Preprocessor    — cleans and structures the raw data

WHAT IS SINGLE RESPONSIBILITY PRINCIPLE?
  Each class does exactly ONE job.
  DataDownloader only downloads — it does not clean or chunk.
  Preprocessor only cleans — it does not download or embed.
  If downloading breaks, you only fix DataDownloader.
  If cleaning breaks, you only fix Preprocessor.
  They never interfere with each other.

HOW DATA FLOWS THROUGH THESE TWO CLASSES:
  DataDownloader.download()
      ↓ returns file path (string)
  Preprocessor.process(file_path)
      ↓ returns list of clean record dicts
  (passed to Chunker in the next file)
"""

import os
import sys
import json
import urllib.request

# Add src/ to path so we can import config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from tqdm import tqdm


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

    HOW TO USE:
        downloader = DataDownloader()
        file_path  = downloader.download()
    """

    def __init__(self):
        # Create the raw data folder if it doesn't exist
        os.makedirs(config.RAW_DIR, exist_ok=True)
        print("📁 DataDownloader initialised")
        print(f"   Data will be saved to: {config.RAW_DIR}")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD — this is what main.py calls
    # ─────────────────────────────────────────────────────────────────────────

    def download(self) -> str:
        """
        Downloads HotpotQA training set.

        Returns:
            str — local file path where the file was saved

        WHY RETURN THE PATH?
          Loose coupling — the caller (main.py) doesn't need to
          know WHERE the file is saved. It just receives the path
          and passes it to Preprocessor. This keeps classes independent.
        """
        print("\n" + "=" * 55)
        print("  DataDownloader: Downloading HotpotQA")
        print("=" * 55)

        # If file already exists and is large enough, skip download
        if self._already_downloaded():
            return config.HOTPOTQA_FILE

        print("\n📥 Downloading HotpotQA training set...")
        print("   Size: ~540 MB")
        print("   Source: CMU official servers")
        print("   This may take 5-15 minutes depending on your internet.\n")

        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"

        try:
            urllib.request.urlretrieve(
                url,
                config.HOTPOTQA_FILE,
                self._show_progress   # shows % downloaded in terminal
            )
            mb = os.path.getsize(config.HOTPOTQA_FILE) / (1024 * 1024)
            print(f"\n   ✅ Downloaded successfully ({mb:.0f} MB)")
            print(f"   Saved to: {config.HOTPOTQA_FILE}")
            return config.HOTPOTQA_FILE

        except Exception as e:
            print(f"\n   ❌ Download failed: {e}")
            print("   Try opening this URL in your browser to download manually:")
            print(f"   {url}")
            print(f"   Then save the file to: {config.HOTPOTQA_FILE}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS — internal helpers (not called from outside)
    # Convention: methods starting with _ are private
    # ─────────────────────────────────────────────────────────────────────────

    def _already_downloaded(self) -> bool:
        """
        Returns True if the file already exists and is big enough.
        WHY: Prevents re-downloading 540MB every time you run the script.
        """
        if not os.path.exists(config.HOTPOTQA_FILE):
            return False
        mb = os.path.getsize(config.HOTPOTQA_FILE) / (1024 * 1024)
        if mb < 100:   # file too small = probably corrupted
            return False
        print(f"\n✅ HotpotQA already downloaded ({mb:.0f} MB) — skipping")
        return True

    def _show_progress(self, block_num, block_size, total_size):
        """Shows a download progress bar in the terminal."""
        if total_size > 0:
            downloaded = block_num * block_size
            percent    = min(downloaded / total_size * 100, 100)
            mb         = downloaded / (1024 * 1024)
            print(f"\r   {percent:.1f}% — {mb:.0f} MB downloaded",
                  end="", flush=True)


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

    OUTPUT SCHEMA (what every record looks like after this class):
      {
        "doc_id":          unique ID for this QA pair
        "title":           titles of supporting paragraphs joined
        "text":            fused supporting paragraph text
        "source":          always "hotpotqa"
        "question":        the multi-hop question
        "answer":          the gold answer
        "is_multihop":     always True for HotpotQA
        "type":            "bridge" or "comparison"
        "level":           "easy", "medium", or "hard"
        "bridge_entities": list of bridge entity titles
      }

    HOW TO USE:
        preprocessor = Preprocessor()
        records      = preprocessor.process("path/to/hotpot_train.json")
    """

    def __init__(self):
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        print("🧹 Preprocessor initialised")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, file_path: str) -> list:
        """
        Reads HotpotQA file and returns list of clean record dicts.

        Args:
            file_path: path to hotpot_train_v1.1.json

        Returns:
            list of clean record dicts (see schema in class docstring)
        """
        print("\n" + "=" * 55)
        print("  Preprocessor: Cleaning HotpotQA data")
        print("=" * 55)

        # Check file exists
        if not file_path or not os.path.exists(file_path):
            print(f"  ❌ File not found: {file_path}")
            print("     Run DataDownloader.download() first.")
            return []

        # Load the raw JSON
        print(f"\n  📂 Loading: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        print(f"  Total entries in file: {len(raw_data):,}")

        # Apply limit if set (e.g. 200 for testing)
        if config.MAX_HOTPOT_RECORDS:
            raw_data = raw_data[:config.MAX_HOTPOT_RECORDS]
            print(f"  Using first {len(raw_data):,} records "
                  f"(MAX_HOTPOT_RECORDS={config.MAX_HOTPOT_RECORDS})")

        # Process each entry
        records = []
        skipped = 0

        for entry in tqdm(raw_data, desc="  Cleaning"):
            record = self._process_one_entry(entry)
            if record:
                records.append(record)
            else:
                skipped += 1

        # Save processed records to disk for inspection
        self._save_to_disk(records)

        print(f"\n  ✅ Preprocessor complete:")
        print(f"     Clean records : {len(records):,}")
        print(f"     Skipped       : {skipped}")

        return records

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _process_one_entry(self, entry: dict) -> dict:
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

        # Skip if missing essential fields
        if not question or not context:
            return None

        # Step 1: Build a map of title → paragraph text
        # context = [["Title", ["sent0", "sent1", ...]], ...]
        para_map = {}
        for title, sentences in context:
            para_map[title] = self._clean_text(" ".join(sentences))

        # Step 2: Get supporting titles (the 2 gold paragraphs we need)
        # supporting_facts = [["Title1", sent_idx], ["Title2", sent_idx], ...]
        # dict.fromkeys preserves order and removes duplicates
        supporting_titles = list(dict.fromkeys(
            title for title, _ in sup_facts
        ))

        # Fallback: if supporting_facts is empty, use all context titles
        if not supporting_titles:
            supporting_titles = [title for title, _ in context]

        # Step 3: Bridge entity = title of the 2nd supporting paragraph
        # WHY: In "bridge" type questions (88.4% of HotpotQA):
        #   Hop 1 paragraph mentions the bridge entity (e.g. "Christopher Nolan")
        #   Hop 2 paragraph IS about the bridge entity
        #   So the 2nd title = the bridge linking the two hops
        bridge_entities = (
            supporting_titles[1:] if len(supporting_titles) > 1 else []
        )

        # Step 4: Fuse ONLY supporting paragraphs into one context string
        # Label each with its title: "[Film Title] The film was..."
        # WHY LABEL: Helps the LLM know which paragraph each sentence came from
        fused_parts = [
            f"[{title}] {para_map[title]}"
            for title in supporting_titles
            if title in para_map and len(para_map.get(title, "")) > 30
        ]
        fused_text = " ".join(fused_parts)

        # Skip if we couldn't build a meaningful context
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
        import re
        if not text or not isinstance(text, str):
            return ""
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        # Collapse multiple whitespace into single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _save_to_disk(self, records: list):
        """
        Saves processed records to src/data/processed/ as JSONL.
        WHY: So you can inspect the output and debug issues without
             re-running the full preprocessing step every time.
        """
        path = os.path.join(config.PROCESSED_DIR, "hotpotqa_processed.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  💾 Saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST (run this file by itself to test these two classes)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing DataDownloader and Preprocessor...\n")

    # Test DataDownloader
    downloader = DataDownloader()
    path       = downloader.download()

    # Test Preprocessor
    if path:
        preprocessor = Preprocessor()
        records      = preprocessor.process(path)
        print(f"\n✅ Test complete. Got {len(records)} clean records.")
        if records:
            print("\nSample record:")
            sample = records[0]
            print(f"  doc_id  : {sample['doc_id']}")
            print(f"  question: {sample['question'][:80]}...")
            print(f"  answer  : {sample['answer']}")
            print(f"  type    : {sample['type']}")
            print(f"  level   : {sample['level']}")
            print(f"  bridge  : {sample['bridge_entities']}")
