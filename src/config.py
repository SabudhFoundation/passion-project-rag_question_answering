"""
src/config.py
=============
Central settings file for the entire RAG project.

WHY THIS FILE EXISTS:
  Instead of writing your API key, file paths, or model names
  inside every class, we put all settings HERE in one place.

  Benefit: If you want to change the embedding model, you change
  ONE line here — every class automatically uses the new value.

HOW TO USE:
  from config import PINECONE_API_KEY, CHUNK_SIZE
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PINECONE — Vector Database
# ─────────────────────────────────────────────────────────────────────────────
# Before running, set your key in the terminal:
#   Windows: set PINECONE_API_KEY=your-key-here
#   Mac/Linux: export PINECONE_API_KEY=your-key-here

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "PASTE_YOUR_KEY_HERE")
PINECONE_REGION  = os.getenv("PINECONE_REGION",  "us-east-1")
PINECONE_INDEX   = "rag-baseline"
NAMESPACE_HOTPOT = "hotpotqa"   # namespace inside Pinecone index

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2:
#   - Free to use, no API key needed
#   - Runs on CPU (no GPU required)
#   - Outputs 384-dimensional vectors
#   - Industry standard for RAG baseline systems

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384   # must match the model AND the Pinecone index

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
# chunk_size = 512 characters (~100-120 words)
#   WHY: MiniLM has a 512-token limit. 512 chars keeps us safely under it.
#
# chunk_overlap = 50 characters
#   WHY: HotpotQA has 88.4% bridge-type questions.
#        The "bridge" entity connects two paragraphs.
#        If it falls at a chunk boundary, overlap ensures it appears
#        in BOTH neighboring chunks so retrieval never misses it.

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
# os.path.dirname(__file__) = the folder where this config.py lives (src/)
# We build all paths relative to src/ so the project works on any computer.

SRC_DIR       = os.path.dirname(os.path.abspath(__file__))

RAW_DIR       = os.path.join(SRC_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(SRC_DIR, "data", "processed")

# HotpotQA — the ONLY dataset used in this pipeline
HOTPOTQA_FILE = os.path.join(RAW_DIR,       "hotpot_train_v1.1.json")
CHUNKS_FILE   = os.path.join(PROCESSED_DIR, "chunks.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET LIMITS
# ─────────────────────────────────────────────────────────────────────────────
# Set to None to process ALL records (full run takes ~30 min)
# Keep at 200 for testing (finishes in ~5 min)

MAX_HOTPOT_RECORDS = 200

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

UPSERT_BATCH_SIZE = 100  # vectors per Pinecone API request (max safe = 100)
EMBED_BATCH_SIZE  = 64   # texts processed per embedding model forward pass

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL (used by retrieval team — do not change)
# ─────────────────────────────────────────────────────────────────────────────

TOP_K = 5   # how many chunks to return per query
