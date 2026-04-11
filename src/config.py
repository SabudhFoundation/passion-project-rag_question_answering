"""
src/config.py
=============
Central configuration file for the RAG project.

WHY THIS FILE EXISTS:
  All settings live here. No class has hardcoded values.
  Change one thing here → the whole project uses the new value.

HOW TO USE IN ANY FILE:
  import sys, os
  sys.path.append(...)
  from config import CHUNK_SIZE, PINECONE_INDEX
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PINECONE
# Set API key in terminal before running:
#   Windows : set PINECONE_API_KEY=your-key-here
#   Mac/Linux: export PINECONE_API_KEY=your-key-here
# ─────────────────────────────────────────────────────────────────────────────

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "PASTE_YOUR_KEY_HERE")
PINECONE_REGION  = os.getenv("PINECONE_REGION",  "us-east-1")
PINECONE_INDEX   = "rag-baseline"
NAMESPACE_HOTPOT = "hotpotqa"

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# all-MiniLM-L6-v2 → free, CPU-friendly, 384-dimensional output
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# chunk_size=512  → under MiniLM's 512-token limit
# chunk_overlap=50 → preserves bridge entities at boundaries
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────

SRC_DIR       = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(SRC_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(SRC_DIR, "data", "processed")

HOTPOTQA_URL  = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
HOTPOTQA_FILE = os.path.join(RAW_DIR,       "hotpot_train_v1.1.json")
CHUNKS_FILE   = os.path.join(PROCESSED_DIR, "chunks.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET LIMITS
# Set MAX_HOTPOT_RECORDS = None for full dataset (90,447 records)
# Keep at 200 during testing to finish in ~5 minutes
# ─────────────────────────────────────────────────────────────────────────────

MAX_HOTPOT_RECORDS = 200

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

UPSERT_BATCH_SIZE = 100   # vectors per Pinecone API request (max safe = 100)
EMBED_BATCH_SIZE  = 64    # texts per embedding model forward pass
TOP_K             = 5     # chunks returned per query (used by retrieval team)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_LEVEL  = "INFO"
