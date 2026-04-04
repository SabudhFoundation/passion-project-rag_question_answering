"""
src/config.py
=============
Central configuration for the entire RAG project.
All teammates import settings from here — never hardcode values.

WHAT IS THIS FILE?
  Think of this as the "settings panel" for the whole project.
  Every class reads its settings from here.
  Change one value here → it updates everywhere automatically.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PINECONE VECTOR DATABASE
# ─────────────────────────────────────────────────────────────────────────────
# How to set your API key (run this in terminal before starting):
#   Windows CMD: set PINECONE_API_KEY=your-key-here
#   Mac/Linux:   export PINECONE_API_KEY=your-key-here

PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "PASTE_YOUR_KEY_HERE")
PINECONE_REGION   = os.getenv("PINECONE_REGION",  "us-east-1")
PINECONE_INDEX    = "rag-baseline"
NAMESPACE_HOTPOT  = "hotpotqa"
NAMESPACE_WIKI    = "wikipedia"

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 → free, CPU-friendly, 384-dimensional vectors
# To upgrade later, just change this line — nothing else needs to change

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# chunk_size=512  → safely under MiniLM's 512-token limit
# chunk_overlap=50 → preserves bridge entities at chunk boundaries

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS  (all relative to project root)
# ─────────────────────────────────────────────────────────────────────────────

SRC_DIR        = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR       = os.path.dirname(SRC_DIR)

RAW_DIR        = os.path.join(SRC_DIR, "data", "raw")
PROCESSED_DIR  = os.path.join(SRC_DIR, "data", "processed")

HOTPOTQA_FILE  = os.path.join(RAW_DIR,       "hotpot_train_v1.1.json")
WIKIPEDIA_FILE = os.path.join(RAW_DIR,       "wikipedia_sample.jsonl")
CHUNKS_FILE    = os.path.join(PROCESSED_DIR, "chunks.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET LIMITS (set to None for full datasets)
# ─────────────────────────────────────────────────────────────────────────────

MAX_HOTPOT_RECORDS = 200   # HotpotQA records to process
MAX_WIKI_ARTICLES  = 500   # Wikipedia articles to download

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

UPSERT_BATCH_SIZE = 100   # vectors per Pinecone request
EMBED_BATCH_SIZE  = 64    # texts per embedding forward pass
TOP_K             = 5     # chunks returned per query (used by retrieval team)
