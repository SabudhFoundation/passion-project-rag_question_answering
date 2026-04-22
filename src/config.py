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
import dotenv

project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

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

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────
# The retriever combines BM25 (keyword) + Vector (semantic) search.
#
# BM25: catches exact names and IDs that embeddings might miss.
# MMR:  Maximal Marginal Relevance — reduces redundancy in vector results.
# Ensemble: combines both with configurable weights.

BM25_K          = 3           # top-K for BM25 keyword retriever
VECTOR_FETCH_K  = 20          # candidates fetched before MMR re-ranking
MMR_LAMBDA      = 0.7         # 1.0 = pure relevance, 0.0 = pure diversity
ENSEMBLE_WEIGHTS = [0.4, 0.6] # [BM25 weight, Vector weight]

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION — LLM (Groq)
# ─────────────────────────────────────────────────────────────────────────────
# Groq provides free, fast inference for open-source LLMs.
# Get your API key at: https://console.groq.com/keys
#
# Before running, set your key:
#   Windows: set GROQ_API_KEY=gsk_...
#   Or add to .env file in project root

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO APP
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE = "RAG Question Answering - HotpotQA"
APP_PORT  = 7860

