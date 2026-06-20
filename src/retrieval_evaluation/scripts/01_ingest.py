"""
STEP 1: INDEX & INGEST
======================
Creates a Pinecone index named 'rag-bge-m3' and ingests HotpotQA contexts
using BAAI/bge-m3 dense embeddings (1024 dims).

BGE-M3 supports multi-functionality:
  - Dense retrieval (used here for Pinecone)
  - Sparse/lexical retrieval (we use BM25 separately for hybrid)
  - ColBERT late interaction (optional, not used in eval pipeline)

Run: python scripts/01_ingest.py
"""

import os
import json
import time
import unicodedata
import hashlib
from dotenv import load_dotenv
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
INDEX_NAME         = os.environ.get("PINECONE_INDEX_NAME", "rag-bge-m3")
EMBEDDING_DIM      = 1024          # BGE-M3 dense output dimension
BATCH_SIZE         = 64            # Upsert batch size
MAX_DOCS           = 5000          # Number of HotpotQA contexts to ingest
HOTPOTQA_SPLIT     = "train"
CLOUD              = "aws"
REGION             = "us-east-1"


def sanitize_id(text: str) -> str:
    """Create a safe ASCII vector ID from arbitrary text."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    clean = "".join(c if (c.isalnum() or c in "-_") else "_" for c in ascii_text)
    clean = clean[:60]
    suffix = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{clean}_{suffix}" if clean else f"doc_{suffix}"


def build_corpus(dataset, max_docs: int):
    """Extract unique contexts from HotpotQA supporting facts."""
    docs = {}
    for item in tqdm(dataset, desc="Building corpus"):
        if len(docs) >= max_docs:
            break
        qid    = item["id"]
        q_type = item["type"]
        level  = item["level"]
        for title, sentences in zip(
            item["context"]["title"],
            item["context"]["sentences"]
        ):
            passage = " ".join(sentences).strip()
            if not passage:
                continue
            raw_id  = f"{title}_{passage[:40]}"
            safe_id = sanitize_id(raw_id)
            if safe_id not in docs:
                docs[safe_id] = {
                    "id":           safe_id,
                    "text":         passage,
                    "title":        title,
                    "source_qid":   qid,
                    "q_type":       q_type,
                    "level":        level,
                    "char_len":     len(passage),
                    "word_count":   len(passage.split()),
                }
        if len(docs) >= max_docs:
            break
    return list(docs.values())


def create_index(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"[INFO] Index '{INDEX_NAME}' already exists — skipping creation.")
        return
    print(f"[INFO] Creating index '{INDEX_NAME}' ({EMBEDDING_DIM} dims, cosine) ...")
    pc.create_index(
        name      = INDEX_NAME,
        dimension = EMBEDDING_DIM,
        metric    = "cosine",
        spec      = ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    # Wait until ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("  waiting for index to be ready ...")
        time.sleep(5)
    print(f"[INFO] Index '{INDEX_NAME}' is ready.")


def embed_texts(model: BGEM3FlagModel, texts: list[str]) -> np.ndarray:
    """Encode a list of texts with BGE-M3, returning dense vectors."""
    output = model.encode(
        texts,
        batch_size           = 32,
        max_length           = 512,
        return_dense         = True,
        return_sparse        = False,
        return_colbert_vecs  = False,
    )
    return np.array(output["dense_vecs"], dtype="float32")


def main():
    print("=" * 60)
    print("  RAG BGE-M3 — STEP 1: INGEST")
    print("=" * 60)

    # ── Load HotpotQA ───────────────────────────────────────────────────────
    print(f"\n[1/5] Loading HotpotQA ({HOTPOTQA_SPLIT}) ...")
    dataset = load_dataset("hotpot_qa", "distractor", split=HOTPOTQA_SPLIT)
    print(f"      Total QA pairs: {len(dataset)}")

    # ── Build corpus ────────────────────────────────────────────────────────
    print(f"\n[2/5] Building document corpus (max {MAX_DOCS} docs) ...")
    corpus = build_corpus(dataset, MAX_DOCS)
    print(f"      Unique passages collected: {len(corpus)}")

    # Save corpus for use in evaluation
    corpus_path = "results/corpus.json"
    os.makedirs("results", exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"      Corpus saved → {corpus_path}")

    # ── Load BGE-M3 ─────────────────────────────────────────────────────────
    print("\n[3/5] Loading BAAI/bge-m3 model ...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("      Model loaded (fp16).")

    # ── Create Pinecone index ───────────────────────────────────────────────
    print("\n[4/5] Initialising Pinecone ...")
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    create_index(pc)
    index = pc.Index(INDEX_NAME)

    # Check if already populated
    stats = index.describe_index_stats()
    existing_vectors = stats.get("total_vector_count", 0)
    if existing_vectors >= 4500:  # already mostly uploaded
        print(f"      Index already has {existing_vectors} vectors — skipping upsert.")
        return

    # ── Embed & upsert ──────────────────────────────────────────────────────
    print(f"\n[5/5] Embedding & upserting {len(corpus)} passages ...")
    texts  = [d["text"] for d in corpus]
    all_vectors = []

    for i in tqdm(range(0, len(texts), 32), desc="Embedding"):
        batch_texts = texts[i : i + 32]
        vecs        = embed_texts(model, batch_texts)
        for j, vec in enumerate(vecs):
            doc = corpus[i + j]
            all_vectors.append({
                "id":     doc["id"],
                "values": vec.tolist(),
                "metadata": {
                    "text":       doc["text"][:500],   # Pinecone metadata limit
                    "title":      doc["title"],
                    "q_type":     doc["q_type"],
                    "level":      doc["level"],
                    "word_count": doc["word_count"],
                },
            })

    # Upsert in batches
    for i in tqdm(range(0, len(all_vectors), BATCH_SIZE), desc="Upserting"):
        batch = all_vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch, timeout=60)

    time.sleep(5)
    final_stats = index.describe_index_stats()
    print(f"\n✅  Ingestion complete!")
    print(f"    Total vectors in index: {final_stats['total_vector_count']}")


if __name__ == "__main__":
    main()
