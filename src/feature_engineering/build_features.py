"""
src/feature_engineering/build_features.py
==========================================
YOUR FILE — INGESTION MODULE (Part 2 of 3)

WHAT THIS FILE CONTAINS:
  Class 1 → Chunker   (splits text into overlapping chunks)
  Class 2 → Embedder  (creates vectors + uploads to Pinecone)

WHERE THIS FITS IN THE PROJECT STRUCTURE:
  feature_engineering/ = "Scripts to turn raw data into features for modeling"
  In NLP/RAG:
    "Features" = vector embeddings of text chunks
    Chunking + Embedding = the feature engineering step
  This is exactly what Chunker and Embedder do.

HOW OTHER FILES USE THIS:
  from src.feature_engineering.build_features import Chunker, Embedder

VIVA EXPLANATION:
  "build_features.py handles the second half of my ingestion pipeline.
   Chunker splits the clean text into 512-character overlapping pieces —
   these are my 'features'. Embedder converts each piece into a 384-number
   vector using MiniLM and uploads everything to Pinecone. The vectors ARE
   the features — they represent the meaning of each text chunk numerically."
"""

import os
import json
import hashlib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    CHUNKS_FILE, PROCESSED_DIR,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    PINECONE_API_KEY, PINECONE_REGION, PINECONE_INDEX,
    NAMESPACE_HOTPOT, NAMESPACE_WIKI,
    UPSERT_BATCH_SIZE, EMBED_BATCH_SIZE,
)

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


# =============================================================================
# CLASS 1: Chunker
# =============================================================================

class Chunker:
    """
    Splits preprocessed records into overlapping text chunks.

    SINGLE RESPONSIBILITY: Only chunks. Nothing else.

    WHY CHUNKING?
      Embedding models read ~100 words at a time.
      Wikipedia articles can be 1000+ words.
      We cut them into 512-character pieces so each piece
      can be individually embedded and searched.

    WHY OVERLAP (50 chars)?
      If a bridge entity (linking word between two hops) falls
      at the boundary of two chunks, the 50-char overlap ensures
      it appears in BOTH neighboring chunks — so retrieval never
      misses the connection.

    Usage:
        chunker = Chunker()
        chunks  = chunker.chunk_records(records)
    """

    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size    = chunk_size    or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP

        # RecursiveCharacterTextSplitter tries to split at:
        # paragraph → sentence → word (natural boundaries first)
        self._splitter = RecursiveCharacterTextSplitter(
            separators      = ["\n\n", "\n", ". ", " "],
            chunk_size      = self.chunk_size,
            chunk_overlap   = self.chunk_overlap,
            length_function = len,
        )
        print(f"✂️  Chunker ready "
              f"(size={self.chunk_size}, overlap={self.chunk_overlap})")

    # ── PUBLIC METHODS ────────────────────────────────────────────────────────

    def chunk_records(self, records: list) -> list:
        """
        Splits all records into chunks with full metadata.
        Saves chunks to src/data/processed/chunks.jsonl
        Returns: list of chunk dicts
        """
        print("\n" + "=" * 55)
        print("  Chunker: Splitting records → chunks")
        print("=" * 55)

        all_chunks = []
        for record in tqdm(records, desc="  Chunking"):
            all_chunks.extend(self._chunk_one(record))

        self._save(all_chunks)

        # Summary
        hpqa = sum(1 for c in all_chunks if c["source"] == "hotpotqa")
        wiki = sum(1 for c in all_chunks if c["source"] == "wikipedia")
        brdg = sum(1 for c in all_chunks if c.get("is_bridge"))

        print(f"\n  ✅ Chunks created:")
        print(f"     HotpotQA : {hpqa:,}")
        print(f"     Wikipedia: {wiki:,}")
        print(f"     Bridge   : {brdg:,}")
        print(f"     Total    : {len(all_chunks):,}")

        return all_chunks

    def get_chunks(self) -> list:
        """
        Loads previously saved chunks from disk.
        Call this to skip re-chunking if it was already done.
        """
        if not os.path.exists(CHUNKS_FILE):
            print("  ⚠️  No chunks.jsonl found. Run chunk_records() first.")
            return []
        chunks = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunks.append(json.loads(line.strip()))
                except Exception:
                    continue
        print(f"  📂 Loaded {len(chunks):,} chunks from disk")
        return chunks

    # ── PRIVATE HELPERS ───────────────────────────────────────────────────────

    def _chunk_one(self, record: dict) -> list:
        """
        Splits one record into chunks with metadata.

        QUESTION-AWARE CHUNKING (HotpotQA):
          Prepend "Q: {question}" before the context text.
          WHY: Shifts the chunk's vector closer to future query vectors
               → improves cosine similarity at retrieval time.
        """
        source  = record["source"]
        bridges = record.get("bridge_entities", [])

        # Prepend question for HotpotQA only
        if source == "hotpotqa" and record.get("question"):
            text = f"Q: {record['question']}\nContext: {record['text']}"
        else:
            text = record["text"]

        chunks = []
        for idx, chunk_text in enumerate(self._splitter.split_text(text)):
            if len(chunk_text.strip()) < 30:
                continue

            # Does this chunk contain a bridge entity?
            low       = chunk_text.lower()
            is_bridge = any(b.lower() in low for b in bridges) if bridges else False

            chunks.append({
                # ── PINECONE METADATA SCHEMA ──────────────────────────────
                # These exact fields appear on your Pinecone dashboard.
                # Each field has a specific purpose at retrieval time.
                "chunk_id":    self._make_id(record["doc_id"], idx, chunk_text),
                "doc_id":      record["doc_id"],      # groups chunks by document
                "chunk_idx":   idx,                   # reading-order position
                "text":        chunk_text,            # raw text for LLM prompt
                "source":      source,                # dataset origin
                "title":       record["title"],       # document title
                "question":    record.get("question", ""),
                "answer":      record.get("answer", ""),
                "is_multihop": record.get("is_multihop", False),
                "is_bridge":   is_bridge,
                "type":        record.get("type", ""),
                "level":       record.get("level", ""),
            })

        return chunks

    def _make_id(self, doc_id: str, idx: int, text: str) -> str:
        """Stable unique ID — same chunk always gets same ID (no duplicates)."""
        return hashlib.md5(
            f"{doc_id}_{idx}_{text[:30]}".encode()
        ).hexdigest()[:20]

    def _save(self, chunks: list):
        """Saves chunks to src/data/processed/chunks.jsonl"""
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\n  💾 Saved → src/data/processed/chunks.jsonl")


# =============================================================================
# CLASS 2: Embedder
# =============================================================================

class Embedder:
    """
    Converts chunks to 384-dim vectors and uploads to Pinecone.

    SINGLE RESPONSIBILITY: Embed + upload. Nothing else.

    WHY EMBEDDINGS?
      Computers can't compare meaning in text directly.
      Embeddings convert text to lists of numbers where
      similar sentences are mathematically close together.
      Pinecone finds the closest vectors to your query vector.

    INTERFACE FOR RETRIEVAL TEAM:
      embed_query()  → converts a user question to a vector
      get_index()    → returns the Pinecone index for searching
      These two methods are the handoff to teammates.

    Usage (ingestion — you):
        embedder = Embedder()
        embedder.embed_and_upload(chunks)

    Usage (retrieval — teammates):
        embedder = Embedder()
        vector   = embedder.embed_query("Where was Einstein born?")
        index    = embedder.get_index()
    """

    def __init__(self):
        # Lazy-loaded: model and index only load when first needed
        self._model = None
        self._index = None
        print("🔢 Embedder ready (model loads on first use)")

    # ── PUBLIC: INGESTION SIDE (your responsibility) ──────────────────────────

    def embed_and_upload(self, chunks: list) -> int:
        """
        Embeds all chunks and uploads to Pinecone.
        Returns: total vectors uploaded.
        """
        print("\n" + "=" * 55)
        print("  Embedder: Generating vectors → Pinecone")
        print("=" * 55)

        if not chunks:
            print("  ❌ No chunks to embed")
            return 0

        # Step 1: Generate embeddings
        print(f"\n  🔢 Embedding {len(chunks):,} chunks...")
        chunks = self._generate_embeddings(chunks)

        # Step 2: Connect to Pinecone
        index  = self._get_index()

        # Step 3: Upload per namespace (dataset separation)
        total  = 0
        for ns in [NAMESPACE_HOTPOT, NAMESPACE_WIKI]:
            ns_chunks = [c for c in chunks if c["source"] == ns]
            if ns_chunks:
                total += self._upsert(index, ns_chunks, ns)

        # Step 4: Confirm
        stats = index.describe_index_stats()
        print(f"\n  ✅ Upload complete!")
        print(f"     Vectors in Pinecone: {stats.get('total_vector_count',0):,}")
        print(f"     Index: {PINECONE_INDEX}")
        print(f"     Namespaces: {NAMESPACE_HOTPOT} | {NAMESPACE_WIKI}")
        return total

    # ── PUBLIC: RETRIEVAL SIDE (teammates call these) ─────────────────────────

    def embed_query(self, query: str) -> list:
        """
        Converts a user query to a 384-dim vector.

        ════════════════════════════════════════════
        HANDOFF METHOD — RETRIEVAL TEAM USES THIS
        ════════════════════════════════════════════

        MUST use the same model as ingestion.
        WHY: Query vector must be in the same mathematical
             space as chunk vectors for similarity to work.
        """
        return self._get_model().encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def get_index(self):
        """
        Returns the live Pinecone index for searching.

        ════════════════════════════════════════════
        HANDOFF METHOD — RETRIEVAL TEAM USES THIS
        ════════════════════════════════════════════
        """
        return self._get_index()

    def get_index_stats(self) -> dict:
        """Returns current Pinecone index statistics."""
        return self._get_index().describe_index_stats()

    # ── PRIVATE HELPERS ───────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Lazy-loads the MiniLM embedding model."""
        if self._model is None:
            print(f"  🤖 Loading model: {EMBEDDING_MODEL}")
            print("     (First run downloads ~80 MB — normal)")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"     ✅ Model loaded "
                  f"({self._model.get_sentence_embedding_dimension()}-dim)")
        return self._model

    def _get_index(self):
        """Lazy-loads and returns Pinecone index."""
        if self._index is None:
            self._index = self._setup_pinecone()
        return self._index

    def _setup_pinecone(self):
        """Connects to Pinecone and creates index if it doesn't exist."""
        if PINECONE_API_KEY == "PASTE_YOUR_KEY_HERE":
            raise ValueError(
                "Pinecone API key not set!\n"
                "Run:  set PINECONE_API_KEY=your-key-here"
            )
        print(f"\n  🌲 Connecting to Pinecone...")
        pc       = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i.name for i in pc.list_indexes()]

        if PINECONE_INDEX not in existing:
            print(f"     Creating index '{PINECONE_INDEX}'...")
            pc.create_index(
                name=PINECONE_INDEX, dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
            )
            print(f"     ✅ Index created")
        else:
            print(f"     ✅ Index '{PINECONE_INDEX}' exists")

        idx   = pc.Index(PINECONE_INDEX)
        stats = idx.describe_index_stats()
        print(f"     Vectors in index: {stats.get('total_vector_count',0):,}")
        return idx

    def _generate_embeddings(self, chunks: list) -> list:
        """Generates normalized 384-dim vectors for all chunk texts."""
        model = self._get_model()
        texts = [c["text"] for c in chunks]
        embs  = model.encode(
            texts,
            batch_size           = EMBED_BATCH_SIZE,
            show_progress_bar    = True,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        )
        for chunk, emb in zip(chunks, embs):
            chunk["embedding"] = emb.tolist()
        return chunks

    def _upsert(self, index, chunks: list, namespace: str) -> int:
        """
        Uploads chunks to Pinecone in batches of 100.
        Uses UPSERT: update if ID exists, insert if new → no duplicates.
        """
        total   = 0
        batches = [chunks[i:i+UPSERT_BATCH_SIZE]
                   for i in range(0, len(chunks), UPSERT_BATCH_SIZE)]
        print(f"\n  📤 Uploading '{namespace}': "
              f"{len(chunks):,} vectors | {len(batches)} batches...")

        for batch in tqdm(batches, desc=f"  [{namespace}]"):
            vectors = [
                (
                    c["chunk_id"],
                    c["embedding"],
                    {   # ── SCHEMA STORED IN PINECONE ──────────────────────
                        "original_text": c["text"][:1000],
                        "source":        c["source"],
                        "doc_id":        c["doc_id"],
                        "chunk_idx":     c["chunk_idx"],
                        "title":         c["title"][:200],
                        "question":      c.get("question","")[:300],
                        "answer":        c.get("answer","")[:200],
                        "is_multihop":   c.get("is_multihop", False),
                        "is_bridge":     c.get("is_bridge", False),
                        "type":          c.get("type",""),
                        "level":         c.get("level",""),
                    }
                )
                for c in batch if "embedding" in c
            ]
            if not vectors:
                continue
            try:
                resp   = index.upsert(vectors=vectors, namespace=namespace)
                total += resp.get("upserted_count", len(vectors))
            except Exception as e:
                print(f"\n  ⚠️  Batch failed: {e} — continuing...")

        print(f"  ✅ '{namespace}': {total:,} vectors uploaded")
        return total


# =============================================================================
# STANDALONE RUN (for testing this file independently)
# =============================================================================

if __name__ == "__main__":
    print("Running build_features.py independently...")

    chunker = Chunker()

    # Try loading existing chunks first
    chunks = chunker.get_chunks()

    if not chunks:
        # Load processed records and chunk them
        import json
        records = []
        for fname in ["hotpotqa_processed.jsonl", "wikipedia_processed.jsonl"]:
            path = os.path.join(PROCESSED_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass
        if records:
            chunks = chunker.chunk_records(records)

    if chunks:
        embedder = Embedder()
        total    = embedder.embed_and_upload(chunks)
        print(f"\n✅ Done. {total:,} vectors in Pinecone.")
    else:
        print("❌ No chunks found. Run pre-processing.py first.")
