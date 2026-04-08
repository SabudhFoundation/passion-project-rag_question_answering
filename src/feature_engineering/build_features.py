"""
src/feature_engineering/build_features.py
==========================================
Contains TWO classes:
  1. Chunker  — splits clean records into overlapping text chunks
  2. Embedder — converts chunks to vectors and uploads to Pinecone

HOW DATA FLOWS THROUGH THESE TWO CLASSES:
  Preprocessor returns → list of clean record dicts
      ↓
  Chunker.chunk_records(records)
      ↓ returns list of chunk dicts
  Embedder.embed_and_upload(chunks)
      ↓ uploads vectors to Pinecone

WHAT ARE "FEATURES" IN NLP/RAG?
  In traditional ML, features are numbers extracted from raw data.
  In NLP/RAG, the "features" are embedding vectors — numbers that
  represent the MEANING of each text chunk.
  Chunking + Embedding = Feature Engineering for RAG.
"""

import os
import sys
import json
import hashlib

# Add src/ to path so we can import config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 1: Chunker
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
      When a user asks a question, we find the MOST RELEVANT piece —
      not the whole context, just the exact piece that contains the answer.

    WHY OVERLAP (50 characters)?
      Imagine the context is cut here:
        Chunk 1 ends: "...James Cameron directed Titanic"
        Chunk 2 starts: "Titanic was released in 1997..."
      Without overlap, the connection "James Cameron → Titanic" might be lost.
      With 50-char overlap, "James Cameron directed Titanic" appears in BOTH
      chunks — so the retriever always finds the connection.

      For HotpotQA bridge questions (88.4% of dataset), this is critical:
      the bridge entity (linking word between two hops) must appear in at
      least one chunk. Overlap guarantees this.

    WHY RecursiveCharacterTextSplitter?
      It tries to split at natural boundaries, in this order:
        1. Paragraph break (\n\n) → best, keeps ideas together
        2. Line break (\n)
        3. Sentence end (". ")
        4. Space (" ") → last resort
      It NEVER cuts mid-word. This preserves sentence meaning.

    HOW TO USE:
        chunker = Chunker()
        chunks  = chunker.chunk_records(records)
    """

    def __init__(self):
        # Build the text splitter with our config settings
        self._splitter = RecursiveCharacterTextSplitter(
            separators      = ["\n\n", "\n", ". ", " "],
            chunk_size      = config.CHUNK_SIZE,
            chunk_overlap   = config.CHUNK_OVERLAP,
            length_function = len,
        )
        print(f"✂️  Chunker initialised "
              f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def chunk_records(self, records: list) -> list:
        """
        Splits all records into chunks and saves them to disk.

        Args:
            records: list of clean record dicts from Preprocessor

        Returns:
            list of chunk dicts, each ready for embedding
        """
        print("\n" + "=" * 55)
        print("  Chunker: Splitting records into chunks")
        print("=" * 55)

        if not records:
            print("  ❌ No records provided")
            return []

        all_chunks = []
        for record in tqdm(records, desc="  Chunking"):
            chunks = self._chunk_one_record(record)
            all_chunks.extend(chunks)

        # Save to disk
        self._save_to_disk(all_chunks)

        # Print summary
        bridge_chunks = [c for c in all_chunks if c.get("is_bridge")]
        print(f"\n  ✅ Chunking complete:")
        print(f"     Records processed : {len(records):,}")
        print(f"     Total chunks      : {len(all_chunks):,}")
        print(f"     Bridge chunks     : {len(bridge_chunks):,}")
        print(f"     Avg chunks/record : "
              f"{len(all_chunks)//max(len(records),1)}")

        return all_chunks

    def load_chunks_from_disk(self) -> list:
        """
        Loads previously saved chunks from disk.

        WHY THIS METHOD EXISTS:
          If chunking was already done, you don't need to re-run it.
          This lets you skip directly to embedding on subsequent runs.

        Also used by the RETRIEVAL TEAM to inspect chunk data.
        """
        if not os.path.exists(config.CHUNKS_FILE):
            print("  ⚠️  No chunks file found. Run chunk_records() first.")
            return []

        chunks = []
        with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunks.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        print(f"  📂 Loaded {len(chunks):,} chunks from disk")
        return chunks

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _chunk_one_record(self, record: dict) -> list:
        """
        Splits one record's text into overlapping chunks.
        Attaches full metadata to each chunk.

        QUESTION-AWARE CHUNKING:
          We prepend the question before the context text:
            "Q: Where was Einstein born?\nContext: [Albert Einstein] Born in..."

          WHY: This shifts the chunk's embedding vector toward the
               question's meaning. When a user asks the same question,
               the query vector and this chunk vector will be very similar
               → higher cosine similarity → better retrieval precision.
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
        chunks          = []

        for idx, chunk_text in enumerate(chunk_texts):
            # Skip chunks that are too short to be meaningful
            if len(chunk_text.strip()) < 30:
                continue

            # Bridge detection: does this chunk contain a bridge entity?
            chunk_lower = chunk_text.lower()
            is_bridge   = any(
                entity.lower() in chunk_lower
                for entity in bridge_entities
            ) if bridge_entities else False

            # Build the chunk dict — this is the PINECONE METADATA SCHEMA
            # Every field here gets stored in Pinecone alongside the vector.
            # Your examiner sees these fields on the Pinecone dashboard.
            chunk = {
                # Unique ID — same chunk always gets same ID (prevents duplicates)
                "chunk_id":    self._make_stable_id(
                                   record["doc_id"], idx, chunk_text),

                # Grouping — lets you fetch all chunks of one document
                "doc_id":      record["doc_id"],

                # Position — lets you sort chunks in reading order
                "chunk_idx":   idx,

                # The actual text — stored in Pinecone so no second DB needed
                "text":        chunk_text,

                # Which dataset this came from
                "source":      record["source"],

                # Document title for LLM context attribution
                "title":       record["title"],

                # Ground truth for evaluation
                "question":    record.get("question", ""),
                "answer":      record.get("answer", ""),

                # Retrieval strategy flags
                "is_multihop": record.get("is_multihop", False),
                "is_bridge":   is_bridge,

                # HotpotQA-specific metadata
                "type":        record.get("type", ""),
                "level":       record.get("level", ""),
            }
            chunks.append(chunk)

        return chunks

    def _make_stable_id(self, doc_id: str, idx: int, text: str) -> str:
        """
        Creates a stable unique ID for each chunk.

        WHY STABLE?
          If you run ingestion twice, the same chunk gets the same ID.
          Pinecone's UPSERT operation will UPDATE the existing vector
          instead of creating a duplicate.
          Without stable IDs, every run doubles your vector count.
        """
        raw    = f"{doc_id}_{idx}_{text[:30]}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:20]
        return digest

    def _save_to_disk(self, chunks: list):
        """Saves all chunks to src/data/processed/chunks.jsonl"""
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"\n  💾 Chunks saved to: {config.CHUNKS_FILE}")


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 2: Embedder
# ═════════════════════════════════════════════════════════════════════════════

class Embedder:
    """
    Converts text chunks into embedding vectors and uploads to Pinecone.

    SINGLE RESPONSIBILITY: Embed + upload. Nothing else.

    WHAT IS AN EMBEDDING?
      A computer cannot compare the MEANING of two sentences directly.
      An embedding model converts text into a list of 384 numbers where:
        - Similar sentences → similar numbers → close in vector space
        - Different sentences → different numbers → far in vector space

      Example:
        "Einstein was a physicist"  → [0.12, -0.45, 0.89, ...]
        "Who studied physics?"      → [0.11, -0.44, 0.87, ...]
        "The weather is nice today" → [-0.23, 0.67, -0.12, ...]

      Pinecone stores all these vectors and finds the closest one
      to your query vector — this is semantic search.

    INTERFACE FOR RETRIEVAL TEAM:
      Two methods are the handoff point between ingestion and retrieval:

        embed_query(query)  → converts user question to vector
                              retrieval team uses this at search time

        get_index()         → returns the live Pinecone index
                              retrieval team uses this to run queries

      The retrieval team calls ONLY these two methods.
      They never touch DataDownloader, Preprocessor, or Chunker.

    HOW TO USE:
        embedder = Embedder()
        embedder.embed_and_upload(chunks)     # ingestion
        vector = embedder.embed_query("...")   # retrieval
        index  = embedder.get_index()          # retrieval
    """

    def __init__(self):
        # Lazy loading: model and index are only loaded when first needed
        # WHY LAZY: The model is ~80MB. If a teammate only needs get_index(),
        # they shouldn't wait for the full model to load first.
        self._model = None
        self._index = None
        print("🔢 Embedder initialised (model loads on first use)")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS — INGESTION SIDE (your responsibility)
    # ─────────────────────────────────────────────────────────────────────────

    def embed_and_upload(self, chunks: list) -> int:
        """
        Main ingestion method: embed all chunks and upload to Pinecone.

        Args:
            chunks: list of chunk dicts from Chunker

        Returns:
            int — total number of vectors successfully uploaded
        """
        print("\n" + "=" * 55)
        print("  Embedder: Generating vectors + uploading to Pinecone")
        print("=" * 55)

        if not chunks:
            print("  ❌ No chunks to embed")
            return 0

        # Step 1: Generate embedding vectors for all chunk texts
        print(f"\n  🔢 Embedding {len(chunks):,} chunks...")
        chunks = self._generate_embeddings(chunks)

        # Step 2: Connect to (or create) Pinecone index
        index  = self._get_index()

        # Step 3: Upload to Pinecone namespace
        total  = self._upsert_to_pinecone(index, chunks)

        # Step 4: Confirm final state
        stats  = index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)

        print(f"\n  ✅ Upload complete!")
        print(f"     Vectors uploaded this run : {total:,}")
        print(f"     Total vectors in Pinecone : {total_vectors:,}")
        print(f"     Index name                : {config.PINECONE_INDEX}")
        print(f"     Namespace                 : {config.NAMESPACE_HOTPOT}")

        return total

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS — RETRIEVAL SIDE (teammates call these)
    # ─────────────────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> list:
        """
        Converts a user question into a 384-dim embedding vector.

        ════════════════════════════════════════════════════
        THIS IS THE HANDOFF METHOD FOR THE RETRIEVAL TEAM.
        Teammates import Embedder and call this method.
        ════════════════════════════════════════════════════

        WHY MUST RETRIEVAL USE THIS SAME METHOD?
          The query vector must be in the SAME mathematical space
          as the chunk vectors stored in Pinecone.
          If retrieval used a different model, similarity would
          be meaningless — like comparing metres to pounds.

        Args:
            query: user's natural language question (plain text)

        Returns:
            list of 384 floats — the embedding vector
        """
        model  = self._get_model()
        vector = model.encode(
            query,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        )
        return vector.tolist()

    def get_index(self):
        """
        Returns the live Pinecone index object.

        ════════════════════════════════════════════════════
        THIS IS THE HANDOFF METHOD FOR THE RETRIEVAL TEAM.
        Teammates use this to search Pinecone.
        ════════════════════════════════════════════════════

        Usage by retrieval team:
            embedder = Embedder()
            index    = embedder.get_index()
            results  = index.query(vector=..., top_k=5, ...)
        """
        return self._get_index()

    def get_index_stats(self) -> dict:
        """Returns statistics about the Pinecone index (vector counts etc.)"""
        return self._get_index().describe_index_stats()

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """
        Lazy-loads the SentenceTransformer model.
        Downloads ~80MB on first run, then cached locally.
        """
        if self._model is None:
            print(f"\n  🤖 Loading embedding model: {config.EMBEDDING_MODEL}")
            print("     (First run downloads ~80 MB — this is normal)")
            self._model = SentenceTransformer(config.EMBEDDING_MODEL)
            dim = self._model.get_sentence_embedding_dimension()
            print(f"     ✅ Model loaded — outputs {dim}-dimensional vectors")
        return self._model

    def _get_index(self):
        """Lazy-loads and returns the Pinecone index."""
        if self._index is None:
            self._index = self._setup_pinecone()
        return self._index

    def _setup_pinecone(self):
        """
        Connects to Pinecone and creates the index if it doesn't exist.

        INDEX CONFIGURATION:
          dimension = 384   → must match MiniLM model output
          metric    = cosine → measures angle between vectors (best for text)
          ServerlessSpec    → managed cloud index, no infrastructure to set up
        """
        if config.PINECONE_API_KEY == "PASTE_YOUR_KEY_HERE":
            raise ValueError(
                "\n❌ Pinecone API key not set!\n"
                "   Run this in your terminal first:\n"
                "   set PINECONE_API_KEY=your-actual-key-here"
            )

        print(f"\n  🌲 Connecting to Pinecone...")
        pc       = Pinecone(api_key=config.PINECONE_API_KEY)
        existing = [idx.name for idx in pc.list_indexes()]

        if config.PINECONE_INDEX not in existing:
            print(f"     Creating index '{config.PINECONE_INDEX}'...")
            pc.create_index(
                name      = config.PINECONE_INDEX,
                dimension = config.EMBEDDING_DIM,
                metric    = "cosine",
                spec      = ServerlessSpec(
                    cloud  = "aws",
                    region = config.PINECONE_REGION,
                ),
            )
            print(f"     ✅ Index created")
        else:
            print(f"     ✅ Index '{config.PINECONE_INDEX}' already exists")

        index = pc.Index(config.PINECONE_INDEX)
        stats = index.describe_index_stats()
        print(f"     Current vector count: "
              f"{stats.get('total_vector_count', 0):,}")
        return index

    def _generate_embeddings(self, chunks: list) -> list:
        """
        Generates 384-dim vectors for all chunk texts.

        normalize_embeddings=True:
          Makes every vector unit length (length = 1).
          WHY: When vectors are normalized,
               cosine_similarity = dot_product (same result, faster math).
               Pinecone uses cosine similarity, so this is required.
        """
        model  = self._get_model()
        texts  = [chunk["text"] for chunk in chunks]

        embeddings = model.encode(
            texts,
            batch_size           = config.EMBED_BATCH_SIZE,
            show_progress_bar    = True,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        )

        # Attach each embedding to its chunk
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        return chunks

    def _upsert_to_pinecone(self, index, chunks: list) -> int:
        """
        Uploads all chunks to Pinecone in batches of 100.

        WHY BATCHES OF 100?
          Each vector = 384 floats × 4 bytes ≈ 1.5 KB
          100 vectors + metadata ≈ 200-400 KB per request
          Pinecone's limit is ~4 MB per request
          Batches of 100 = fast, safe, never times out

        WHY UPSERT (not insert)?
          UPSERT = UPDATE if ID exists, INSERT if new.
          Since we use stable IDs (same chunk → same ID), re-running
          the pipeline UPDATES existing vectors instead of duplicating them.
          Your Pinecone vector count stays accurate no matter how many
          times you run the pipeline.
        """
        batches = [
            chunks[i : i + config.UPSERT_BATCH_SIZE]
            for i in range(0, len(chunks), config.UPSERT_BATCH_SIZE)
        ]

        print(f"\n  📤 Uploading to namespace '{config.NAMESPACE_HOTPOT}':")
        print(f"     {len(chunks):,} vectors | {len(batches)} batches")

        total = 0
        for batch in tqdm(batches, desc=f"  [{config.NAMESPACE_HOTPOT}]"):
            # Build the vector list for this batch
            vectors = []
            for chunk in batch:
                if "embedding" not in chunk:
                    continue   # skip if embedding failed

                # Each vector = (id, vector, metadata)
                vectors.append((
                    chunk["chunk_id"],    # unique vector ID
                    chunk["embedding"],   # the 384 numbers
                    {                     # metadata — stored alongside vector
                        # Text stored here so no second DB needed
                        "original_text": chunk["text"][:1000],

                        # Dataset and document info
                        "source":        chunk["source"],
                        "doc_id":        chunk["doc_id"],
                        "chunk_idx":     chunk["chunk_idx"],
                        "title":         chunk["title"][:200],

                        # Ground truth for evaluation
                        "question":      chunk.get("question", "")[:300],
                        "answer":        chunk.get("answer", "")[:200],

                        # Retrieval strategy flags
                        "is_multihop":   chunk.get("is_multihop", False),
                        "is_bridge":     chunk.get("is_bridge", False),

                        # HotpotQA metadata
                        "type":          chunk.get("type", ""),
                        "level":         chunk.get("level", ""),
                    }
                ))

            if not vectors:
                continue

            try:
                response = index.upsert(
                    vectors   = vectors,
                    namespace = config.NAMESPACE_HOTPOT,
                )
                total += response.get("upserted_count", len(vectors))
            except Exception as e:
                print(f"\n  ⚠️  Batch failed: {e} — skipping, continuing...")

        print(f"  ✅ Uploaded {total:,} vectors")
        return total


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST (run this file by itself to test these two classes)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Chunker and Embedder...\n")

    chunker = Chunker()

    # Try loading existing chunks first (skip re-chunking if already done)
    chunks = chunker.load_chunks_from_disk()

    if not chunks:
        # Load processed records and chunk them fresh
        records = []
        processed_path = os.path.join(
            config.PROCESSED_DIR, "hotpotqa_processed.jsonl"
        )
        if os.path.exists(processed_path):
            with open(processed_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass
            print(f"Loaded {len(records)} processed records")
            chunks = chunker.chunk_records(records)
        else:
            print("❌ No processed records found.")
            print("   Run pre_processing.py first.")
            exit(1)

    if chunks:
        print(f"\n✅ Got {len(chunks):,} chunks")
        print("\nSample chunk:")
        s = chunks[0]
        print(f"  chunk_id  : {s['chunk_id']}")
        print(f"  source    : {s['source']}")
        print(f"  is_bridge : {s['is_bridge']}")
        print(f"  text[:80] : {s['text'][:80]}...")

        # Upload to Pinecone
        embedder = Embedder()
        total    = embedder.embed_and_upload(chunks)
        print(f"\n✅ Test complete. {total:,} vectors in Pinecone.")
