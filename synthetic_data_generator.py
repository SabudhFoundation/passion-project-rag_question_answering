"""
src/evaluation/synthetic_data_generator.py
==========================================
Automated synthetic QA dataset generator for RAG pipeline evaluation.

Pipeline
--------
1.  Load processed chunks from  data/processed/
2.  Embed chunks with SentenceTransformer (reuses the same model as ingest)
3.  Cluster with BERTopic (topic modelling) → semantically coherent groups
4.  For every group of 4-5 similar chunks, call the LLM to produce:
      - 1 easy question  (single-hop, direct fact extraction)
      - 1 medium question (two-step bridge reasoning)
      - 1 hard question   (multi-hop, requires synthesis across 3+ chunks)
    Each question comes with its answer, the chunk IDs that support it,
    and full provenance metadata.
5.  Persist to  data/evaluation/synthetic_qa_<dataset>_<timestamp>.jsonl
    and a companion CSV for quick inspection.

Entry points
------------
    # CLI
    python src/evaluation/synthetic_data_generator.py --dataset hotpotqa --top-clusters 50

    # Programmatic
    from evaluation.synthetic_data_generator import SyntheticDataGenerator
    gen = SyntheticDataGenerator(dataset="hotpotqa")
    gen.run()

Output schema (one JSON array in .json)
--------------------------------------------------
{
  "qid":              "hotpotqa_clu012_hard_0",
  "dataset":          "hotpotqa",
  "cluster_id":       12,
  "difficulty":       "hard",
  "question":         "...",
  "answer":           "...",
  "reasoning_chain":  "...",
  "supporting_chunk_ids":   ["c0", "c3", "c4"],
  "supporting_chunks": [
    {
      "chunk_id":      "c0",
      "source_title":  "...",
      "source_type":   "Wiki",
      "publish_date":  "2023-01-10",
      "chunk_index":   1,
      "total_chunks":  8,
      "text":          "..."
    },
    ...
  ],
  "topic_label":      "Multi-hop QA datasets",
  "topic_keywords":   ["hotpotqa", "multi-hop", "wikipedia", "reasoning"],
  "generated_at":     "2025-04-22T10:31:00",
  "model_used":       "llama-3.3-70b-versatile",
  "generation_ok":    true
}
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Path bootstrap ────────────────────────────────────────────────────────────
_FILE_DIR = Path(__file__).resolve().parent
if _FILE_DIR.name == "evaluation":
    _SRC = _FILE_DIR.parent
    _ROOT = _SRC.parent
else:
    _ROOT = _FILE_DIR
    _SRC = _ROOT / "src"

for p in [str(_SRC), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Third-party (installed in project venv) ───────────────────────────────────
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    _BERTOPIC_AVAILABLE = True
except ImportError:
    _BERTOPIC_AVAILABLE = False

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ── Project imports ───────────────────────────────────────────────────────────
try:
    from logger import get_logger
    log = get_logger(__name__)
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    log = logging.getLogger(__name__)

try:
    import config
    _MODEL_NAME   = getattr(config, "LLM_MODEL",   "llama-3.3-70b-versatile")
    _EMBED_MODEL  = getattr(config, "EMBED_MODEL",  "all-MiniLM-L6-v2")
    _GROQ_API_KEY = getattr(config, "GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    _PROCESSED_DIR = Path(getattr(config, "PROCESSED_DIR", _ROOT / "data" / "processed"))
    _EVAL_DIR      = Path(getattr(config, "EVAL_DIR",      _ROOT / "data" / "evaluation"))
except ImportError:
    _MODEL_NAME    = os.getenv("LLM_MODEL",   "llama-3.3-70b-versatile")
    _EMBED_MODEL   = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    _GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
    _PROCESSED_DIR = _ROOT / "data" / "processed"
    _EVAL_DIR      = _ROOT / "data" / "evaluation"

try:
    from preprocessing_data.pre_processing import Chunker
    _Chunker = Chunker
except ImportError:
    _Chunker = None
    log.warning("Chunker not found — will load raw .jsonl from processed dir.")


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class RawChunk:
    """Normalised chunk as loaded from disk."""
    chunk_id:     str
    source_title: str
    source_type:  str
    publish_date: str
    chunk_index:  int
    total_chunks: int
    text:         str
    dataset:      str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SyntheticQA:
    """One generated QA triple with provenance."""
    qid:                   str
    dataset:               str
    cluster_id:            int
    difficulty:            str           # "easy" | "medium" | "hard"
    question:              str
    answer:                str
    reasoning_chain:       str
    supporting_chunk_ids:  list[str]
    supporting_chunks:     list[dict]    # serialised RawChunk dicts
    topic_label:           str
    topic_keywords:        list[str]
    generated_at:          str
    model_used:            str
    generation_ok:         bool
    error:                 str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALLER  (Groq / OpenAI compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, model: str, temperature: float = 0.4) -> str:
    """
    Call the LLM via Groq or OpenAI SDK.
    Returns the raw text response or raises RuntimeError.
    """
    # ── Try Groq first ────────────────────────────────────────────────────────
    try:
        from groq import Groq
        client = Groq(api_key=_GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        pass  # groq not installed — try openai
    except Exception as e:
        raise RuntimeError(f"Groq call failed: {e}") from e

    # ── Fallback to OpenAI-compatible ─────────────────────────────────────────
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        raise RuntimeError(
            "Neither groq nor openai SDK is installed. "
            "Run: pip install groq   or   pip install openai"
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

_QA_SYSTEM = """You are an expert question-answer pair generator for evaluating
retrieval-augmented generation (RAG) systems.

You will receive {n_chunks} text passages that are semantically related.
Your task is to create ONE question at the requested difficulty level and
its corresponding answer using ONLY information present in the passages.

Difficulty definitions
----------------------
EASY   — Single-hop. The answer is explicitly stated in ONE passage.
         The question tests direct fact retrieval. No inference required.
MEDIUM — Two-hop bridge. The answer requires reading TWO passages and
         connecting an entity or fact from one to a fact in the other.
         One passage alone is not sufficient.
HARD   — Multi-hop synthesis. The answer requires reading THREE OR MORE
         passages and synthesising information across all of them.
         The question should require reasoning, not just extraction.

Output format (strict JSON — no extra text before or after)
-----------------------------------------------------------
{{
  "question": "<the question text>",
  "answer": "<concise but complete answer>",
  "reasoning_chain": "<step-by-step chain showing which passage supports which part>",
  "supporting_chunk_ids": ["<id1>", "<id2>", ...]
}}

Rules
-----
- supporting_chunk_ids must reference ONLY the IDs provided in the input
- The answer must be fully supported by the passages — no hallucination
- The question must NOT be answerable from a single passage for MEDIUM/HARD
- Be specific: include entity names, dates, numbers where appropriate
- reasoning_chain must mention passage IDs explicitly
"""

def _build_qa_prompt(chunks: list[RawChunk], difficulty: str) -> str:
    passages = "\n\n".join(
        f"[Passage {c.chunk_id}]  (Source: {c.source_title}, {c.source_type}, {c.publish_date})\n{c.text}"
        for c in chunks
    )
    return (
        _QA_SYSTEM.format(n_chunks=len(chunks))
        + f"\n\nDifficulty: {difficulty.upper()}\n\n"
        + "Passages:\n"
        + "-" * 60 + "\n"
        + passages
        + "\n" + "-" * 60
        + "\n\nGenerate the JSON now:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSER  — robust extraction from LLM output
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """
    Extract JSON from LLM output that may contain markdown fences,
    leading text, or minor formatting issues.
    """
    # Strip markdown code fences
    clean = re.sub(r"```(?:json)?\s*", "", raw, flags=re.I)
    clean = re.sub(r"```", "", clean)

    # Try direct parse first
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    m = re.search(r"\{[\s\S]*\}", clean)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM output:\n{raw[:400]}")


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_chunks(dataset: str) -> list[RawChunk]:
    """
    Load chunks from disk.
    Priority:
      1. Use Chunker.load_chunks_from_disk() if available
      2. Scan data/processed/<dataset>/*.jsonl  for raw chunk files
      3. Raise FileNotFoundError if neither works
    """
    chunks: list[RawChunk] = []

    # ── Strategy 1: project Chunker ───────────────────────────────────────────
    if _Chunker is not None:
        try:
            raw_chunks = _Chunker().load_chunks_from_disk()
            if raw_chunks:
                log.info("Loaded %d chunks via Chunker.load_chunks_from_disk()", len(raw_chunks))
                for i, rc in enumerate(raw_chunks):
                    text = rc.get("text") or rc.get("page_content") or rc.get("content", "")
                    meta = rc.get("metadata", rc)
                    chunks.append(RawChunk(
                        chunk_id=rc.get("chunk_id", f"c{i}"),
                        source_title=meta.get("title", meta.get("source", f"doc_{i}")),
                        source_type=meta.get("source_type", "Wiki"),
                        publish_date=meta.get("publish_date", meta.get("date", "N/A")),
                        chunk_index=meta.get("chunk_index", i),
                        total_chunks=meta.get("total_chunks", len(raw_chunks)),
                        text=text,
                        dataset=dataset,
                    ))
                return chunks
        except Exception as e:
            log.warning("Chunker.load_chunks_from_disk() failed: %s — falling back to file scan.", e)

    # ── Strategy 2: scan processed directory ──────────────────────────────────
    search_dirs = [
        _PROCESSED_DIR / dataset,
        _PROCESSED_DIR,
        _ROOT / "data" / "processed" / dataset,
        _ROOT / "data" / "processed",
    ]

    chunk_files: list[Path] = []
    for d in search_dirs:
        if d.exists():
            chunk_files = list(d.glob("*.jsonl")) + list(d.glob("chunks*.json"))
            if chunk_files:
                log.info("Found %d chunk file(s) in %s", len(chunk_files), d)
                break

    if not chunk_files:
        raise FileNotFoundError(
            f"No chunk files found for dataset '{dataset}'. "
            f"Searched: {[str(d) for d in search_dirs]}. "
            f"Run the ingest pipeline first."
        )

    for fpath in chunk_files:
        try:
            with fpath.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = (
                        obj.get("text")
                        or obj.get("page_content")
                        or obj.get("content")
                        or ""
                    )
                    if not text:
                        continue
                    meta = obj.get("metadata", obj)
                    i = len(chunks)
                    chunks.append(RawChunk(
                        chunk_id=obj.get("chunk_id", f"c{i}"),
                        source_title=meta.get("title", meta.get("source", f"doc_{i}")),
                        source_type=meta.get("source_type", "Wiki"),
                        publish_date=meta.get("publish_date", meta.get("date", "N/A")),
                        chunk_index=meta.get("chunk_index", i),
                        total_chunks=meta.get("total_chunks", 0),
                        text=text,
                        dataset=dataset,
                    ))
        except Exception as e:
            log.warning("Could not read %s: %s", fpath, e)

    if not chunks:
        raise ValueError(f"Loaded 0 valid chunks for dataset '{dataset}' — check file format.")

    log.info("Loaded %d chunks from disk for dataset '%s'.", len(chunks), dataset)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def _embed_chunks(chunks: list[RawChunk], model_name: str) -> np.ndarray:
    """
    Return (N, D) float32 embedding matrix.
    Uses SentenceTransformer if available, else falls back to TF-IDF.
    """
    texts = [c.text for c in chunks]

    if _ST_AVAILABLE:
        log.info("Embedding %d chunks with SentenceTransformer '%s'…", len(chunks), model_name)
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        log.info("Embeddings shape: %s", embeddings.shape)
        return embeddings.astype(np.float32)

    # ── Fallback: TF-IDF ──────────────────────────────────────────────────────
    log.warning(
        "sentence-transformers not installed — using TF-IDF fallback. "
        "Install with: pip install sentence-transformers"
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=768, sublinear_tf=True, stop_words="english")
    return vec.fit_transform(texts).toarray().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Cluster:
    cluster_id:   int
    chunks:       list[RawChunk]
    topic_label:  str
    topic_keywords: list[str]


def _cluster_chunks(
    chunks: list[RawChunk],
    embeddings: np.ndarray,
    min_cluster_size: int = 4,
    max_cluster_size: int = 6,
    top_clusters: int | None = None,
) -> list[Cluster]:
    """
    Group chunks into semantically coherent clusters of 4-6 chunks each.

    Strategy
    --------
    1. BERTopic (preferred) — topic modelling with semantic clustering.
       Produces human-readable topic labels and keywords automatically.
    2. AgglomerativeClustering fallback — pure cosine-distance clustering
       when BERTopic is not installed. Generates keyword labels via TF-IDF.
    """

    # ── BERTopic path ─────────────────────────────────────────────────────────
    if _BERTOPIC_AVAILABLE:
        return _cluster_bertopic(
            chunks, embeddings, min_cluster_size, max_cluster_size, top_clusters
        )

    # ── Agglomerative fallback ────────────────────────────────────────────────
    log.warning(
        "BERTopic not installed — using AgglomerativeClustering. "
        "Install with: pip install bertopic"
    )
    return _cluster_agglomerative(
        chunks, embeddings, min_cluster_size, max_cluster_size, top_clusters
    )


def _cluster_bertopic(
    chunks, embeddings, min_cluster_size, max_cluster_size, top_clusters
) -> list[Cluster]:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    log.info("Running BERTopic on %d chunks…", len(chunks))

    umap_model = UMAP(
        n_neighbors=min(15, len(chunks) - 1),
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=8,
        verbose=False,
    )

    texts = [c.text for c in chunks]
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Map topic id → list of chunk indices
    topic_map: dict[int, list[int]] = {}
    for idx, topic_id in enumerate(topics):
        if topic_id == -1:
            continue  # outlier — skip
        topic_map.setdefault(topic_id, []).append(idx)

    log.info("BERTopic found %d non-outlier topics.", len(topic_map))

    # Get topic info once
    topic_info_df = topic_model.get_topic_info()

    clusters: list[Cluster] = []
    for cluster_id, (topic_id, indices) in enumerate(topic_map.items()):
        topic_row = topic_info_df[topic_info_df["Topic"] == topic_id]
        label     = topic_row["Name"].values[0] if not topic_row.empty else f"topic_{topic_id}"
        kw_list   = topic_model.get_topic(topic_id) or []
        keywords  = [w for w, _ in kw_list[:6]]

        # Split large clusters into windows of max_cluster_size
        for start in range(0, len(indices), max_cluster_size):
            window = indices[start : start + max_cluster_size]
            if len(window) < min_cluster_size:
                continue
            clusters.append(Cluster(
                cluster_id=len(clusters),
                chunks=[chunks[i] for i in window],
                topic_label=label,
                topic_keywords=keywords,
            ))

    if top_clusters:
        clusters = clusters[:top_clusters]

    log.info("Produced %d usable clusters (size %d–%d).",
             len(clusters), min_cluster_size, max_cluster_size)
    return clusters


def _cluster_agglomerative(
    chunks, embeddings, min_cluster_size, max_cluster_size, top_clusters
) -> list[Cluster]:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(chunks)
    # Target ~n / avg_group_size clusters
    avg = (min_cluster_size + max_cluster_size) // 2
    n_clusters = max(1, n // avg)

    log.info("AgglomerativeClustering: %d chunks → %d clusters", n, n_clusters)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(embeddings)

    # Group indices by label
    groups: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        groups.setdefault(int(lbl), []).append(idx)

    # TF-IDF keywords per group
    texts  = [c.text for c in chunks]
    tfidf  = TfidfVectorizer(max_features=5000, stop_words="english", sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(texts)
    vocab  = tfidf.get_feature_names_out()

    clusters: list[Cluster] = []
    for lbl, indices in groups.items():
        if len(indices) < min_cluster_size:
            continue
        # Average TF-IDF vector for keyword extraction
        group_vec = np.asarray(tfidf_matrix[indices].mean(axis=0)).flatten()
        top_ids   = group_vec.argsort()[::-1][:6]
        keywords  = [vocab[i] for i in top_ids]
        label     = " + ".join(keywords[:3])

        for start in range(0, len(indices), max_cluster_size):
            window = indices[start : start + max_cluster_size]
            if len(window) < min_cluster_size:
                continue
            clusters.append(Cluster(
                cluster_id=len(clusters),
                chunks=[chunks[i] for i in window],
                topic_label=label,
                topic_keywords=keywords,
            ))

    if top_clusters:
        clusters = clusters[:top_clusters]

    log.info("Produced %d usable clusters.", len(clusters))
    return clusters


# ─────────────────────────────────────────────────────────────────────────────
# QA GENERATOR  — one cluster → 3 QA pairs (easy / medium / hard)
# ─────────────────────────────────────────────────────────────────────────────

_DIFFICULTIES = ("easy", "medium", "hard")


def _generate_qa_for_cluster(
    cluster: Cluster,
    dataset: str,
    model: str,
    max_retries: int = 3,
) -> list[SyntheticQA]:
    """
    Call LLM once per difficulty level for this cluster.
    Returns 0–3 SyntheticQA objects (skips on parse/call failure after retries).
    """
    results: list[SyntheticQA] = []
    ts = datetime.utcnow().isoformat(timespec="seconds")

    for diff_idx, difficulty in enumerate(_DIFFICULTIES):
        qid = f"{dataset}_clu{cluster.cluster_id:04d}_{difficulty}_{diff_idx}"
        prompt = _build_qa_prompt(cluster.chunks, difficulty)

        last_error = ""
        ok = False
        parsed: dict = {}

        for attempt in range(1, max_retries + 1):
            try:
                raw    = _call_llm(prompt, model=model, temperature=0.4)
                parsed = _parse_llm_json(raw)
                ok     = True
                break
            except Exception as e:
                last_error = str(e)
                log.warning(
                    "Cluster %d / %s / attempt %d failed: %s",
                    cluster.cluster_id, difficulty, attempt, e,
                )
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)

        # Validate required keys
        if ok:
            for required_key in ("question", "answer", "reasoning_chain", "supporting_chunk_ids"):
                if required_key not in parsed:
                    ok = False
                    last_error = f"Missing key '{required_key}' in LLM response"
                    break

        # Validate chunk IDs are real
        valid_ids = {c.chunk_id for c in cluster.chunks}
        if ok:
            bad_ids = [cid for cid in parsed.get("supporting_chunk_ids", []) if cid not in valid_ids]
            if bad_ids:
                # Soft-fix: remove phantom IDs rather than discarding entirely
                log.warning("Cluster %d: removing phantom chunk IDs %s", cluster.cluster_id, bad_ids)
                parsed["supporting_chunk_ids"] = [
                    cid for cid in parsed["supporting_chunk_ids"] if cid in valid_ids
                ]

        # Build supporting_chunks from validated IDs
        id_to_chunk = {c.chunk_id: c for c in cluster.chunks}
        supporting_chunks = [
            id_to_chunk[cid].to_dict()
            for cid in parsed.get("supporting_chunk_ids", [])
            if cid in id_to_chunk
        ]

        results.append(SyntheticQA(
            qid=qid,
            dataset=dataset,
            cluster_id=cluster.cluster_id,
            difficulty=difficulty,
            question=parsed.get("question", ""),
            answer=parsed.get("answer", ""),
            reasoning_chain=parsed.get("reasoning_chain", ""),
            supporting_chunk_ids=parsed.get("supporting_chunk_ids", []),
            supporting_chunks=supporting_chunks,
            topic_label=cluster.topic_label,
            topic_keywords=cluster.topic_keywords,
            generated_at=ts,
            model_used=model,
            generation_ok=ok,
            error=last_error if not ok else "",
        ))

        # Small pause to respect rate limits
        time.sleep(0.5)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def _write_outputs(
    qa_items: list[SyntheticQA],
    dataset: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write .jsonl and .csv, return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_str  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem    = f"synthetic_qa_{dataset}_{ts_str}"
    json_path = output_dir / f"{stem}.json"
    csv_path  = output_dir / f"{stem}.csv"

    # ── JSON ──────────────────────────────────────────────────────────────────
    with json_path.open("w", encoding="utf-8") as f:
        json_data = [item.to_dict() for item in qa_items]
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    log.info("Wrote %d QA pairs -> %s", len(qa_items), json_path)

    # ── CSV (flat, human-readable) ────────────────────────────────────────────
    import csv
    csv_fields = [
        "qid", "dataset", "cluster_id", "difficulty",
        "question", "answer", "reasoning_chain",
        "supporting_chunk_ids", "topic_label", "topic_keywords",
        "generated_at", "model_used", "generation_ok", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for item in qa_items:
            row = item.to_dict()
            row["supporting_chunk_ids"] = "|".join(row["supporting_chunk_ids"])
            row["topic_keywords"]       = "|".join(row["topic_keywords"])
            writer.writerow(row)
    log.info("Wrote CSV -> %s", csv_path)

    return json_path, csv_path


# ─────────────────────────────────────────────────────────────────────────────
# STATS REPORTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_stats(qa_items: list[SyntheticQA], elapsed: float) -> None:
    total  = len(qa_items)
    ok     = sum(1 for q in qa_items if q.generation_ok)
    failed = total - ok

    by_diff: dict[str, int] = {}
    for q in qa_items:
        by_diff[q.difficulty] = by_diff.get(q.difficulty, 0) + 1

    print("\n" + "=" * 60)
    print("  SYNTHETIC QA GENERATION — SUMMARY")
    print("=" * 60)
    print(f"  Total generated   : {total}")
    print(f"  Successful        : {ok}  ({ok/max(total,1)*100:.1f}%)")
    print(f"  Failed / skipped  : {failed}")
    print()
    for d in ("easy", "medium", "hard"):
        print(f"  {d.capitalize():<8}: {by_diff.get(d, 0)}")
    print()
    print(f"  Elapsed           : {elapsed:.1f}s")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    End-to-end synthetic QA dataset generator.

    Parameters
    ----------
    dataset         : name of the dataset (hotpotqa, cord19, …)
    model           : LLM model name for question generation
    embed_model     : SentenceTransformer model name for embeddings
    min_cluster     : minimum chunks per cluster (default 4)
    max_cluster     : maximum chunks per cluster (default 6)
    top_clusters    : cap on number of clusters to process (None = all)
    output_dir      : where to write output files
    max_retries     : LLM call retries per QA generation attempt
    """

    def __init__(
        self,
        dataset:      str           = "hotpotqa",
        model:        str           = _MODEL_NAME,
        embed_model:  str           = _EMBED_MODEL,
        min_cluster:  int           = 4,
        max_cluster:  int           = 6,
        top_clusters: int | None    = None,
        output_dir:   Path | str    = _EVAL_DIR,
        max_retries:  int           = 3,
    ) -> None:
        self.dataset      = dataset
        self.model        = model
        self.embed_model  = embed_model
        self.min_cluster  = min_cluster
        self.max_cluster  = max_cluster
        self.top_clusters = top_clusters
        self.output_dir   = Path(output_dir)
        self.max_retries  = max_retries

    def run(self) -> tuple[Path, Path]:
        """
        Execute the full pipeline:
          load -> embed -> cluster -> generate -> write -> report

        Returns (json_path, csv_path).
        """
        t_start = time.time()

        log.info("=" * 60)
        log.info("SyntheticDataGenerator starting")
        log.info("  dataset      : %s", self.dataset)
        log.info("  model        : %s", self.model)
        log.info("  embed_model  : %s", self.embed_model)
        log.info("  cluster size : %d - %d", self.min_cluster, self.max_cluster)
        log.info("  top_clusters : %s", self.top_clusters or "all")
        log.info("=" * 60)

        # ── Step 1: Load chunks ───────────────────────────────────────────────
        log.info("STEP 1/4 -- Loading chunks...")
        chunks = _load_chunks(self.dataset)

        # ── Step 2: Embed ─────────────────────────────────────────────────────
        log.info("STEP 2/4 -- Embedding chunks...")
        embeddings = _embed_chunks(chunks, self.embed_model)

        # ── Step 3: Cluster ───────────────────────────────────────────────────
        log.info("STEP 3/4 -- Clustering into semantic groups...")
        clusters = _cluster_chunks(
            chunks, embeddings,
            min_cluster_size=self.min_cluster,
            max_cluster_size=self.max_cluster,
            top_clusters=self.top_clusters,
        )
        log.info("Using %d clusters for QA generation.", len(clusters))

        # ── Step 4: Generate QA ───────────────────────────────────────────────
        log.info("STEP 4/4 -- Generating QA pairs (3 per cluster)...")
        all_qa: list[SyntheticQA] = []

        for clu_idx, cluster in enumerate(clusters):
            log.info(
                "  Cluster %d/%d  |  topic: %s  |  chunks: %d",
                clu_idx + 1, len(clusters), cluster.topic_label[:50], len(cluster.chunks),
            )
            qa_list = _generate_qa_for_cluster(
                cluster=cluster,
                dataset=self.dataset,
                model=self.model,
                max_retries=self.max_retries,
            )
            all_qa.extend(qa_list)

            ok_count = sum(1 for q in qa_list if q.generation_ok)
            log.info("    -> generated %d/3 QA pairs (%d ok)", len(qa_list), ok_count)

        # ── Write outputs ─────────────────────────────────────────────────────
        json_path, csv_path = _write_outputs(all_qa, self.dataset, self.output_dir)

        # ── Report ────────────────────────────────────────────────────────────
        _print_stats(all_qa, elapsed=time.time() - t_start)

        return json_path, csv_path

    def resume_from(self, resume_file: Path) -> tuple[Path, Path]:
        """
        Resume generation from a previous output file.
        Loads existing results, identifies failed clusters,
        re-generates only those, and merges into a new output.

        Returns (json_path, csv_path).
        """
        t_start = time.time()

        # ── Load previous results ─────────────────────────────────────────────
        log.info("=" * 60)
        log.info("RESUME MODE -- loading previous results from:")
        log.info("  %s", resume_file)
        log.info("=" * 60)

        with resume_file.open("r", encoding="utf-8") as f:
            prev_data = json.load(f)

        # Separate successful and failed entries
        successful_entries = [e for e in prev_data if e.get("generation_ok", False)]
        failed_entries     = [e for e in prev_data if not e.get("generation_ok", False)]

        # Find unique cluster IDs that had failures
        failed_cluster_ids = sorted(set(e["cluster_id"] for e in failed_entries))

        log.info("  Previous total   : %d QA pairs", len(prev_data))
        log.info("  Already OK       : %d", len(successful_entries))
        log.info("  Failed (to redo) : %d across %d clusters",
                 len(failed_entries), len(failed_cluster_ids))

        if not failed_cluster_ids:
            log.info("  Nothing to resume -- all entries are successful!")
            return resume_file, resume_file.with_suffix(".csv")

        # ── Re-cluster to get the same clusters ───────────────────────────────
        log.info("STEP 1/3 -- Loading and re-clustering chunks...")
        chunks     = _load_chunks(self.dataset)
        embeddings = _embed_chunks(chunks, self.embed_model)
        clusters   = _cluster_chunks(
            chunks, embeddings,
            min_cluster_size=self.min_cluster,
            max_cluster_size=self.max_cluster,
            top_clusters=self.top_clusters,
        )

        # Filter to only the clusters that need regeneration
        clusters_to_redo = [
            c for i, c in enumerate(clusters) if i in failed_cluster_ids
        ]
        log.info("  Found %d clusters to regenerate.", len(clusters_to_redo))

        # ── Regenerate only failed clusters ───────────────────────────────────
        log.info("STEP 2/3 -- Regenerating QA pairs for failed clusters...")
        new_qa: list[SyntheticQA] = []

        for clu_idx, cluster in enumerate(clusters_to_redo):
            log.info(
                "  Cluster %d/%d (id=%d)  |  topic: %s  |  chunks: %d",
                clu_idx + 1, len(clusters_to_redo),
                cluster.cluster_id, cluster.topic_label[:50], len(cluster.chunks),
            )
            qa_list = _generate_qa_for_cluster(
                cluster=cluster,
                dataset=self.dataset,
                model=self.model,
                max_retries=self.max_retries,
            )
            new_qa.extend(qa_list)

            ok_count = sum(1 for q in qa_list if q.generation_ok)
            log.info("    -> generated %d/3 QA pairs (%d ok)", len(qa_list), ok_count)

        # ── Merge: keep old successes + replace failures with new results ─────
        log.info("STEP 3/3 -- Merging results...")

        # Convert successful old entries back to SyntheticQA objects
        merged_qa: list[SyntheticQA] = []
        for entry in successful_entries:
            merged_qa.append(SyntheticQA(
                qid=entry["qid"],
                dataset=entry["dataset"],
                cluster_id=entry["cluster_id"],
                difficulty=entry["difficulty"],
                question=entry["question"],
                answer=entry["answer"],
                reasoning_chain=entry["reasoning_chain"],
                supporting_chunk_ids=entry["supporting_chunk_ids"],
                supporting_chunks=entry.get("supporting_chunks", []),
                topic_label=entry["topic_label"],
                topic_keywords=entry["topic_keywords"],
                generated_at=entry["generated_at"],
                model_used=entry["model_used"],
                generation_ok=entry["generation_ok"],
                error=entry.get("error", ""),
            ))

        # Add newly generated results
        merged_qa.extend(new_qa)

        # Sort by cluster_id then difficulty for clean output
        diff_order = {"easy": 0, "medium": 1, "hard": 2}
        merged_qa.sort(key=lambda q: (q.cluster_id, diff_order.get(q.difficulty, 9)))

        # ── Write merged outputs ──────────────────────────────────────────────
        json_path, csv_path = _write_outputs(merged_qa, self.dataset, self.output_dir)

        # ── Report ────────────────────────────────────────────────────────────
        _print_stats(merged_qa, elapsed=time.time() - t_start)

        new_ok = sum(1 for q in new_qa if q.generation_ok)
        log.info("  Resumed: %d old OK + %d new OK = %d total OK",
                 len(successful_entries), new_ok, len(successful_entries) + new_ok)

        return json_path, csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic QA evaluation data for a RAG pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="hotpotqa",
        help="Dataset name — must match a folder or config in data/processed/",
    )
    parser.add_argument(
        "--model",
        default=_MODEL_NAME,
        help="LLM model name for question generation",
    )
    parser.add_argument(
        "--embed-model",
        default=_EMBED_MODEL,
        help="SentenceTransformer model for chunk embeddings",
    )
    parser.add_argument(
        "--min-cluster", type=int, default=4,
        help="Minimum number of chunks per cluster",
    )
    parser.add_argument(
        "--max-cluster", type=int, default=6,
        help="Maximum number of chunks per cluster",
    )
    parser.add_argument(
        "--top-clusters", type=int, default=None,
        help="Process only the first N clusters (useful for testing)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_EVAL_DIR),
        help="Directory to write synthetic QA files",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="LLM call retries per QA pair",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a previous output JSON file. Only failed clusters will be regenerated.",
    )
    args = parser.parse_args()

    gen = SyntheticDataGenerator(
        dataset=args.dataset,
        model=args.model,
        embed_model=args.embed_model,
        min_cluster=args.min_cluster,
        max_cluster=args.max_cluster,
        top_clusters=args.top_clusters,
        output_dir=Path(args.output_dir),
        max_retries=args.max_retries,
    )

    if args.resume:
        json_path, csv_path = gen.resume_from(Path(args.resume))
    else:
        json_path, csv_path = gen.run()
    print(f"\nOutputs written:\n  JSON  : {json_path}\n  CSV   : {csv_path}\n")


if __name__ == "__main__":
    _cli()
