"""
STEP 2: EVALUATE
================
Runs 6 retrieval pipelines on HotpotQA queries and computes:
  Precision@K | Recall@K | Hit@K | NDCG@K | MRR@K   for K = 3, 5, 10

Pipelines evaluated:
  1. dense            — BGE-M3 dense embedding → Pinecone
  2. bm25             — BM25 lexical retrieval
  3. hybrid           — BM25 + Dense score fusion (alpha=0.6)
  4. hybrid_reranked  — Hybrid + cross-encoder reranking
  5. query_expansion  — Short/vague query ko expand karke retrieve
  6. query_splitting  — Long/complex query ko sub-queries mein split karke retrieve

Run: python scripts/02_evaluate.py
"""

import os
import json
import math
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from pinecone import Pinecone
from tqdm import tqdm
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY        = os.environ["GROQ_API_KEY"]
INDEX_NAME          = os.environ.get("PINECONE_INDEX_NAME", "rag-bge-m3")

TOP_K               = 10
RERANK_TOP_N        = 20
ALPHA               = 0.6
EVAL_QUERIES        = 500
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL          = "llama-3.1-8b-instant"

# Query expansion/splitting thresholds
SHORT_QUERY_WORDS   = 6    # 6 words se kam = expand karo
LONG_QUERY_WORDS    = 12   # 12 words se zyada = split karo

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

K_VALUES = [3, 5, 10]


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def precision_at_k(ranked_ids, relevant_ids, k):
    return sum(1 for r in ranked_ids[:k] if r in relevant_ids) / k if k > 0 else 0.0

def recall_at_k(ranked_ids, relevant_ids, k):
    hits = sum(1 for r in ranked_ids[:k] if r in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0

def hit_at_k(ranked_ids, relevant_ids, k):
    return float(any(r in relevant_ids for r in ranked_ids[:k]))

def ndcg_at_k(ranked_ids, relevant_ids, k):
    dcg  = sum(1.0/math.log2(i+2) for i,r in enumerate(ranked_ids[:k]) if r in relevant_ids)
    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(relevant_ids), k)))
    return dcg/idcg if idcg > 0 else 0.0

def mrr_at_k(ranked_ids, relevant_ids, k):
    for i, r in enumerate(ranked_ids[:k]):
        if r in relevant_ids:
            return 1.0/(i+1)
    return 0.0

def compute_metrics(ranked_ids, relevant_ids):
    m = {}
    for k in K_VALUES:
        m[f"precision@{k}"] = precision_at_k(ranked_ids, relevant_ids, k)
        m[f"recall@{k}"]    = recall_at_k(ranked_ids, relevant_ids, k)
        m[f"hit@{k}"]       = hit_at_k(ranked_ids, relevant_ids, k)
        m[f"ndcg@{k}"]      = ndcg_at_k(ranked_ids, relevant_ids, k)
        m[f"mrr@{k}"]       = mrr_at_k(ranked_ids, relevant_ids, k)
    return m

def aggregate(metric_list):
    return {k: round(float(np.mean([m[k] for m in metric_list])), 4) for k in metric_list[0]}


# ═══════════════════════════════════════════════════════════════════════════
#  GROQ HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def paraphrase_query(question, groq_client):
    """Paraphrase a query using Groq."""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a paraphrasing assistant. "
                        "Rephrase the given question in different words while keeping the exact same meaning. "
                        "Return ONLY the rephrased question, nothing else."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=128,
            temperature=0.7,
        )
        p = response.choices[0].message.content.strip()
        return p if p else question
    except Exception as e:
        print(f"  [WARN] Groq paraphrase failed: {e}")
        return question


def expand_query(question, groq_client):
    """
    QUERY EXPANSION:
    Short ya vague query ko expand karo with more context, synonyms, related terms.
    Goal: Better retrieval by giving more information to search engine.

    Example:
        Input:  "Einstein theory"
        Output: "Albert Einstein physicist special general theory of relativity
                 E=mc2 Nobel Prize 1921 spacetime physics"
    """
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query expansion assistant. "
                        "Expand a short or vague search query into a more detailed query. "
                        "Add relevant context, synonyms, and related terms. "
                        "Return ONLY the expanded query as a single paragraph. "
                        "No explanation, no bullets, no extra text. Max 50 words."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Expand this search query: {question}"
                },
            ],
            max_tokens=150,
            temperature=0.5,
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else question
    except Exception as e:
        print(f"  [WARN] Groq expansion failed: {e}")
        return question


def split_query(question, groq_client):
    """
    QUERY SPLITTING:
    Complex multi-part query ko 2-3 simpler sub-queries mein tod do.
    Each sub-query ek specific aspect cover kare.
    Returns list of sub-queries.

    Example:
        Input:  "Who was the director of the 1994 film that won Best Picture
                 and starred Tom Hanks as a slow-witted man?"
        Output: ["What 1994 film won Academy Award Best Picture?",
                 "Which film starred Tom Hanks as a slow-witted man?",
                 "Who directed the Best Picture winner of 1994?"]
    """
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query decomposition assistant. "
                        "Break down a complex multi-part question into 2-3 simpler sub-questions. "
                        "Each sub-question should focus on ONE specific aspect. "
                        "Return ONLY the sub-questions, one per line, no numbering, no extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Break this complex question into sub-questions: {question}"
                },
            ],
            max_tokens=200,
            temperature=0.3,
        )
        result     = response.choices[0].message.content.strip()
        sub_queries = [q.strip() for q in result.split('\n') if q.strip()]
        sub_queries = [q for q in sub_queries if len(q) > 5][:3]
        return sub_queries if sub_queries else [question]
    except Exception as e:
        print(f"  [WARN] Groq splitting failed: {e}")
        return [question]


def should_expand(question):
    """Short query check — 6 words ya kam."""
    return len(question.split()) <= SHORT_QUERY_WORDS


def should_split(question):
    """Long query check — 12 words ya zyada."""
    return len(question.split()) >= LONG_QUERY_WORDS


def merge_results_rrf(results_list, k=TOP_K):
    """
    Reciprocal Rank Fusion (RRF) — Multiple result lists ko merge karo.

    Formula: RRF_score(doc) = sum over all lists of [ 1 / (60 + rank) ]

    60 is standard RRF constant (from original paper).
    Higher total score = better combined rank.

    Example:
        List1: [doc_A=rank1, doc_B=rank2, doc_C=rank3]
        List2: [doc_B=rank1, doc_A=rank3, doc_D=rank2]

        doc_A score = 1/(60+1) + 1/(60+3) = 0.01613 + 0.01587 = 0.03200
        doc_B score = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01613 = 0.03225
        doc_B wins!
    """
    rrf_scores = {}
    for results in results_list:
        for rank, doc_id in enumerate(results):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (60 + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [did for did, _ in ranked[:k]]


# ═══════════════════════════════════════════════════════════════════════════
#  RETRIEVAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def dense_retrieve(query, model, index, k=TOP_K):
    out  = model.encode([query], batch_size=1, max_length=256,
                         return_dense=True, return_sparse=False, return_colbert_vecs=False)
    qvec = out["dense_vecs"][0].tolist()
    res  = index.query(vector=qvec, top_k=k, include_metadata=True)
    return [m["id"] for m in res["matches"]]


def bm25_retrieve(query, bm25, corpus_ids, k=TOP_K):
    scores  = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    return [corpus_ids[i] for i in top_idx]


def hybrid_retrieve(query, model, index, bm25, corpus_ids, corpus_texts,
                    alpha=ALPHA, k=TOP_K, pool=50):
    # Dense
    out  = model.encode([query], batch_size=1, max_length=256,
                         return_dense=True, return_sparse=False, return_colbert_vecs=False)
    qvec = out["dense_vecs"][0].tolist()
    dense_res    = index.query(vector=qvec, top_k=pool, include_metadata=True)
    dense_scores = {m["id"]: m["score"] for m in dense_res["matches"]}

    # BM25
    bm25_raw    = bm25.get_scores(query.lower().split())
    top_idx     = np.argsort(bm25_raw)[::-1][:pool]
    bm25_scores = {corpus_ids[i]: bm25_raw[i] for i in top_idx}

    all_ids = set(dense_scores) | set(bm25_scores)

    def minmax(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        denom  = mx - mn if mx != mn else 1e-9
        return {k: (v-mn)/denom for k,v in d.items()}

    dn = minmax(dense_scores) if dense_scores else {}
    bn = minmax(bm25_scores)  if bm25_scores  else {}

    fused  = {did: alpha*dn.get(did,0) + (1-alpha)*bn.get(did,0) for did in all_ids}
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [did for did,_ in ranked[:k]]


def hybrid_reranked_retrieve(query, model, index, bm25, corpus_ids, corpus_texts,
                              cross_encoder, rerank_top_n=RERANK_TOP_N, k=TOP_K):
    candidates = hybrid_retrieve(query, model, index, bm25, corpus_ids, corpus_texts,
                                  k=rerank_top_n, pool=rerank_top_n*2)
    pairs      = [(query, corpus_texts.get(c, "")) for c in candidates]
    ce_scores  = cross_encoder.predict(pairs)
    reranked   = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [did for did,_ in reranked[:k]]


def query_expansion_retrieve(question, model, index, bm25, corpus_ids, corpus_texts,
                              cross_encoder, groq_client, k=TOP_K):
    """
    QUERY EXPANSION PIPELINE:

    Step 1: Groq se expanded query lo
    Step 2: Original query se hybrid retrieve
    Step 3: Expanded query se hybrid retrieve
    Step 4: RRF se dono merge karo
    Step 5: Cross-encoder se rerank karo
    Step 6: Top-K return karo
    """
    # Step 1: Expand
    expanded = expand_query(question, groq_client)
    time.sleep(0.1)

    # Step 2 & 3: Retrieve from both
    original_results = hybrid_retrieve(
        question, model, index, bm25, corpus_ids, corpus_texts,
        k=RERANK_TOP_N, pool=RERANK_TOP_N*2
    )
    expanded_results = hybrid_retrieve(
        expanded, model, index, bm25, corpus_ids, corpus_texts,
        k=RERANK_TOP_N, pool=RERANK_TOP_N*2
    )

    # Step 4: RRF merge
    merged = merge_results_rrf([original_results, expanded_results], k=RERANK_TOP_N*2)

    # Step 5: Rerank
    candidates = merged[:RERANK_TOP_N]
    pairs      = [(question, corpus_texts.get(c, "")) for c in candidates]
    ce_scores  = cross_encoder.predict(pairs)
    reranked   = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [did for did,_ in reranked[:k]]


def query_splitting_retrieve(question, model, index, bm25, corpus_ids, corpus_texts,
                              cross_encoder, groq_client, k=TOP_K):
    """
    QUERY SPLITTING PIPELINE:

    Step 1: Groq se sub-queries lo (2-3 sub-questions)
    Step 2: Original + har sub-query se alag alag hybrid retrieve
    Step 3: RRF se sab merge karo
    Step 4: Cross-encoder se rerank karo
    Step 5: Top-K return karo
    """
    # Step 1: Split
    sub_queries = split_query(question, groq_client)
    time.sleep(0.1)

    # Original query bhi include karo
    all_queries = [question] + sub_queries

    # Step 2: Retrieve for each
    all_results = []
    for sq in all_queries:
        results = hybrid_retrieve(
            sq, model, index, bm25, corpus_ids, corpus_texts,
            k=RERANK_TOP_N, pool=RERANK_TOP_N*2
        )
        all_results.append(results)

    # Step 3: RRF merge
    merged = merge_results_rrf(all_results, k=RERANK_TOP_N*2)

    # Step 4: Rerank
    candidates = merged[:RERANK_TOP_N]
    pairs      = [(question, corpus_texts.get(c, "")) for c in candidates]
    ce_scores  = cross_encoder.predict(pairs)
    reranked   = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [did for did,_ in reranked[:k]]


# ═══════════════════════════════════════════════════════════════════════════
#  RELEVANCE MAP
# ═══════════════════════════════════════════════════════════════════════════

def build_relevance_map(dataset, corpus_ids_set, max_q):
    import unicodedata
    relevance = {}; count = 0
    for item in dataset:
        if count >= max_q: break
        sup_titles = set(item["supporting_facts"]["title"])
        rel_ids    = set()
        for cid in corpus_ids_set:
            for title in sup_titles:
                norm  = unicodedata.normalize("NFKD", title).encode("ascii","ignore").decode("ascii")
                clean = "".join(c if (c.isalnum() or c in "-_") else "_" for c in norm)[:60]
                if cid.startswith(clean):
                    rel_ids.add(cid)
        if rel_ids:
            relevance[item["id"]] = {
                "question":     item["question"],
                "relevant_ids": rel_ids,
                "q_type":       item["type"],
                "word_count":   len(item["question"].split()),
            }
            count += 1
    return relevance


# ═══════════════════════════════════════════════════════════════════════════
#  EVAL RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_eval(queries, embed_model, index, bm25, corpus_ids, corpus_texts,
             cross_encoder, groq_client, label):

    pipelines = {
        "dense":           [],
        "bm25":            [],
        "hybrid":          [],
        "hybrid_reranked": [],
        "query_expansion": [],
        "query_splitting": [],
    }

    for q in tqdm(queries, desc=f"Evaluating [{label}]"):
        question = q["question"]
        relevant = q["relevant_ids"]

        # 1. Dense
        pipelines["dense"].append(
            compute_metrics(dense_retrieve(question, embed_model, index), relevant))

        # 2. BM25
        pipelines["bm25"].append(
            compute_metrics(bm25_retrieve(question, bm25, corpus_ids), relevant))

        # 3. Hybrid
        pipelines["hybrid"].append(
            compute_metrics(hybrid_retrieve(
                question, embed_model, index, bm25, corpus_ids, corpus_texts), relevant))

        # 4. Hybrid + Reranked
        pipelines["hybrid_reranked"].append(
            compute_metrics(hybrid_reranked_retrieve(
                question, embed_model, index, bm25, corpus_ids,
                corpus_texts, cross_encoder), relevant))

        # 5. Query Expansion
        pipelines["query_expansion"].append(
            compute_metrics(query_expansion_retrieve(
                question, embed_model, index, bm25, corpus_ids,
                corpus_texts, cross_encoder, groq_client), relevant))

        # 6. Query Splitting
        pipelines["query_splitting"].append(
            compute_metrics(query_splitting_retrieve(
                question, embed_model, index, bm25, corpus_ids,
                corpus_texts, cross_encoder, groq_client), relevant))

        time.sleep(0.05)

    return {name: aggregate(ml) for name, ml in pipelines.items()}


def print_results_table(results, label):
    pipeline_labels = {
        "dense":           "Dense (BGE-M3)",
        "bm25":            "BM25 (Lexical)",
        "hybrid":          "Hybrid (a=0.6)",
        "hybrid_reranked": "Hybrid+Reranked",
        "query_expansion": "Query Expansion",
        "query_splitting": "Query Splitting",
    }
    print(f"\n{'='*110}")
    print(f"  RESULTS — {label}")
    print(f"  [P=Precision  R=Recall  H=Hit  N=NDCG  M=MRR]")
    print(f"{'='*110}")
    hdr = f"{'Pipeline':<22}"
    for k in K_VALUES:
        hdr += f"  {'P@'+str(k):>7} {'R@'+str(k):>7} {'H@'+str(k):>7} {'N@'+str(k):>7} {'M@'+str(k):>7}"
    print(hdr)
    print("-"*len(hdr))
    for p, m in results.items():
        row = f"{pipeline_labels.get(p,p):<22}"
        for k in K_VALUES:
            row += (f"  {m[f'precision@{k}']:>7.4f}"
                    f" {m[f'recall@{k}']:>7.4f}"
                    f" {m[f'hit@{k}']:>7.4f}"
                    f" {m[f'ndcg@{k}']:>7.4f}"
                    f" {m[f'mrr@{k}']:>7.4f}")
        print(row)
    print("="*len(hdr))


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  RAG BGE-M3 — STEP 2: EVALUATE")
    print("  Pipelines: Dense | BM25 | Hybrid | Hybrid+Reranked |")
    print("             Query Expansion | Query Splitting")
    print("  Metrics  : Precision | Recall | Hit | NDCG | MRR @ 3,5,10")
    print("=" * 70)

    # ── Load corpus ───────────────────────────────────────────────────────
    print("\n[1/7] Loading corpus ...")
    with open(f"{RESULTS_DIR}/corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_ids     = [d["id"]   for d in corpus]
    corpus_ids_set = set(corpus_ids)
    corpus_texts   = {d["id"]: d["text"] for d in corpus}
    print(f"      Corpus size: {len(corpus)} passages")

    # ── Load HotpotQA ─────────────────────────────────────────────────────
    print("\n[2/7] Loading HotpotQA validation split ...")
    val_dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    print(f"      Validation questions: {len(val_dataset)}")

    # ── Build relevance map ───────────────────────────────────────────────
    print(f"\n[3/7] Building relevance map (up to {EVAL_QUERIES} queries) ...")
    relevance = build_relevance_map(val_dataset, corpus_ids_set, EVAL_QUERIES)
    queries   = list(relevance.values())
    print(f"      Matched queries: {len(queries)}")

    if len(queries) == 0:
        print("  Falling back to train split ...")
        train_ds  = load_dataset("hotpot_qa", "distractor", split="train")
        relevance = build_relevance_map(train_ds, corpus_ids_set, EVAL_QUERIES)
        queries   = list(relevance.values())
        print(f"      Queries found: {len(queries)}")

    # Print distribution
    word_counts = [q["word_count"] for q in queries]
    short_qs = sum(1 for w in word_counts if w <= SHORT_QUERY_WORDS)
    long_qs  = sum(1 for w in word_counts if w >= LONG_QUERY_WORDS)
    print(f"\n      Query stats:")
    print(f"      Short (<=6 words)  = {short_qs}  → will be EXPANDED")
    print(f"      Long  (>=12 words) = {long_qs}   → will be SPLIT")
    print(f"      Medium             = {len(queries)-short_qs-long_qs}")

    # ── Load models ───────────────────────────────────────────────────────
    print("\n[4/7] Loading models ...")
    print("      → BAAI/bge-m3 (fp16) ...")
    embed_model   = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("      → Cross-encoder ...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print("      → Groq client ...")
    groq_client   = Groq(api_key=GROQ_API_KEY)
    print("      All models ready.")

    # ── Build BM25 ────────────────────────────────────────────────────────
    print("\n[5/7] Building BM25 index ...")
    tokenized = [corpus_texts[cid].lower().split() for cid in corpus_ids]
    bm25      = BM25Okapi(tokenized)
    print("      BM25 ready.")

    # ── Connect Pinecone ──────────────────────────────────────────────────
    print("\n[6/7] Connecting to Pinecone ...")
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"      Connected. Vectors: {stats['total_vector_count']}")

    # ── PART A: Original Queries ──────────────────────────────────────────
    print(f"\n[7/7] Running evaluation on {len(queries)} queries ...")
    print(f"\n  ── PART A: Original Queries ──")
    original_results = run_eval(
        queries, embed_model, index, bm25, corpus_ids,
        corpus_texts, cross_encoder, groq_client, label="ORIGINAL"
    )
    print_results_table(original_results, "ORIGINAL QUERIES")

    # ── PART B: Paraphrased Queries ───────────────────────────────────────
    print(f"\n  ── PART B: Paraphrasing queries via Groq ({GROQ_MODEL}) ──")
    paraphrased_queries = []
    for q in tqdm(queries, desc="Paraphrasing"):
        para = paraphrase_query(q["question"], groq_client)
        paraphrased_queries.append({
            "question":     para,
            "relevant_ids": list(q["relevant_ids"]),
            "q_type":       q["q_type"],
            "word_count":   len(para.split()),
            "original":     q["question"],
        })
        time.sleep(0.1)

    with open(f"{RESULTS_DIR}/paraphrased_queries.json", "w", encoding="utf-8") as f:
        json.dump(paraphrased_queries, f, ensure_ascii=False, indent=2)
    print(f"      Saved paraphrased queries.")

    # Convert for eval
    para_queries_eval = [{
        "question":     pq["question"],
        "relevant_ids": set(pq["relevant_ids"]),
        "q_type":       pq["q_type"],
        "word_count":   pq["word_count"],
    } for pq in paraphrased_queries]

    paraphrase_results = run_eval(
        para_queries_eval, embed_model, index, bm25, corpus_ids,
        corpus_texts, cross_encoder, groq_client, label="PARAPHRASE"
    )
    print_results_table(paraphrase_results, "PARAPHRASED QUERIES")

    # ── Save Results ──────────────────────────────────────────────────────
    final = {"original": original_results, "paraphrase": paraphrase_results}

    with open(f"{RESULTS_DIR}/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"\n✅  Results saved → {RESULTS_DIR}/eval_results.json")

    rows = []
    for mode, res in final.items():
        for pipeline, metrics in res.items():
            row = {"mode": mode, "pipeline": pipeline}
            row.update(metrics)
            rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/eval_results.csv", index=False)
    print(f"✅  CSV saved → {RESULTS_DIR}/eval_results.csv")

    return final


if __name__ == "__main__":
    main()
