"""
src/models/retriever.py
=======================
Pinecone Hybrid Retriever with On-the-fly ColBERT Re-ranking.
"""

import os
import sys
from typing import List, Dict, Any

from pinecone import Pinecone
from FlagEmbedding import BGEM3FlagModel

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from exceptions import RetrievalError

logger = get_logger(__name__)

class HybridRetriever:
    """
    Pinecone Hybrid Retriever (Dense + Sparse) + ColBERT Re-ranking.
    """

    def __init__(self, top_k: int = None, alpha: float = None):
        self.top_k = top_k or config.TOP_K
        # Use config.HYBRID_ALPHA as the default; caller can override per-instance.
        self.alpha = alpha if alpha is not None else config.HYBRID_ALPHA
        self._model = None
        self._index = None
        
        try:
            self._connect_pinecone()
            self._load_model()
            logger.info("HybridRetriever initialised (Pinecone + ColBERT Re-rank)")
        except Exception as e:
            raise RetrievalError(f"Failed to initialize: {e}")

    def _connect_pinecone(self):
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self._index = pc.Index(config.PINECONE_INDEX)

    def _load_model(self):
        logger.info("Loading BGE-M3 model for query encoding and re-ranking...")
        self._model = BGEM3FlagModel(config.EMBEDDING_MODEL, use_fp16=True)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if not query or not isinstance(query, str):
            raise RetrievalError("Query must be a string.")

        try:
            logger.info("Hybrid retrieval for: '%s'", query[:60])
            
            # 1. Encode query for Pinecone
            output = self._model.encode([query], return_dense=True, return_sparse=True)
            dense_vec = output["dense_vecs"][0].tolist()
            sparse_dict = output["lexical_weights"][0]
            
            # Scale with alpha (Pinecone hybrid weighting)
            alpha = self.alpha
            scaled_dense = [v * alpha for v in dense_vec]
            scaled_sparse = {
                "indices": [int(k) for k in sparse_dict.keys()],
                "values": [float(v) * (1.0 - alpha) for v in sparse_dict.values()]
            }

            # 2. Query Pinecone (fetch top 50 candidates)
            fetch_k = 50
            results = self._index.query(
                vector=scaled_dense,
                sparse_vector=scaled_sparse,
                top_k=fetch_k,
                namespace=config.NAMESPACE_HOTPOT,
                include_metadata=True
            )

            matches = results.get("matches", [])
            if not matches:
                return []

            # 3. On-the-fly ColBERT Re-ranking
            logger.info("  Re-ranking %d candidates using ColBERT...", len(matches))
            sentence_pairs = []
            for match in matches:
                text = match.get("metadata", {}).get("original_text", "")
                sentence_pairs.append([query, text])
            
            # Compute ColBERT scores
            # weights_for_different_modes: [dense, sparse, colbert]
            scores = self._model.compute_score(sentence_pairs, weights_for_different_modes=[0, 0, 1])
            colbert_scores = scores["colbert"]
            
            # Combine and sort
            retrieved_chunks = []
            for i, match in enumerate(matches):
                meta = match.get("metadata", {})
                retrieved_chunks.append({
                    "text": meta.get("original_text", ""),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "score": round(float(colbert_scores[i]), 4),
                    "metric": "colbert",
                    "chunk_index": meta.get("chunk_idx", 0),
                    "is_bridge": meta.get("is_bridge", False),
                    "is_multihop": meta.get("is_multihop", False)
                })

            retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)
            
            final_results = retrieved_chunks[:self.top_k]
            logger.info("  Returned top %d chunks", len(final_results))
            return final_results

        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}") from e
