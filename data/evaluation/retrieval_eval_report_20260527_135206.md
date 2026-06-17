# Retrieval Evaluation Report (20260527_135206)

**Dataset:** HotpotQA (Synthetic Ground-Truth Set)
**Retriever:** `HybridRetriever` (Pinecone Hybrid + BGE-M3 + ColBERT Reranker)
**Questions Evaluated:** 50

## Metrics Summary

### [ORIGINAL]
| Metric | K=1 | K=3 | K=5 | K=10 |
| :--- | :---: | :---: | :---: | :---: |
| **Hit Rate** | 0.640 | 1.000 | 1.000 | 1.000 |
| **Recall** | 41.2% | 82.5% | 94.0% | 97.2% |
| **Precision** | 64.0% | 52.0% | 39.6% | 20.8% |
| **MRR** | 0.640 | 0.807 | 0.807 | 0.807 |
| **NDCG** | 0.640 | 0.796 | 0.882 | 0.897 |

### [PARAPHRASE]
| Metric | K=1 | K=3 | K=5 | K=10 |
| :--- | :---: | :---: | :---: | :---: |
| **Hit Rate** | 0.640 | 1.000 | 1.000 | 1.000 |
| **Recall** | 42.3% | 82.3% | 90.8% | 96.5% |
| **Precision** | 64.0% | 52.7% | 38.4% | 20.6% |
| **MRR** | 0.640 | 0.810 | 0.810 | 0.810 |
| **NDCG** | 0.640 | 0.803 | 0.870 | 0.895 |

## Paraphrase Robustness Summary
Shows how much performance drops when questions are reworded.

| Metric | Original | Paraphrased | Drop |
| :--- | :---: | :---: | :---: |
| **Hit@5** | 1.000 | 1.000 | +0.000 |
| **Recall@5** | 0.940 | 0.908 | +0.032 |

## Metric Definitions
*   **Hit@K**: Binary — did at least one relevant (gold) document appear in the top-K retrieved chunks?
*   **Recall@K**: What fraction of all relevant (gold) chunks (usually 2 in HotpotQA) were retrieved in the top-K?
*   **Precision@K**: What fraction of the retrieved top-K chunks were actually relevant (gold)?
*   **MRR@K (Mean Reciprocal Rank)**: How high up did the first relevant chunk appear? (e.g. Rank 1 = 1.0, Rank 2 = 0.5).
*   **NDCG@K (Normalized Discounted Cumulative Gain)**: Measures the overall ranking quality, considering both presence and position of relevant documents.