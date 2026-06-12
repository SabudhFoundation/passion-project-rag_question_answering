# RAG QnA Bot — HotpotQA

Welcome to the **RAG Question Answering** system, powered by the **HotpotQA** dataset.

## 🔍 How it works

1. **Ask** a multi-hop question in the chat box below
2. **Retrieve** — the system searches 649 Wikipedia documents using hybrid BM25 + vector search
3. **Generate** — an LLM synthesizes the answer from retrieved passages
4. **Inspect** — expand the retrieval & generation steps to see sources and scores

## 📊 Dataset

- **HotpotQA** — 113k multi-hop question-answer pairs
- **649 documents** indexed in Pinecone
- **Hybrid retrieval** — BM25 (40%) + Vector (60%)

## ⚙️ Settings

Click the **⚙️ Settings** icon in the sidebar to adjust:
- **Top-k** — number of chunks to retrieve
- **Min threshold** — minimum similarity score
- **Model** — LLM to use for generation
- **Temperature** — creativity of the response

> Try clicking one of the starter questions below to get started! 🚀
