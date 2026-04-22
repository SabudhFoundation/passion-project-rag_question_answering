"""
src/visualization/app.py
========================
RAG QnA Bot — Chainlit-based interactive chatbot UI.

Dataset : HotpotQA (649 documents)
Retrieval: Hybrid BM25 + Vector (Pinecone)
LLM     : Groq llama-3.3-70b-versatile

Run:
    chainlit run src/visualization/app.py -w
    python src/main.py --app
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import random
import sys
import time
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Slider

# ── Path setup ───────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Logger ───────────────────────────────────────────────────────────────────
try:
    from logger import get_logger
    log = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    log = logging.getLogger(__name__)

# ── Pipeline imports (graceful stubs when not present) ────────────────────────
_BackendGenerator = _Chunker = _VectorStore = _HybridRetriever = None
try:
    from models.predict_model import Generator as BackendGenerator
    _BackendGenerator = BackendGenerator
except ImportError:
    log.warning("BackendGenerator not found — stub mode.")
try:
    from preprocessing_data.pre_processing import Chunker
    from models.train_model import LangChainVectorStore
    from models.retriever import HybridRetriever
    _Chunker = Chunker
    _VectorStore = LangChainVectorStore
    _HybridRetriever = HybridRetriever
except ImportError:
    log.warning("Retriever pipeline not found — stub mode.")


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class ChunkResult:
    def __init__(self, chunk_id: str, source_title: str, source_type: str, publish_date: str, chunk_index: int, total_chunks: int, similarity_score: float, similarity_metric: str, content_preview: str, full_content: str):
        self.chunk_id = chunk_id
        self.source_title = source_title
        self.source_type = source_type
        self.publish_date = publish_date
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.similarity_score = similarity_score
        self.similarity_metric = similarity_metric
        self.content_preview = content_preview
        self.full_content = full_content

class RAGResult:
    def __init__(self, answer: str, chunks: list[ChunkResult], model_name: str, tokens_used: int, latency_seconds: float, confidence_score: float, top_k: int, score_threshold: float):
        self.answer = answer
        self.chunks = chunks
        self.model_name = model_name
        self.tokens_used = tokens_used
        self.latency_seconds = latency_seconds
        self.confidence_score = confidence_score
        self.top_k = top_k
        self.score_threshold = score_threshold


# ─────────────────────────────────────────────────────────────────────────────
# STUB DATA — HotpotQA flavour (used when pipeline is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

_STUB_META = [
    ("Yang et al. 2018 — HotpotQA Dataset Paper",     "ArXiv", "2018-09-21", 0.94),
    ("Wikipedia — Eiffel Tower Construction History",  "Wiki",  "2023-01-10", 0.89),
    ("Wikipedia — Albert Einstein Early Life",         "Wiki",  "2022-08-05", 0.84),
    ("Fader et al. — Open Domain QA with Multi-Hop",  "ArXiv", "2019-04-12", 0.78),
    ("Wikipedia — Marie Curie Nobel Prize 1911",       "Wiki",  "2023-03-22", 0.73),
]

_STUB_TEXTS = [
    "HotpotQA is a question answering dataset featuring natural, multi-hop questions, "
    "with strong supervision for supporting facts to enable more explainable QA systems.",
    "The Eiffel Tower was constructed between 1887 and 1889 as the entrance arch to the "
    "1889 World's Fair, designed by engineer Gustave Eiffel.",
    "Albert Einstein was born on 14 March 1879 in Ulm, in the Kingdom of Württemberg in "
    "the German Empire. His parents were Hermann Einstein and Pauline Koch.",
    "Open-domain multi-hop question answering requires a model to retrieve and reason "
    "across multiple documents to arrive at a single answer.",
    "Marie Curie was awarded the Nobel Prize in Chemistry in 1911, making her the first "
    "person to win Nobel Prizes in two different sciences.",
]

_STUB_ANSWER = (
    "Based on the retrieved passages, here is a comprehensive answer:\n\n"
    "HotpotQA is a **multi-hop question answering** dataset that requires reasoning "
    "across two or more Wikipedia passages to answer a single question.\n\n"
    "**Key characteristics:**\n\n"
    "- Questions are written by crowdworkers shown two related Wikipedia paragraphs\n"
    "- Each question has **supporting fact annotations** for explainability\n"
    "- Contains ~113k question-answer pairs across *distractor* and *fullwiki* settings\n"
    "- Answers are free-form spans extracted from the retrieved passages\n\n"
    "The dataset is widely used to benchmark retrieval-augmented generation, "
    "chain-of-thought reasoning, and multi-document reading comprehension."
)


def _stub(query: str, top_k: int, thr: float, model: str) -> RAGResult:
    n = min(top_k, len(_STUB_META))
    chunks: list[ChunkResult] = []
    for i in range(n):
        title, src, date, base = _STUB_META[i]
        score = round(min(max(base + random.uniform(-0.02, 0.02), 0.0), 1.0), 2)
        if score < thr:
            continue
        chunks.append(ChunkResult(
            chunk_id=f"c{i}",
            source_title=title,
            source_type=src,
            publish_date=date,
            chunk_index=i + 1,
            total_chunks=n,
            similarity_score=score,
            similarity_metric="cosine",
            content_preview=_STUB_TEXTS[i],
            full_content=_STUB_TEXTS[i],
        ))
    conf = (
        round(sum(c.similarity_score for c in chunks[:3]) / max(len(chunks[:3]), 1), 2)
        if chunks else 0.0
    )
    return RAGResult(
        answer=_STUB_ANSWER,
        chunks=chunks,
        model_name=model,
        tokens_used=len(_STUB_ANSWER) // 4 + sum(len(c.full_content) // 4 for c in chunks),
        latency_seconds=round(random.uniform(0.7, 2.0), 2),
        confidence_score=conf,
        top_k=top_k,
        score_threshold=thr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_retriever():
    """Build the hybrid retriever from disk chunks + Pinecone."""
    if not (_Chunker and _VectorStore and _HybridRetriever):
        return None
    try:
        chunker = _Chunker()
        chunks = chunker.load_chunks_from_disk()
        if not chunks:
            log.warning("No chunks on disk — run ingest.")
            return None
        docs = _Chunker.to_langchain_documents(chunks)
        store = _VectorStore()
        store.connect_existing()
        return _HybridRetriever(documents=docs, langchain_store=store)
    except Exception as e:
        log.error("Retriever build failed: %s — stub mode.", e)
        return None


def _run_rag(
    query: str,
    retriever: Any,
    generator: Any,
    top_k: int = 5,
    thr: float = 0.0,
    model: str = "llama-3.3-70b-versatile",
) -> RAGResult:
    """Execute the RAG pipeline: retrieve → generate → package result."""
    if not retriever or not generator:
        return _stub(query, top_k, thr, model)

    t0 = time.time()
    try:
        raw = retriever.retrieve(query)[:top_k]
    except Exception as e:
        log.error("Retrieval error: %s", e)
        return _stub(query, top_k, thr, model)

    chunks: list[ChunkResult] = []
    for i, rc in enumerate(raw):
        score = float(rc.get("score", 0.0))
        if score < thr:
            continue
        chunks.append(ChunkResult(
            chunk_id=f"c{i}",
            source_title=rc.get("title", rc.get("source", f"Doc {i+1}")),
            source_type=rc.get("source_type", "Wiki"),
            publish_date=rc.get("publish_date", "N/A"),
            chunk_index=i + 1,
            total_chunks=len(raw),
            similarity_score=round(score, 3),
            similarity_metric=rc.get("metric", "cosine"),
            content_preview=rc.get("text", "")[:150],
            full_content=rc.get("text", ""),
        ))

    try:
        out = generator.generate(query=query, chunks=raw)
        answer = out.get("answer", "No answer returned.")
    except Exception as e:
        log.error("Generation error: %s", e)
        answer = f"Generation error: {e}"

    conf = (
        round(sum(c.similarity_score for c in chunks[:3]) / max(len(chunks[:3]), 1), 2)
        if chunks else 0.0
    )
    return RAGResult(
        answer=answer,
        chunks=chunks,
        model_name=model,
        tokens_used=len(answer) // 4 + sum(len(c.full_content) // 4 for c in chunks),
        latency_seconds=round(time.time() - t0, 2),
        confidence_score=conf,
        top_k=top_k,
        score_threshold=thr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHAINLIT LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Initialise the RAG pipeline and store in user session."""
    log.info("New chat session started.")

    # Build pipeline components (runs in thread to avoid blocking)
    retriever = await asyncio.to_thread(_build_retriever)
    generator = _BackendGenerator() if _BackendGenerator else None

    # Store in session
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("generator", generator)
    cl.user_session.set("settings", {
        "top_k": 5,
        "threshold": 0.0,
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
    })

    # Configure chat settings (sidebar widgets)
    settings = await cl.ChatSettings(
        [
            Slider(
                id="top_k",
                label="Top-k chunks",
                initial=5,
                min=1,
                max=20,
                step=1,
                description="Number of chunks to retrieve",
            ),
            Slider(
                id="threshold",
                label="Min similarity threshold",
                initial=0.0,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Minimum score to include a chunk",
            ),
            Select(
                id="model",
                label="LLM Model",
                values=[
                    "llama-3.3-70b-versatile",
                    "gpt-4o",
                    "gpt-3.5-turbo",
                    "claude-3-5-sonnet",
                ],
                initial_value="llama-3.3-70b-versatile",
                description="Model used for answer generation",
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.2,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Creativity of the response",
            ),
        ]
    ).send()

    # Status indicator
    pipeline_status = "🟢 Connected" if retriever else "🟡 Stub mode"
    doc_count = "649" if retriever else "0"

    await cl.Message(
        content=(
            f"**RAG QnA Bot** ready!\n\n"
            f"**Knowledge base:** HotpotQA · {doc_count} documents · {pipeline_status}\n"
            f"**Model:** {cl.user_session.get('settings')['model']}\n\n"
            f"Ask a multi-hop question, or click a starter below. 👇"
        ),
        author="System",
    ).send()


# ─────────────────────────────────────────────────────────────────────────────
# STARTER QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Director of Titanic",
            message="What was the occupation of the director of the film Titanic?",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Einstein vs Curie",
            message="Who was born first — Albert Einstein or Marie Curie?",
            icon="/public/learn.svg",
        ),
        cl.Starter(
            label="Eiffel Tower",
            message="Which country is the Eiffel Tower located in, and who designed it?",
            icon="/public/globe.svg",
        ),
        cl.Starter(
            label="Multi-hop QA",
            message="What is HotpotQA and why is it important for question answering research?",
            icon="/public/terminal.svg",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS UPDATE
# ─────────────────────────────────────────────────────────────────────────────

@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Called when user changes any setting in the sidebar."""
    cl.user_session.set("settings", {
        "top_k": int(settings.get("top_k", 5)),
        "threshold": float(settings.get("threshold", 0.0)),
        "model": settings.get("model", "llama-3.3-70b-versatile"),
        "temperature": float(settings.get("temperature", 0.2)),
    })
    log.info("Settings updated: %s", cl.user_session.get("settings"))


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def _score_emoji(score: float) -> str:
    if score >= 0.85:
        return "🟢"
    if score >= 0.70:
        return "🟡"
    return "🔴"


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message: retrieve → generate → stream answer."""
    query = message.content.strip()
    if not query:
        return

    # Get session state
    retriever = cl.user_session.get("retriever")
    generator = cl.user_session.get("generator")
    settings = cl.user_session.get("settings") or {}
    top_k = int(settings.get("top_k", 5))
    thr = float(settings.get("threshold", 0.0))
    model = settings.get("model", "llama-3.3-70b-versatile")

    # ── Step 1: Retrieval ─────────────────────────────────────────────────
    async with cl.Step(name="🔍 Retrieving documents", type="retrieval") as retrieval_step:
        result = await asyncio.to_thread(
            _run_rag, query, retriever, generator, top_k, thr, model
        )

        # Build retrieval summary
        if result.chunks:
            chunks_md = f"**Retrieved {len(result.chunks)} chunks** "
            chunks_md += f"(confidence: {result.confidence_score:.2f})\n\n"
            for i, c in enumerate(result.chunks):
                emoji = _score_emoji(c.similarity_score)
                chunks_md += (
                    f"---\n"
                    f"**[{i+1}] {c.source_title}** {emoji}\n"
                    f"- Score: `{c.similarity_score:.3f}` ({c.similarity_metric})\n"
                    f"- Source: `{c.source_type}` · {c.publish_date} "
                    f"· Chunk {c.chunk_index}/{c.total_chunks}\n"
                    f"- Preview: *{c.content_preview[:120]}…*\n\n"
                )
        else:
            chunks_md = "No chunks retrieved above the threshold."

        retrieval_step.output = chunks_md

    # ── Step 2: Generation ────────────────────────────────────────────────
    async with cl.Step(name="🧠 Generating answer", type="llm") as gen_step:
        gen_step.output = (
            f"**Model:** `{model}`\n"
            f"**Tokens:** ~{result.tokens_used:,}\n"
            f"**Latency:** {result.latency_seconds:.2f}s\n"
            f"**Confidence:** {result.confidence_score:.2f}"
        )

    # ── Step 3: Stream the answer ─────────────────────────────────────────
    msg = cl.Message(content="", author="RAG Bot")
    await msg.send()

    # Simulate word-by-word streaming
    words = result.answer.split(" ")
    streamed = ""
    for i in range(0, len(words), 4):
        batch = " ".join(words[i:i + 4])
        streamed += batch + " "
        await msg.stream_token(batch + " ")
        await asyncio.sleep(0.03)

    # Append source citations
    if result.chunks:
        citations = "\n\n---\n📚 **Sources:**\n"
        for i, c in enumerate(result.chunks):
            emoji = _score_emoji(c.similarity_score)
            citations += (
                f"  {i+1}. {emoji} **{c.source_title}** "
                f"— score: {c.similarity_score:.2f}\n"
            )
        await msg.stream_token(citations)

    await msg.update()

    # Attach source elements for expandable preview
    elements = []
    for i, c in enumerate(result.chunks):
        elements.append(
            cl.Text(
                name=f"[{i+1}] {c.source_title}",
                content=(
                    f"**Source:** {c.source_type} · {c.publish_date}\n"
                    f"**Score:** {c.similarity_score:.3f} ({c.similarity_metric})\n"
                    f"**Chunk:** {c.chunk_index}/{c.total_chunks}\n\n"
                    f"---\n\n{c.full_content}"
                ),
                display="side",
            )
        )
    if elements:
        msg.elements = elements
        await msg.update()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — for main.py backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

class RAGApp:
    """Thin wrapper to maintain backward compatibility with main.py."""

    def __init__(self) -> None:
        log.info("RAGApp initialised (Chainlit mode).")

    def launch(self, **kwargs) -> None:
        import subprocess
        port = kwargs.get("port", 8000)
        host = kwargs.get("host", "0.0.0.0")
        app_path = os.path.abspath(__file__)

        log.info("Launching Chainlit on http://%s:%d", host, port)
        subprocess.run(
            [
                sys.executable, "-m", "chainlit", "run",
                app_path,
                "--host", host,
                "--port", str(port),
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(app_path))),
        )
