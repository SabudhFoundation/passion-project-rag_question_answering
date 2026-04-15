"""
src/models/predict_model.py
===========================
Generator — the final step of the RAG pipeline.

Takes retrieved chunks + user question, sends them to the LLM (Groq),
and returns a structured answer.

DATA FLOW:
    Retrieved chunks (from HybridRetriever)
        |
    Generator._build_context(chunks) -> formatted context string
        |
    RAG_SYSTEM_PROMPT.format(context, query) -> full prompt
        |
    Groq LLM (llama-3.3-70b-versatile) -> JSON response
        |
    Parse JSON -> dict with answer_found, answer, sources

USAGE:
    generator = Generator()
    result    = generator.generate("Who directed Titanic?", chunks)
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from exceptions import GenerationError
from models.prompts import RAG_SYSTEM_PROMPT
from models.llm_setup import init_llm

logger = get_logger(__name__)


class Generator:
    """
    Orchestrates the generation step of the RAG pipeline.

    SINGLE RESPONSIBILITY: Take context + question -> LLM -> answer.

    COMPONENTS:
      - prompts.py:   prompt template (RAG_SYSTEM_PROMPT)
      - llm_setup.py: Groq client initialization
      - This class:   context building, LLM calling, response parsing

    INTERFACE:
      generator = Generator()
      result    = generator.generate(query, chunks)
      result    = {"status": "success", "answer_found": True, "answer": "...", "sources": [...]}
    """

    def __init__(self) -> None:
        """
        Initializes the Generator with the Groq LLM client.

        Raises:
            GenerationError -- if GROQ_API_KEY is missing.
        """
        try:
            self._client = init_llm()
            self._model = config.GROQ_MODEL
            logger.info("Generator initialised (model: %s)", self._model)
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(
                f"Failed to initialise Generator: {e}"
            ) from e

    def __repr__(self) -> str:
        return f"Generator(model='{self._model}')"

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an answer using the LLM with retrieved context.

        Args:
            query:  the user's question.
            chunks: list of chunk dicts from the retriever.

        Returns:
            dict with keys: status, answer_found, answer, sources.
        """
        if not chunks:
            logger.warning("No chunks provided -- returning fallback answer")
            return {
                "status": "success",
                "answer_found": False,
                "answer": "I don't have enough context to answer this question.",
                "sources": [],
            }

        try:
            # 1. Build context from chunks
            context = self._build_context(chunks)

            # 2. Format prompt
            prompt = RAG_SYSTEM_PROMPT.format(context=context, query=query)

            # 3. Call LLM
            logger.info("Calling Groq LLM (%s)...", self._model)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # 4. Parse JSON response
            raw_content = response.choices[0].message.content
            llm_output = json.loads(raw_content)

            answer_found = llm_output.get("answer_found", False)
            answer = llm_output.get("answer", "Error generating text.")

            # 5. Extract source titles
            sources = list({
                c.get("title", c.get("source", "Unknown"))
                for c in chunks
            })

            logger.info("  Answer generated (found=%s, sources=%d)", answer_found, len(sources))

            return {
                "status": "success",
                "answer_found": answer_found,
                "answer": answer,
                "sources": sources,
            }

        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON")
            return {
                "status": "error",
                "answer_found": False,
                "answer": "LLM returned invalid JSON. Please try again.",
                "sources": [],
            }
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return {
                "status": "error",
                "answer_found": False,
                "answer": f"Generation error: {e}",
                "sources": [],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into a single context string for the LLM.

        Each chunk is wrapped with source markers so the LLM knows
        which information came from which document.

        Args:
            chunks: list of chunk dicts from retrieval.

        Returns:
            formatted context string.
        """
        context_parts = []
        for i, chunk in enumerate(chunks):
            title = chunk.get("title", chunk.get("source", f"Source {i + 1}"))
            text = chunk.get("text", "")
            context_parts.append(
                f"--- START CHUNK FROM {title} ---\n"
                f"{text}\n"
                f"--- END CHUNK ---"
            )
        return "\n\n".join(context_parts)
