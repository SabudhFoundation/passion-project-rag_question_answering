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


    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an answer using the LLM with retrieved context.

        Args:
            query:  the user's question.
            chunks: list of chunk dicts from the retriever.

        Returns:
            dict with keys: status, answer_found, answer, scratchpad, sources.
            ``scratchpad`` contains the LLM's step-by-step chain-of-thought
            reasoning (multi-hop bridging, arithmetic, comparisons) that
            produced the final answer.  Useful for debugging and evaluation.
        """
        if not chunks:
            logger.warning("No chunks provided -- returning fallback answer")
            return {
                "status": "success",
                "answer_found": False,
                "answer": "I don't have enough context to answer this question.",
                "scratchpad": "",
                "sources": [],
            }

        try:
            # 1. Build context from chunks
            context = self._build_context(chunks)

            # 2. Format prompt
            prompt = RAG_SYSTEM_PROMPT.format(context=context, query=query)

            # 3. Call LLM with fallback mechanism for rate limits
            fallback_models = [
                self._model,
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "gemma2-9b-it",
            ]
            
            # Remove duplicates while preserving order
            fallback_models = list(dict.fromkeys(fallback_models))
            
            response = None
            last_error = None
            
            for m in fallback_models:
                logger.info("Calling Groq LLM (%s)...", m)
                try:
                    response = self._client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                    )
                    # Update active model so subsequent calls prefer this one
                    self._model = m
                    break
                except Exception as e:
                    last_error = e
                    # Fallback on any API-related error (rate limit, decommissioned model, etc)
                    logger.warning("Generation failed for model %s: %s. Trying fallback...", m, str(e)[:100])
                    continue
                        
            if response is None:
                raise last_error

            # 4. Parse JSON response
            raw_content = response.choices[0].message.content
            llm_output = json.loads(raw_content)

            answer_found = llm_output.get("answer_found", False)
            answer       = llm_output.get("answer", "Error generating text.")
            # scratchpad contains the model's <thinking>…</thinking> CoT block
            # (multi-hop bridging, arithmetic, citation verification) — kept for
            # debugging and evaluation; not shown to end-users by default.
            scratchpad   = llm_output.get("scratchpad", "")
            # citations: list of "chunk_N | doc_id | Title" strings the model
            # verified against during STEP 5 — Citation Verification.
            citations    = llm_output.get("citations", [])

            # 5. Extract source titles (from all retrieved chunks)
            sources = list({
                c.get("title", c.get("source", "Unknown"))
                for c in chunks
            })

            logger.info(
                "  Answer generated (found=%s, sources=%d, citations=%d, scratchpad_len=%d)",
                answer_found, len(sources), len(citations), len(scratchpad)
            )

            return {
                "status": "success",
                "answer_found": answer_found,
                "answer": answer,
                "scratchpad": scratchpad,
                "citations": citations,
                "sources": sources,
            }

        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON")
            return {
                "status": "error",
                "answer_found": False,
                "answer": "[ERROR] LLM returned invalid JSON. Please try again.",
                "scratchpad": "",
                "sources": [],
            }
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return {
                "status": "error",
                "answer_found": False,
                "answer": f"[ERROR] Generation error: {e}",
                "scratchpad": "",
                "sources": [],
            }

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into a labelled context string for the LLM.

        Each chunk header is formatted as:
            [chunk_N | doc_id | Title]
        so the CoT prompt's citation rules work correctly. The LLM can then
        cite these chunks in its scratchpad/thinking block, and STEP 5 (Citation Verification)
        can trace every claim back to its exact source chunk, keeping the final
        answer clean of inline bracketed references.

        Args:
            chunks: list of chunk dicts from retrieval.

        Returns:
            formatted, labelled context string.
        """
        context_parts = []
        for i, chunk in enumerate(chunks):
            num    = i + 1
            title  = chunk.get("title",     chunk.get("source", f"Source {num}"))
            doc_id = chunk.get("doc_id",    "unknown")
            text   = chunk.get("text",      "")
            context_parts.append(
                f"[chunk_{num} | {doc_id} | {title}]\n"
                f"{text}"
            )
        return "\n\n".join(context_parts)
