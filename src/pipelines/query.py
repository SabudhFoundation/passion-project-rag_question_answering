"""
src/pipelines/query.py
======================
QueryPipeline — handles user question answering via hybrid retrieval.

DATA FLOW:
    User question (string)
        ↓
    Load chunks from disk → convert to LangChain Documents
        ↓
    LangChainVectorStore.connect_existing() → vector retriever
        ↓
    HybridRetriever(documents, store) → BM25 + Vector ensemble
        ↓
    HybridRetriever.retrieve(question) → ranked chunks
        ↓
    Generate answer from top chunks
        ↓
    Return dict with query, retrieved_chunks, answer

USAGE:
    pipeline = QueryPipeline()
    result   = pipeline.run("Who directed Titanic?")
"""

import os
import sys
from typing import Dict, Any, List, Optional
from src.agents.query_router import QueryRouter
from src.agents.query_rewriter import QueryRewriter
from src.models.router_llm import RouterLLM

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from exceptions import RAGPipelineError, RetrievalError
from models.retriever import HybridRetriever
from models.predict_model import Generator

logger = get_logger(__name__)


class QueryPipeline:
    """
    Retrieves relevant chunks using hybrid search and generates an answer.

    RETRIEVAL STRATEGY:
      Combines BM25 (keyword) + Vector (semantic) search via the
      HybridRetriever. This catches both exact entity matches and
      semantic similarities.

    COMPONENTS:
      - HybridRetriever:      Pinecone Hybrid search (BM25 + Dense) + ColBERT Re-rank
      - Generator:            Groq LLM for answering

    USAGE:
        pipeline = QueryPipeline()
        result   = pipeline.run("Who directed Titanic?")
    """

    def __init__(self) -> None:
        """
        Initializes the query pipeline.

        Loads chunks and builds the hybrid retriever on first init.
        This takes a few seconds but only happens once.
        """
        self._retriever: Optional[HybridRetriever] = None
        self.chat_history = []
        self._generator = Generator()
        router_llm = RouterLLM()

        self.router = QueryRouter(
            router_llm
        )

        self.rewriter = QueryRewriter(
            router_llm
        )
        self._initialized = False
        logger.info("QueryPipeline initialised (retriever builds on first query)")

    def handle_non_rag_query(
        self,
        query: str,
        intent: str
    ) -> Dict[str, Any]:

        # Greeting
        if intent == "greeting":

            if self.has_chat_history():

                response = self._generator.llm.invoke(
                    f"""
                    Continue this conversation naturally.

                    User: {query}
                    """
                )

                return {
                    "query": query,
                    "retrieved_chunks": [],
                    "answer": str(response)
                }

            return {
                "query": query,
                "retrieved_chunks": [],
                "answer": "Hello! How can I help you today?"
            }

        # Clarification
        elif intent == "clarification":

            if not self.has_chat_history():

                return {
                    "query": query,
                    "retrieved_chunks": [],
                    "answer": "Please enter a valid query."
                }

            rewritten_query = self.rewriter.rewrite_with_history(
                query,
                self.chat_history
            )
 
            return {
                "query": query,
                "rewritten_query": rewritten_query,
                "retrieved_chunks": [],
                "answer": (
                    f"I understood your reference as: "
                    f"'{rewritten_query}'"
                )
            }

        # Coding
        elif intent == "coding":

            response = self._generator.llm.invoke(
                f"""
                Write code for:

                {query}

                Return only the solution.
                """
            )

            return {
                "query": query,
                "retrieved_chunks": [],
                "answer": str(response)
            }

        # Math
        elif intent == "math":

            response = self._generator.llm.invoke(
                f"Solve:\n{query}"
            )

            return {
                "query": query,
                "retrieved_chunks": [],
                "answer": str(response)
            }

        # Unknown / irrelevant
        return {
            "query": query,
            "retrieved_chunks": [],
            "answer": "Please enter a valid query."
        }

    def __repr__(self) -> str:
        return f"QueryPipeline(top_k={config.TOP_K}, initialized={self._initialized})"

    def has_chat_history(self) -> bool:
        return len(self.chat_history) > 0

    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Retrieves relevant chunks and generates an answer for a query.

        Args:
            query: user's natural language question.
            verbose: whether to log details.

        Returns:
            dict containing answer and retrieval results.
        """

        if verbose:
            logger.info("=" * 55)
            logger.info("  Query: %s", query)
            logger.info("=" * 55)

        try:
            # --------------------------------------------------
            # Route query
            # --------------------------------------------------
            route = self.router.route(query)
            intent = route["intent"]

            logger.info(
                f"Intent={intent} "
                f"Retrieval={route.get('needs_retrieval', True)}"
            )

            # --------------------------------------------------
            # Greeting
            # --------------------------------------------------

            if intent == "greeting":

                return self.handle_non_rag_query(
                    query,
                    intent
                )

            if intent == "small_talk":

                return {
                    "query": query,
                    "retrieved_chunks": [],
                    "answer": (
                        "Please enter a valid query."
                    )
                }
    

            # --------------------------------------------------
            # Clarification queries
            # --------------------------------------------------

            if intent == "clarification":

                if not self.chat_history:

                    return {
                        "query": query,
                        "retrieved_chunks": [],
                        "answer": (
                            "Please enter a valid query. "
                            "I don't have any previous context "
                            "to clarify."
                        )
                    }
                
                rewritten_query = (
                    self.rewriter.rewrite_with_history(
                        query,
                        self.chat_history
                    )
                )

                logger.info(
                    f"Clarification rewritten as: "
                    f"{rewritten_query}"
                )

                self._ensure_initialized()

                retrieved_chunks = self._retrieve(
                    rewritten_query,
                    verbose
                )

                answer = self._generate(
                    rewritten_query,
                    retrieved_chunks
                )

                self.chat_history.append({
                    "query": query,
                    "rewritten_query": rewritten_query,
                    "answer": answer
                })

                MAX_HISTORY = 10

                if len(self.chat_history) > MAX_HISTORY:
                    self.chat_history = (
                        self.chat_history[-MAX_HISTORY:]
                    )

                return {
                    "query": query,
                    "rewritten_query": rewritten_query,
                    "intent": intent,
                    "retrieved_chunks": retrieved_chunks,
                    "answer": answer
                }

            # --------------------------------------------------
            # Math / Coding
            # --------------------------------------------------

            if intent in ["math", "coding"]:

                return self.handle_non_rag_query(
                    query,
                    intent
                )

            # --------------------------------------------------
            # Unknown / unsupported query
            # --------------------------------------------------

            if not route.get(
                "needs_retrieval",
                True
            ):

                return {
                    "query": query,
                    "retrieved_chunks": [],
                    "answer": (
                        "Please enter a valid query."
                    )
                }

            # --------------------------------------------------
            # Decide whether query should be rewritten
            # --------------------------------------------------

            needs_rewrite = self.router.should_rewrite(
                query,
                self.chat_history
            )

            if needs_rewrite:

                rewritten_query = (
                    self.rewriter.rewrite_with_history(
                        query,
                        self.chat_history
                    )
                )

                logger.info(
                    f"Rewritten query: {rewritten_query}"
                )

            else:

                rewritten_query = query

            # --------------------------------------------------
            # Retrieval
            # --------------------------------------------------

            self._ensure_initialized()

            retrieved_chunks = self._retrieve(
                rewritten_query,
                verbose
            )

            # --------------------------------------------------
            # Generation
            # --------------------------------------------------

            answer = self._generate(
                query,
                retrieved_chunks
            )

            if verbose:
                logger.info("  %s", "─" * 53)
                logger.info("  Answer: %s", answer)
                logger.info("  %s", "─" * 53)

            # --------------------------------------------------
            # Save chat history
            # --------------------------------------------------

            self.chat_history.append({
                "query": query,
                "rewritten_query": rewritten_query,
                "answer": answer
            })

            MAX_HISTORY = 10

            if len(self.chat_history) > MAX_HISTORY:
                self.chat_history = (
                    self.chat_history[-MAX_HISTORY:]
                )

            # --------------------------------------------------
            # Return result
            # --------------------------------------------------

            return {
                "query": query,
                "rewritten_query": rewritten_query,
                "intent": intent,
                "retrieved_chunks": retrieved_chunks,
                "answer": answer
            }

        except RAGPipelineError as e:

            logger.error(
                "Query failed: %s",
                e
            )

            return {
                "query": query,
                "retrieved_chunks": [],
                "answer": f"Error: {e}"
            }

        except Exception as e:

            logger.exception(
                "Unexpected error"
            )

            return {
                "query": query,
                "retrieved_chunks": [],
                "answer": f"Unexpected error: {e}"
            }


    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """
        Lazily builds the hybrid retriever on first use.
        """
        if self._initialized:
            return

        logger.info("Building hybrid retriever (first query — one-time setup)...")

        try:
            self._retriever = HybridRetriever()
            self._initialized = True
            logger.info("Hybrid retriever ready")

        except RAGPipelineError:
            raise
        except Exception as e:
            raise RetrievalError(
                f"Failed to initialize retriever: {e}"
            ) from e

    def _retrieve(
        self,
        rewritten_query: str,
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """
        Runs hybrid retrieval using the rewritten query.
        """

        if verbose:
            logger.info(
                "Retrieving via hybrid search (BM25 + Vector)..."
            )

        logger.info(
            f"Retrieval Query: {rewritten_query}"
        )

        if not self._retriever:
            raise RetrievalError(
                "Retriever not initialized"
            )

        retrieved_chunks = self._retriever.retrieve(
            rewritten_query
        )

        if verbose:

            logger.info(
                "Found %d chunks:",
                len(retrieved_chunks)
            )

            for i, c in enumerate(
                retrieved_chunks,
                1
            ):
                logger.info(
                    "[%d] score=%.4f | title=%s",
                    i,
                    c.get("score", 0),
                    c.get("title", "")[:50]
                )

        return retrieved_chunks

    def _generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the Groq LLM via the Generator class.

        Args:
            query: user's natural language question.
            retrieved_chunks: list of chunk dicts from retrieval.

        Returns:
            str — the generated answer.
        """
        if not retrieved_chunks:
            return "No relevant context found."
            
        result = self._generator.generate(query=query, chunks=retrieved_chunks)
        return result.get("answer", "Failed to generate an answer.")

