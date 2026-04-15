"""
src/pipelines/evaluation.py
============================
EvaluationPipeline — evaluates system accuracy on HotpotQA questions.

METRIC: Exact Match (EM)
    EM = 1 if the gold answer appears in the retrieved text
    EM = 0 otherwise
    Final score = (correct / total) × 100%

For a baseline RAG system, EM of 20-35% is normal.
Improvements come from better embeddings, re-ranking, or larger LLMs.

USAGE:
    pipeline = EvaluationPipeline()
    pipeline.run(num_questions=10)
"""

import os
import sys
import json
from typing import List, Dict, Any

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from pipelines.query import QueryPipeline

logger = get_logger(__name__)


class EvaluationPipeline:
    """
    Evaluates system accuracy on HotpotQA questions using Exact Match.

    USAGE:
        pipeline = EvaluationPipeline()
        pipeline.run(num_questions=20)
    """

    def __init__(self) -> None:
        """Initializes the evaluation pipeline with a QueryPipeline."""
        self._query_pipeline = QueryPipeline()
        logger.info("EvaluationPipeline initialised")

    def __repr__(self) -> str:
        return "EvaluationPipeline(metric='ExactMatch')"

    def run(self, num_questions: int = 10) -> None:
        """
        Evaluates system on the first N HotpotQA questions.

        Args:
            num_questions: number of questions to evaluate on.
        """
        logger.info("=" * 55)
        logger.info("  EVALUATION — %d HotpotQA questions", num_questions)
        logger.info("=" * 55)

        # ── Load dataset ──────────────────────────────────────────────────
        data = self._load_dataset(num_questions)
        if data is None:
            return

        # ── Evaluate each question ────────────────────────────────────────
        results = self._evaluate_questions(data)

        # ── Print results ─────────────────────────────────────────────────
        self._print_results(results)

    def _load_dataset(self, num_questions: int) -> list:
        """
        Loads HotpotQA dataset from disk.

        Args:
            num_questions: number of entries to load.

        Returns:
            list of dataset entries, or None if file not found.
        """
        if not os.path.exists(config.HOTPOTQA_FILE):
            logger.error("HotpotQA file not found. Run --ingest first.")
            return None

        try:
            with open(config.HOTPOTQA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)[:num_questions]
            logger.info("  Loaded %d questions from dataset", len(data))
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Cannot load HotpotQA file: %s", e)
            return None

    def _evaluate_questions(
        self, data: List[Dict[str, Any]]
    ) -> List[bool]:
        """
        Evaluates each question against the gold answer.

        Args:
            data: list of HotpotQA entries with question and answer.

        Returns:
            list of booleans (True = correct, False = incorrect).
        """
        correct = 0
        results: List[bool] = []

        for i, entry in enumerate(data, 1):
            question    = entry.get("question", "")
            gold_answer = entry.get("answer", "").lower().strip()
            q_type      = entry.get("type", "")
            level       = entry.get("level", "")

            if not question:
                continue

            output = self._query_pipeline.run(question, verbose=False)
            pred   = output["answer"].lower()

            is_correct = gold_answer in pred
            if is_correct:
                correct += 1

            status = "PASS" if is_correct else "FAIL"
            logger.info(
                "  [%2d] %s [%s/%s] Q: %s",
                i, status, level, q_type, question[:55],
            )
            results.append(is_correct)

        return results

    @staticmethod
    def _print_results(results: List[bool]) -> None:
        """Prints the final evaluation summary."""
        total   = len(results)
        correct = sum(results)
        em      = correct / total * 100 if total > 0 else 0

        logger.info("─" * 55)
        logger.info("  Exact Match Score: %.1f%% (%d/%d)", em, correct, total)
        logger.info("─" * 55)
