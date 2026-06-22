#!/usr/bin/env python3

"""
src/evaluation/improved_ragas_eval.py
===========================================================
RAG OUTPUT GENERATION PIPELINE
===========================================================

Purpose
-------
This file ONLY generates answers once and stores them.

Later you can evaluate:
- 5 samples
- 20 samples
- 100 samples
without regenerating answers again.

This massively reduces:
✓ API cost
✓ Pinecone calls
✓ Retrieval time
✓ LLM latency

-----------------------------------------------------------
USAGE
-----------------------------------------------------------

Generate 10 samples:
python src/evaluation/improved_ragas_eval.py --samples 10

Generate 100 samples:
python src/evaluation/improved_ragas_eval.py --samples 100

Custom output:
python src/evaluation/improved_ragas_eval.py \
    --samples 50 \
    --output generated_outputs.json

-----------------------------------------------------------
OUTPUT
-----------------------------------------------------------

generated_rag_outputs_TIMESTAMP.json

Contains:
- question
- ground truth
- generated answer
- retrieved contexts
- metadata
"""

import os
import sys
import json
import glob
import time
import logging
import argparse

from pathlib import Path
from typing import List, Dict, Any

# =========================================================
# PATH SETUP
# =========================================================

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

# =========================================================
# IMPORTS
# =========================================================

from pipelines.query import QueryPipeline

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =========================================================
# DATASET
# =========================================================

DATA_PATTERN = str(
    ROOT_DIR /
    "data" /
    "evaluation" /
    "synthetic_qa_hotpotqa_*.json"
)

# =========================================================
# HELPERS
# =========================================================

def clean_context(text: str) -> str:
    """
    Clean retrieved context text.
    """

    if not text:
        return ""

    text = text.replace("Context:", "")
    text = text.replace("\n", " ")

    return " ".join(text.split()).strip()


def is_valid_generation(answer: str) -> bool:
    """
    Detect failed generations.
    """

    if not answer:
        return False

    answer = answer.lower()

    invalid_patterns = [
        "rate limit",
        "429",
        "generation error",
        "api error",
        "timeout",
        "traceback",
        "exception",
        "failed",
        "invalid_generation",
        "error"
    ]

    return not any(
        pattern in answer
        for pattern in invalid_patterns
    )


# =========================================================
# DATASET LOADING
# =========================================================

def load_dataset(
    num_samples: int
) -> List[Dict[str, Any]]:

    files = glob.glob(DATA_PATTERN)

    if not files:
        raise FileNotFoundError(
            f"\nNo dataset found.\nExpected pattern:\n{DATA_PATTERN}"
        )

    dataset_path = max(
        files,
        key=os.path.getctime
    )

    logger.info(
        f"Loading dataset: {dataset_path}"
    )

    with open(
        dataset_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # -----------------------------------------------------
    # Handle multiple JSON structures
    # -----------------------------------------------------

    if isinstance(data, dict):

        if "data" in data:
            data = data["data"]

        else:
            data = list(data.values())

    # Keep only valid dict samples
    data = [
        x for x in data
        if isinstance(x, dict)
    ]

    logger.info(
        f"Loaded {len(data)} total samples"
    )

    # deterministic ordering
    data = sorted(
        data,
        key=lambda x: str(
            x.get("qid", "")
        )
    )

    return data[:num_samples]


# =========================================================
# QUERY EXECUTION
# =========================================================

def run_query(
    pipeline: QueryPipeline,
    question: str
) -> Dict[str, Any]:

    try:

        result = pipeline.run(question)

        if not isinstance(result, dict):
            return {}

        return result

    except Exception as e:

        logger.error(
            f"Query failed: {e}"
        )

        return {}


# =========================================================
# MAIN GENERATION LOOP
# =========================================================

def generate_outputs(
    dataset: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    query_pipeline = QueryPipeline()

    generated_outputs = []

    logger.info(
        "=" * 60
    )

    logger.info(
        f"Generating outputs for "
        f"{len(dataset)} samples"
    )

    logger.info(
        "=" * 60
    )

    for idx, sample in enumerate(dataset):

        question = (
            sample.get("question", "")
            .strip()
        )

        ground_truth = (
            sample.get("answer", "")
        )

        sample_id = sample.get(
            "qid",
            f"sample_{idx}"
        )

        if not question:

            logger.warning(
                f"[{idx+1}] Empty question skipped"
            )

            continue

        logger.info(
            f"[{idx+1}/{len(dataset)}] "
            f"Running query..."
        )

        result = run_query(
            query_pipeline,
            question
        )

        answer = result.get(
            "answer",
            ""
        )

        if not is_valid_generation(answer):

            logger.warning(
                f"[{idx+1}] Invalid generation"
            )

            generated_outputs.append({

                "sample_id":
                    sample_id,

                "question":
                    question,

                "ground_truth_answer":
                    ground_truth,

                "generated_answer":
                    "INVALID_GENERATION",

                "retrieved_contexts":
                    [],

                "success":
                    False
            })

            continue

        # -------------------------------------------------
        # Extract retrieved contexts
        # -------------------------------------------------

        retrieved_chunks = result.get(
            "retrieved_chunks",
            []
        )

        contexts = []

        for chunk in retrieved_chunks:

            if isinstance(chunk, dict):

                text = clean_context(
                    chunk.get("text", "")
                )

                if text:
                    contexts.append(text)

            elif isinstance(chunk, str):

                text = clean_context(chunk)

                if text:
                    contexts.append(text)

        # fallback
        if not contexts:

            supporting_chunks = sample.get(
                "supporting_chunks",
                []
            )

            for chunk in supporting_chunks:

                if isinstance(chunk, dict):

                    text = clean_context(
                        chunk.get("text", "")
                    )

                    if text:
                        contexts.append(text)

        # final fallback
        if not contexts:

            contexts = [
                "No retrieved context available."
            ]

        generated_outputs.append({

            "sample_id":
                sample_id,

            "question":
                question,

            "ground_truth_answer":
                ground_truth,

            "generated_answer":
                answer,

            "retrieved_contexts":
                contexts,

            "num_contexts":
                len(contexts),

            "success":
                True
        })

        logger.info(
            f"[{idx+1}/{len(dataset)}] SUCCESS"
        )

    return generated_outputs


# =========================================================
# SAVE
# =========================================================

def save_outputs(
    outputs: List[Dict[str, Any]],
    output_path: str = None
):

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    if output_path is None:

        output_path = (
            ROOT_DIR /
            f"generated_rag_outputs_{timestamp}.json"
        )

    else:

        output_path = Path(output_path)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            outputs,
            f,
            indent=2,
            ensure_ascii=False
        )

    logger.info(
        "=" * 60
    )

    logger.info(
        f"Saved generated outputs:\n{output_path}"
    )

    logger.info(
        "=" * 60
    )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to generate"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output path"
    )

    args = parser.parse_args()

    logger.info(
        "=" * 60
    )

    logger.info(
        "RAG OUTPUT GENERATION PIPELINE"
    )

    logger.info(
        "=" * 60
    )

    dataset = load_dataset(
        num_samples=args.samples
    )

    outputs = generate_outputs(
        dataset
    )

    successful = sum(
        1 for x in outputs
        if x.get("success")
    )

    failed = len(outputs) - successful

    logger.info(
        f"Successful generations: {successful}"
    )

    logger.info(
        f"Failed generations: {failed}"
    )

    save_outputs(
        outputs,
        output_path=args.output
    )

    logger.info(
        "Generation pipeline completed."
    )


if __name__ == "__main__":
    main()