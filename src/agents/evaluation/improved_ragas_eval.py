#!/usr/bin/env python3

"""
src/evaluation/evaluate_saved_outputs.py
===========================================================
RAG EVALUATION PIPELINE (NO RE-GENERATION)
===========================================================

Purpose
-------
This file evaluates ALREADY generated RAG outputs.

It does NOT:
✗ call QueryPipeline
✗ hit Pinecone
✗ regenerate answers

It ONLY:
✓ loads generated outputs JSON
✓ evaluates metrics
✓ computes averages
✓ exports detailed reports

-----------------------------------------------------------
USAGE
-----------------------------------------------------------

Evaluate latest generated file:
python src/evaluation/evaluate_saved_outputs.py

Evaluate 20 samples:
python src/evaluation/evaluate_saved_outputs.py \
    --samples 20

Evaluate custom file:
python src/evaluation/evaluate_saved_outputs.py \
    --input generated_rag_outputs.json

Custom threshold:
python src/evaluation/evaluate_saved_outputs.py \
    --threshold 0.75

-----------------------------------------------------------
OUTPUT FILES
-----------------------------------------------------------

1. ragas_eval_TIMESTAMP.json

2. low_quality_samples_TIMESTAMP.json

===========================================================
"""

import os
import re
import json
import glob
import time
import logging
import argparse
import google.generativeai as genai

from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# PATH SETUP
# =========================================================

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "gemini-3.1-flash-lite"

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY not found in .env"
    )

genai.configure(
    api_key=GOOGLE_API_KEY
)

judge_model = genai.GenerativeModel(
    MODEL_NAME
)

MAX_RETRIES = 3

REQUEST_TIMEOUT = 60

DEFAULT_THRESHOLD = 0.7

MAX_CONTEXT_CHARS = 5000

MAX_WORKERS = 1

# =========================================================
# HELPERS
# =========================================================

def safe_mean(values: List[float]) -> float:

    if not values:
        return 0.0

    return sum(values) / len(values)


def extract_score(text: str) -> float:

    if not text:
        return 0.5

    text = text.strip()

    matches = re.findall(
        r"0(?:\.\d+)?|1(?:\.0+)?",
        text
    )

    if matches:

        try:

            score = float(matches[0])

            return max(
                0.0,
                min(1.0, score)
            )

        except:
            pass

    matches = re.findall(
        r"\d+\.\d+|\d+",
        text
    )

    if matches:

        try:

            score = float(matches[0])

            if score > 1:
                score /= 10

            return max(
                0.0,
                min(1.0, score)
            )

        except:
            pass

    return 0.5


# =========================================================
# LOAD GENERATED OUTPUTS
# =========================================================

def load_generated_outputs(
    input_path: str = None,
    num_samples: int = None,
    offset: int = 0
) -> List[Dict[str, Any]]:

    if input_path is None:

        pattern = str(
            ROOT_DIR /
            "generated_rag_outputs_*.json"
        )

        files = glob.glob(pattern)

        if not files:

            raise FileNotFoundError(
                "\nNo generated outputs found.\n"
                "Run improved_ragas_eval.py first."
            )

        input_path = max(
            files,
            key=os.path.getctime
        )

    logger.info(
        f"Loading generated outputs:\n{input_path}"
    )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    data = [
        x for x in data
        if isinstance(x, dict)
    ]

    # keep only successful generations
    data = [
        x for x in data
        if x.get("success") is True
    ]

    start = offset

    if num_samples:
        data = data[start:start + num_samples]
    else:
        data = data[start:]
    
    logger.info(
        f"Loaded {len(data)} generated samples"
    )

    return data


# =========================================================
# GEMINI FLASH CALL
# =========================================================

def evaluate_with_gemini(
    question: str,
    answer: str,
    ground_truth: str,
    context_text: str
) -> Dict[str, float]:

    prompt = f"""
You are an expert evaluator for Retrieval-Augmented Generation systems.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

GENERATED ANSWER:
{answer}

RETRIEVED CONTEXT:
{context_text}

Evaluate:

1. faithfulness
   How well answer is supported by context.

2. answer_relevancy
   How directly answer addresses the question.

3. context_recall
   Whether retrieved context contains required information.

4. context_precision
   Relevance of retrieved context.

5. answer_correctness
   Similarity to ground truth.

Return ONLY valid JSON.

{{
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "context_recall": 0.0,
    "context_precision": 0.0,
    "answer_correctness": 0.0
}}

All values must be between 0 and 1.
"""

    for attempt in range(MAX_RETRIES):

        try:

            response = judge_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0
                }
            )

            text = response.text.strip()

            text = (
                text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            scores = json.loads(text)

            time.sleep(5)

            return {

                "faithfulness":
                    float(
                        scores.get(
                            "faithfulness",
                            0.5
                        )
                    ),

                "answer_relevancy":
                    float(
                        scores.get(
                            "answer_relevancy",
                            0.5
                        )
                    ),

                "context_recall":
                    float(
                        scores.get(
                            "context_recall",
                            0.5
                        )
                    ),

                "context_precision":
                    float(
                        scores.get(
                            "context_precision",
                            0.5
                        )
                    ),

                "answer_correctness":
                    float(
                        scores.get(
                            "answer_correctness",
                            0.5
                        )
                    )
            }

        except Exception as e:

            error_text = str(e)

            if "429" in error_text:

                retry_match = re.search(
                    r"retry in ([0-9.]+)s",
                    error_text,
                    re.IGNORECASE
                )

                if retry_match:

                    wait_time = float(
                        retry_match.group(1)
                    ) + 2

                else:

                    wait_time = 30

                logger.warning(
                    f"Rate limited. Sleeping "
                    f"{wait_time:.1f}s"
                )

                time.sleep(wait_time)

                continue

            logger.warning(
                f"Gemini evaluation failed: {e}"
            )


# =========================================================
# SAMPLE EVALUATION
# =========================================================

def evaluate_sample(
    sample: Dict[str, Any]
) -> Dict[str, Any]:

    question = sample["question"]

    answer = sample["generated_answer"]

    ground_truth = sample["ground_truth_answer"]

    contexts = sample["retrieved_contexts"]

    context_text = "\n\n".join(
        contexts[:5]
    )[:MAX_CONTEXT_CHARS]

    scores = evaluate_with_gemini(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        context_text=context_text
    )

    return {

        "sample_id":
            sample["sample_id"],

        "question":
            question,

        "ground_truth_answer":
            ground_truth,

        "generated_answer":
            answer,

        "retrieved_contexts":
            contexts,

        "faithfulness":
            scores["faithfulness"],

        "answer_relevancy":
            scores["answer_relevancy"],

        "context_recall":
            scores["context_recall"],

        "context_precision":
            scores["context_precision"],

        "answer_correctness":
            scores["answer_correctness"]
    }


# =========================================================
# MAIN EVALUATION
# =========================================================

def run_evaluation(
    generated_outputs: List[Dict[str, Any]],
    threshold: float
):

    detailed_results = []

    logger.info(
        "=" * 60
    )

    logger.info(
        f"Evaluating "
        f"{len(generated_outputs)} samples"
    )

    logger.info(
        "=" * 60
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        detailed_results = list(
            executor.map(
                evaluate_sample,
                generated_outputs
            )
        )
    # -----------------------------------------------------
    # FILTER LOW QUALITY
    # -----------------------------------------------------

    kept_samples = []

    low_quality_samples = []

    for sample in detailed_results:

        if (
            sample["faithfulness"] >= threshold
            and sample["answer_relevancy"] >= 0.5
            and sample["answer_correctness"] >= 0.5
        ):

            kept_samples.append(sample)

        else:

            low_quality_samples.append(sample)

    evaluation_set = (
        kept_samples
        if kept_samples
        else detailed_results
    )

    # -----------------------------------------------------
    # AGGREGATE
    # -----------------------------------------------------

    avg_faithfulness = safe_mean([
        s["faithfulness"]
        for s in evaluation_set
    ])

    avg_relevancy = safe_mean([
        s["answer_relevancy"]
        for s in evaluation_set
    ])

    avg_recall = safe_mean([
        s["context_recall"]
        for s in evaluation_set
    ])

    avg_precision = safe_mean([
        s["context_precision"]
        for s in evaluation_set
    ])
    
    avg_correctness = safe_mean([
        s["answer_correctness"]
        for s in evaluation_set
    ])
    
    overall = safe_mean([

    avg_faithfulness,
    avg_relevancy,
    avg_recall,
    avg_precision,
    avg_correctness
    ])

    return {

        "summary": {

            "faithfulness":
                avg_faithfulness,

            "answer_relevancy":
                avg_relevancy,

            "context_recall":
                avg_recall,

            "context_precision":
                avg_precision,

            "answer_correctness":
                avg_correctness,

            "overall_score":
                overall,

            "total_samples":
                len(generated_outputs),

            "kept_samples":
                len(kept_samples),

            "removed_low_quality_samples":
                len(low_quality_samples),

            "judge_model":
                MODEL_NAME
        },

        "detailed_results":
            evaluation_set,

        "low_quality_samples":
            low_quality_samples
    }


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(results: Dict[str, Any]):

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        ROOT_DIR /
        f"ragas_eval_{timestamp}.json"
    )

    low_quality_path = (
        ROOT_DIR /
        f"low_quality_samples_{timestamp}.json"
    )

    cumulative_path = (
        ROOT_DIR /
        "all_evaluation_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(
        low_quality_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results["low_quality_samples"],
            f,
            indent=2,
            ensure_ascii=False
        )

    if cumulative_path.exists():

        with open(
            cumulative_path,
            "r",
            encoding="utf-8"
        ) as f:

            try:
                cumulative = json.load(f)
            except:
                cumulative = []

    else:

        cumulative = []

    cumulative.extend(
        results["detailed_results"]
    )

    with open(
        cumulative_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cumulative,
            f,
            indent=2,
            ensure_ascii=False
        )

    logger.info(
        "=" * 60
    )

    logger.info(
        f"Saved evaluation:\n{output_path}"
    )

    logger.info(
        f"Saved low quality samples:\n"
        f"{low_quality_path}"
    )

    logger.info(
        f"Updated cumulative file:\n"
        f"{cumulative_path}"
    )

    logger.info(
        "=" * 60
    )


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(results: Dict[str, Any]):

    summary = results["summary"]

    print("\n" + "=" * 70)

    print("RAG EVALUATION RESULTS")

    print("=" * 70)

    print(
        f"Total Samples: "
        f"{summary['total_samples']}"
    )

    print(
        f"Kept Samples: "
        f"{summary['kept_samples']}"
    )

    print(
        f"Removed Low Quality Samples: "
        f"{summary['removed_low_quality_samples']}"
    )

    print("\nMetric Scores")

    print("-" * 70)

    print(
        f"Faithfulness: "
        f"{summary['faithfulness']:.3f}"
    )

    print(
        f"Answer Relevancy: "
        f"{summary['answer_relevancy']:.3f}"
    )

    print(
        f"Context Recall: "
        f"{summary['context_recall']:.3f}"
    )

    print(
        f"Context Precision: "
        f"{summary['context_precision']:.3f}"
    )

    print(
        f"Answer Correctness: "
        f"{summary['answer_correctness']:.3f}"
    )

    print("-" * 70)

    print(
        f"Overall Score: "
        f"{summary['overall_score']:.3f}"
    )

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Generated outputs JSON"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Evaluate first N samples"
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting sample index"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD
    )

    args = parser.parse_args()

    generated_outputs = load_generated_outputs(
        input_path=args.input,
        num_samples=args.samples,
        offset=args.offset
    )

    results = run_evaluation(
        generated_outputs,
        threshold=args.threshold
    )

    print_results(results)

    save_results(results)

    logger.info(
        "Evaluation completed successfully."
    )


if __name__ == "__main__":
    main()