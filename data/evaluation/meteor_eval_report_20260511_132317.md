# RAG Generation Evaluation Report -- METEOR Score

**Date:** 2026-05-11 13:24:14
**Source CSV:** `generation_eval_summary_20260509_210627.csv`
**Total Questions Evaluated:** 295

## What is METEOR?

> METEOR (Metric for Evaluation of Translation with Explicit ORdering) scores
> how well a generated answer aligns with a reference answer by matching words
> using **exact match**, **stemming** (run/running), and **WordNet synonyms**
> (car/automobile). It then applies a **fragmentation penalty** to reward
> answers whose matching words appear in the correct order.

## Overall METEOR Performance

| Metric | Value |
|---|---|
| **Mean METEOR Score** | 0.5414 |
| **Min Score**         | 0.0000 |
| **Max Score**         | 1.0000 |
| **Total Samples**     | 295 |

## Interpretation

> **Excellent** -- Answers are highly aligned with references (synonyms & stems matched).

## Performance by Difficulty

| Difficulty | Mean METEOR | Min | Max | Count |
|---|---|---|---|---|
| Easy | 0.5987 | 0.0000 | 1.0000 | 100 |
| Medium | 0.5289 | 0.0000 | 0.9922 | 98 |
| Hard | 0.4948 | 0.0000 | 0.9990 | 97 |

## Comparison with BERTScore

| Metric | Mean Score |
|---|---|
| **METEOR**         | 0.5414 |
| **BERTScore F1**    | 0.9092 |

> **Note:** METEOR scores are typically lower than BERTScore F1 because
> BERTScore uses deep contextual embeddings while METEOR uses lexical
> matching with WordNet. Both metrics are complementary.

## Sample Generations

### Top Performing Examples

**1. Question:** What was the role of producing an autograph in Assyriology? *(Difficulty: easy)*

- **Reference Answer:** Producing an autograph is often the first step of a tablet's archaeological interpretation and the autograph is frequently the authoritative form that is published as source material.
- **Generated Answer:** Producing an autograph is often the first step of a tablet's archaeological interpretation and the autograph is frequently the authoritative form that is published as source material.
- **METEOR Score:** 1.0000

---

**2. Question:** What event marked the end of the time period during which Chang Ucchin was born and under which Korea was ruled by Japan? *(Difficulty: hard)*

- **Reference Answer:** The conclusion of World War II in 1945
- **Generated Answer:** the conclusion of World War II in 1945
- **METEOR Score:** 0.9990

---

**3. Question:** What is the best known song of the Californian rock band Lit? *(Difficulty: easy)*

- **Reference Answer:** My Own Worst Enemy
- **Generated Answer:** My Own Worst Enemy
- **METEOR Score:** 0.9922

---

### Lowest Performing Examples

**1. Question:** What is the population of the city where James Iroha Uchechukwu was born, according to the 2006 census? *(Difficulty: medium)*

- **Reference Answer:** 722,664
- **Generated Answer:** "Population data for the city of Enugu is not available in the provided context for the 2006 census."
- **METEOR Score:** 0.0000

---

**2. Question:** What was the record of the Utah Jazz in the 2000-01 NBA season? *(Difficulty: easy)*

- **Reference Answer:** 53-29
- **Generated Answer:** The Utah Jazz had a 53–29 record in the 2000-01 NBA season.
- **METEOR Score:** 0.0000

---

**3. Question:** When was Chris Noonan born? *(Difficulty: easy)*

- **Reference Answer:** 14 November 1952
- **Generated Answer:** The Context Information doesn't provide the birth date of Chris Noonan. It only mentions that he was nominated for an award for his work in 'Babe'.
- **METEOR Score:** 0.0000

---
