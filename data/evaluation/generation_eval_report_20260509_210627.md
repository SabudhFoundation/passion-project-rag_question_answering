# RAG Generation Evaluation Report

**Date:** 2026-05-09 21:38:31
**BERT Model Used:** `roberta-large`
**Total Questions Evaluated:** 295

## Overall Performance

| Metric | Score |
|---|---|
| **Precision** | 0.8903 |
| **Recall**    | 0.9298 |
| **F1 Score**  | 0.9092 |

## Performance by Difficulty

| Difficulty | Precision | Recall | F1 Score | Count |
|---|---|---|---|---|
| Easy | 0.9108 | 0.9423 | 0.9260 | 100 |
| Medium | 0.8847 | 0.9302 | 0.9063 | 98 |
| Hard | 0.8749 | 0.9164 | 0.8948 | 97 |

## Interpretation

> **Excellent** -- Generated answers are highly aligned with references.

## Sample Generations

### Top Performing Examples

**1. Question:** In what year was the film Saludos Amigos released in the United States? (Difficulty: easy)

- **Reference Answer:** 1943
- **Generated Answer:** 1943
- **Scores:** Precision: 1.0000, Recall: 1.0000, F1: 1.0000

---

**2. Question:** What year was the company that co-developed and distributed the BlackBerry DTEK60 founded? (Difficulty: medium)

- **Reference Answer:** 1984
- **Generated Answer:** 1984
- **Scores:** Precision: 1.0000, Recall: 1.0000, F1: 1.0000

---

**3. Question:** What is the name of the forensic psychiatrist who has consulted or testified in many of the highest profile US criminal cases? (Difficulty: easy)

- **Reference Answer:** Park Dietz
- **Generated Answer:** Park Dietz
- **Scores:** Precision: 1.0000, Recall: 1.0000, F1: 1.0000

---

### Lowest Performing Examples

**1. Question:** How many Grand Slam doubles titles did Jonathan Stark win? (Difficulty: easy)

- **Reference Answer:** 2
- **Generated Answer:** Jonathan Stark won two Grand Slam doubles titles (the 1994 French Open Men's Doubles and the 1995 Wimbledon Championships Mixed Doubles).
- **Scores:** Precision: 0.7536, Recall: 0.8395, F1: 0.7943

---

**2. Question:** How old is the female main protagonist of The Hunger Games? (Difficulty: easy)

- **Reference Answer:** 16
- **Generated Answer:** At the beginning of the story, the female main protagonist is 16 years old. As specified in the context information: 'The Hunger Games is a 2008 dystopian novel by the American writer Suzanne Collins. It is written in the voice of 16-year-old Katniss Everdeen...'
- **Scores:** Precision: 0.7705, Recall: 0.8210, F1: 0.7949

---

**3. Question:** What is the time span between the birth of Joseph Roth and the death of Robert E. Howard? (Difficulty: hard)

- **Reference Answer:** 42 years
- **Generated Answer:** Joseph Roth was born on 2 September 1894, and Robert E. Howard died on 12 June 1936, indicating a 7.4-year difference in their lifetimes. However, to answer your question accurately: Robert E. Howard was born in 1906 not provided in context and Joseph Roth in 1894 also not provided in context, so, actual time span can't be determined using this information. Thus the exact time span between their birth and death can't be accurately found in context with the available information.
- **Scores:** Precision: 0.7827, Recall: 0.8292, F1: 0.8053

---

