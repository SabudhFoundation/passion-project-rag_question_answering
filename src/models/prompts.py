"""
src/models/prompts.py
=====================
Defines all prompt templates used by the Generator.

WHY A SEPARATE FILE?
  Prompts are long strings that clutter the Generator class.
  Keeping them here makes it easy to:
    - A/B test different prompt versions
    - Share prompts across multiple generators
    - Review/edit prompts without touching Python logic

TEMPLATE VARIABLES:
  {context} — joined text from retrieved chunks
  {query}   — the user's original question
"""

# ─────────────────────────────────────────────────────────────────────────────
# RAG System Prompt — used by Generator.generate()
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """
You are an expert, highly reliable AI assistant. Your task is to answer the
user's question strictly using ONLY the information provided in the Context below.

Context Information:
{context}

User Question:
### {query} ###

Instructions:
1. Base your answer ONLY on the provided Context Information.
2. If the Context contains conflicting information, point out the conflict.
3. Output your response in valid JSON with exactly two keys:
   - "answer_found": boolean -- false ONLY if context lacks the answer.
   - "answer": string -- your detailed response.
"""
