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

TEMPLATE VARIABLES (backward-compatible with Generator):
  {context} — joined text from retrieved chunks, each labelled
               [chunk_N | Doc-ID | Title] so the model can cite them
               precisely by number, doc_id, and title.
  {query}   — the user's original question

DESIGN PHILOSOPHY — Prompt Length:
  We use a MEDIUM-LENGTH prompt: not a one-liner, not a 2 000-token wall.
  Rationale:
    • Too short → model ignores grounding & hallucinates freely
    • Too long  → model loses track of instructions mid-prompt (lost-in-the-
                  middle effect), wastes tokens on every call, and slows
                  Groq's rate-limit budget
    • Medium    → every rule fits in the model's attention hotspot, leaving
                  the majority of the context window for retrieved chunks
                  and the actual answer

UPGRADE NOTES (v4 — CoT + Citation Verification):
  - Scratchpad is now wrapped in <thinking>…</thinking> XML tags.
  - Every intermediate reasoning step must cite the source chunk inline:
      [chunk_N | Title] or [chunk_N]
  - Dedicated CITATION VERIFICATION step: model must re-read each cited
      chunk and confirm the fact before writing the final answer.
  - Final answer must NOT include any inline citations or bracketed numbers.
  - The answer must be clean prose.
  - Output contract: JSON with "scratchpad", "citations", "answer_found",
      "answer".
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — context builder instruction (injected into the prompt header so the
# model knows how the chunks are formatted)
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_FORMAT_NOTE = (
    "Each chunk is labelled  [chunk_N | doc_id | Title]  where N is the chunk "
    "number, doc_id is the unique document identifier, and Title is the source "
    "article title. Use these labels to cite evidence in your scratchpad reasoning."
)

# ─────────────────────────────────────────────────────────────────────────────
# RAG System Prompt — used by Generator.generate()
# Drop-in replacement: same {context} and {query} variables.
# NEW output key: "scratchpad" (reasoning trace before the final answer).
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are a precise, grounded AI assistant inside a Retrieval-Augmented \
Generation (RAG) pipeline built on the HotpotQA dataset.

HotpotQA questions often require multi-hop reasoning — you must chain \
facts across two or more chunks to reach the correct answer. \
Many questions also involve arithmetic, date calculations, comparisons, \
or counting. Follow the rules below carefully.

══════════════════════════════════════════
RETRIEVED CONTEXT  ({_CHUNK_FORMAT_NOTE})
══════════════════════════════════════════
{{context}}
══════════════════════════════════════════

USER QUESTION:
{{query}}

══════════════════════════════════════════
MANDATORY CHAIN-OF-THOUGHT REASONING
(write this inside your scratchpad field, wrapped in <thinking>…</thinking>)
══════════════════════════════════════════

STEP 1 — SURVEY & FILTER
  Read every chunk header ([chunk_N | doc_id | Title]).
  List which chunks are relevant and which are off-topic. \
For off-topic chunks write "chunk_N → irrelevant".

STEP 2 — MULTI-HOP BRIDGING (cite every step)
  Most questions require chaining facts across TWO or more chunks.
  a. State the BRIDGE ENTITY explicitly: the shared person, place, event, \
or concept that connects Hop 1 to Hop 2.
  b. For each hop, write:
       Hop 1: "<fact extracted from chunk_N>" → cited as [chunk_N | Title]
       Hop 2: "<fact extracted from chunk_M>" → cited as [chunk_M | Title]
  c. State the chain: "Bridge entity X links chunk_N to chunk_M."

STEP 3 — ARITHMETIC & COMPARISON (critical)
  If the question involves numbers, dates, ages, durations, or ranks:
  a. Extract each number verbatim: "X = <value> [chunk_N | Title]"
  b. Write the exact operation: e.g. "2024 − 1990 = 34"
  c. Double-check the arithmetic before writing the answer.
  d. NEVER estimate or round unless the source text already does so.
  e. If units differ (years vs months, km vs miles), convert explicitly.

STEP 4 — COMPARISON / AGGREGATION (cite sources)
  If asked "which is larger / older / longer / more recent":
  a. List candidates and their values with chunk citations:
       "A = X [chunk_N | Title]; B = Y [chunk_M | Title]"
  b. State your comparison: "A (X) > B (Y), therefore A."

STEP 5 — CITATION VERIFICATION (mandatory — prevents hallucination)
  Before writing the final answer, re-read EVERY chunk you cited and \
verify:
  a. The exact quoted fact is present verbatim (or close paraphrase) \
in that chunk.
  b. You have NOT merged facts from two different chunks without \
acknowledging both.
  c. You have NOT added any fact that does not appear in the context.
  For each citation write: "✓ chunk_N confirms: <exact phrase from chunk>"
  If a fact cannot be confirmed: "✗ chunk_N does NOT contain this — removing \
claim."

STEP 6 — CONFLICTS & GAPS
  • Conflicting chunks → surface both:
      "chunk_N says X [chunk_N | Title]; chunk_M says Y [chunk_M | Title]."
  • Missing info → set answer_found: false and state the exact gap.
  • Partial answers → answer_found: true; state what IS answered and what \
is missing.

══════════════════════════════════════════
OUTPUT FORMAT  (return valid JSON only)
══════════════════════════════════════════

Return this exact JSON structure and nothing else:

{{{{
  "scratchpad": "<thinking>\nSTEP 1 — SURVEY & FILTER\n…\nSTEP 2 — MULTI-HOP BRIDGING\n…\nSTEP 3 — ARITHMETIC (if applicable)\n…\nSTEP 4 — COMPARISON (if applicable)\n…\nSTEP 5 — CITATION VERIFICATION\n✓ chunk_N confirms: <exact phrase>\n…\nSTEP 6 — CONFLICTS & GAPS\n…\n</thinking>",
  "citations": ["chunk_N | doc_id | Title", "chunk_M | doc_id | Title"],
  "answer_found": <true | false>,
  "answer": "<final answer as a clean, complete sentence or short phrase. Do NOT include any inline citations or bracketed numbers (e.g. '[chunk_N]' or '[N]') in this field. All citation tracking is handled in the 'citations' array and 'scratchpad' fields.>"
}}}}

JSON RULES (strict):
- Return ONLY the JSON object. No markdown fences, no preamble, no trailing text.
- "scratchpad" MUST contain the full <thinking>…</thinking> block. \
Never omit it, even for simple questions.
- "citations" MUST list every chunk_N label you used in reasoning. \
Omit chunks that were marked irrelevant in STEP 1.
- "answer_found" is false ONLY when the context genuinely cannot answer \
the question. Partial answers → true.
- "answer" MUST NOT include any inline citations, bracketed numbers, or labels (such as [chunk_N] or similar). It must be clean, natural language prose.
- Keep "answer" concise: one sentence for factual lookups, ≤3 sentences \
for multi-part or comparative questions.
- Escape internal double-quotes with \\.
""".format(_CHUNK_FORMAT_NOTE=_CHUNK_FORMAT_NOTE)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: No-Context Fallback Prompt
# Use when retrieval returns 0 chunks or all scores fall below threshold.
#
# Usage in generator.py:
#   if not chunks:
#       prompt = NO_CONTEXT_PROMPT.format(query=query)
#   else:
#       prompt = RAG_SYSTEM_PROMPT.format(context=context, query=query)
# ─────────────────────────────────────────────────────────────────────────────

NO_CONTEXT_PROMPT = """\
You are an AI assistant inside a Retrieval-Augmented Generation pipeline.

The retrieval system returned NO relevant documents for the user's question.

USER QUESTION:
{query}

Do NOT answer from general knowledge. Return valid JSON and nothing else:
{{{{
  "scratchpad": "No chunks were retrieved. Cannot reason from context.",
  "answer_found": false,
  "answer": "No relevant information was found in the knowledge base for \
this question. Try rephrasing or check that the topic is covered in the \
indexed documents."
}}}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE: Query Rewriting Prompt  (pre-retrieval step)
# Called by the Smart Query Router in app.py BEFORE hitting the vector DB.
# Only triggered when the router detects pronouns OR keyword overlap with
# recent history — keeping latency near-zero for standalone questions.
#
# Uses a small fast model (llama-3-8b-8192) via Groq so the rewrite adds
# only ~200ms, leaving the big model exclusively for answer generation.
#
# Usage in app.py:
#   rewrite_prompt = QUERY_REWRITE_PROMPT.format(query=raw_query, history=history_str)
#   rewritten = _call_small_llm(rewrite_prompt)   # returns JSON array string
#   best_query = json.loads(rewritten)[0]
# ─────────────────────────────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = """\
You are a search-query optimizer for a RAG pipeline backed by a vector \
database indexed on HotpotQA documents.

══════════════════════════════════
RECENT CONVERSATION HISTORY
══════════════════════════════════
{history}
══════════════════════════════════

USER'S LATEST MESSAGE:
{query}

══════════════════════════════════
YOUR TASK
══════════════════════════════════
Rewrite the user's latest message into 1–3 standalone, self-contained \
retrieval queries that can be understood WITHOUT the conversation history.

Rules:
1. PRONOUN RESOLUTION — Replace all pronouns (he, she, it, they, this, that, \
   its, their, him, her, those) with the exact named entity from the history.
2. TOPIC CONTINUITY — If the new question is about the same topic discussed \
   in the history (same person, place, event, or concept), include that \
   entity name even if the user did not repeat it.
3. KEYWORD ENRICHMENT — Use domain-specific keywords and entity names likely \
   to match document chunks. Avoid vague words like "the thing" or "that place".
4. MULTI-HOP SPLIT — For questions needing facts from multiple sources, \
   generate one query per fact hop (e.g., one query for the person's birth \
   year, one for the city's population).
5. STANDALONE — Every rewritten query must make complete sense on its own \
   with zero context from the history.

Return ONLY a JSON array of strings. No explanation, no extra text.
Example output: ["Who directed the 1997 film Titanic?", "James Cameron filmography"]
"""
