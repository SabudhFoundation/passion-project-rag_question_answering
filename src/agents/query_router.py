import json
import logging
import re

logger = logging.getLogger(__name__)


class QueryRouter:

    def __init__(self, llm_client):

        self.llm = llm_client

        self.system_prompt = """
You are an intelligent query routing agent.

Classify the user's query into ONE category:

1. greeting
2. small_talk
3. clarification
4. retrieval
5. coding
6. math

Rules:

- greeting:
  hi, hello, hey, good morning, good afternoon, good evening, good night, dear,sir,madam

- small_talk:
  tell me a joke
  who are you
  how are you

- clarification:
  vague queries such as:
  tell me about it
  explain this
  what happened there

- coding:
  write code
  debug code
  explain algorithms

- math:
 arithmetic
 calculations
 equations
 algebra
 statistics
 formulas

 Math queries DO NOT require retrieval.

 For all math queries return:

 {
  "intent":"math",
  "needs_retrieval":false
 }

- retrieval:
  factual knowledge requiring retrieval

Return ONLY valid JSON.

Example:

{
  "intent":"retrieval",
  "needs_retrieval":true
}
"""

    def is_followup_query(self,
        query: str
    ) -> bool:

        query = query.lower()
        
        VAGUE_TERMS = [
            "it",
            "this",
            "that",
            "these",
            "those",
            "he",
            "she",
            "they",
            "them",
            "more",
            "explain further",
            "something",
            "anything",
            "everything",
            "some",
            "any",
            "every"
        ]
        
        return any(
            term in query
            for term in VAGUE_TERMS
        )
    
    def should_rewrite(
        self,
        query: str,
        chat_history=None
    ) -> bool:
        """
        Decide whether a query should be rewritten before retrieval.

        Returns:
        True  -> rewrite query
        False -> use query as-is
        """

        query = query.strip().lower()

        if not query:
            return False

        # Follow-up / reference words
        followup_words = {
        "it",
        "they",
        "them",
        "he",
        "she",
        "that",
        "those",
        "this",
        "these",
        "its",
        "their",
        "his",
        "her"
        }

        words = query.split()

        # Very short queries are usually ambiguous
        if len(words) <= 4:
            return True

        # Contains follow-up references
        if any(word in followup_words for word in words):
            return True

        # Starts with follow-up phrases
        followup_phrases = [
        "tell me more",
        "explain more",
        "what about",
        "how about",
        "why is that",
        "can you explain",
        "elaborate",
        "give more details",
        "continue"
        ]

        if any(
            query.startswith(phrase)
            for phrase in followup_phrases
        ):
            return True

        # No previous conversation -> no need
        if not chat_history:
            return False

        return False
    
    def route(self, query: str):
        
        query = query.strip()

        # Fast math detection
        if re.fullmatch(
        r"[0-9\+\-\*/\(\)\.\s=]+",
        query
        ):
            return {
            "intent": "math",
            "needs_retrieval": False
            }

        # Otherwise call LLM
        try:

            prompt = f"""
User Query:
{query}
"""

            response = self.llm.invoke(
                self.system_prompt,
                prompt
            )

            result = json.loads(response)

            return result

        except Exception as e:

            logger.warning(
                f"Router failed: {e}"
            )

            return {
                "intent": "retrieval",
                "needs_retrieval": True
            }