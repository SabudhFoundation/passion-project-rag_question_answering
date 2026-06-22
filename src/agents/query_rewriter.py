import logging

logger = logging.getLogger(__name__)


class QueryRewriter:

    def __init__(self, llm_client):

        self.llm = llm_client

        self.system_prompt = """
You are a query rewriting agent.

Rewrite the user query into a
standalone search query.

Rules:

1. Preserve meaning.
2. Expand references like:
   it
   this
   that
   they

3. Use conversation history.

4. Return ONLY the rewritten query.
"""

    def rewrite(
        self,
        query,
        history=""
    ):

        try:

            prompt = f"""
Conversation:

{history}

Current Query:

{query}

Rewrite:
"""

            rewritten = self.llm.invoke(
                self.system_prompt,
                prompt
            )

            return rewritten.strip()

        except Exception as e:

            logger.warning(
                f"Rewrite failed: {e}"
            )

            return query

    def rewrite_with_history(
        self,
        query,
        history
    ):

        if not history:
            return query

        recent_context = "\n".join(
            [
                f"User: {x['query']}"
                for x in history[-3:]
            ]
        )

        prompt = f"""
Conversation History:

{recent_context}

Current Query:
{query}

Rewrite the current query into a
standalone search query.

Return only the rewritten query.
"""

        return self.llm.invoke(prompt).strip()