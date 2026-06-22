import os
from groq import Groq


class RouterLLM:

    def __init__(self):

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.model = "llama-3.1-8b-instant"

    def invoke(
        self,
        system_prompt,
        user_prompt
    ):

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0
        )

        return (
            response
            .choices[0]
            .message
            .content
        )