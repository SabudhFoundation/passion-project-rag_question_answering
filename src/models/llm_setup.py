"""
src/models/llm_setup.py
=======================
Handles initialization of the LLM client (Groq).

WHY GROQ?
  Groq provides free, ultra-fast inference for open-source LLMs
  like Llama 3.3 70B. No GPU needed — it runs on their cloud.

  Get your API key at: https://console.groq.com/keys

USAGE:
  from models.llm_setup import init_llm
  client = init_llm()
  response = client.chat.completions.create(...)
"""

import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from groq import Groq

import config
from logger import get_logger
from exceptions import GenerationError

logger = get_logger(__name__)


def init_llm() -> Groq:
    """
    Initializes the Groq LLM client.

    Reads the API key from config (which pulls from environment/.env).

    Returns:
        Groq client instance.

    Raises:
        GenerationError -- if API key is missing.
    """
    api_key = config.GROQ_API_KEY

    if not api_key:
        raise GenerationError(
            "GROQ_API_KEY is not set!\\n"
            "Get your free key at: https://console.groq.com/keys\\n"
            "Then set it:\\n"
            "  Windows: set GROQ_API_KEY=gsk_...\\n"
            "  Or add to .env file in project root"
        )

    logger.info("Groq LLM client initialised (model: %s)", config.GROQ_MODEL)
    return Groq(api_key=api_key)
