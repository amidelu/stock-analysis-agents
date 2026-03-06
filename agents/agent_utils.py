import os
import sys

from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from .groq_llm import GroqLlm
from google.api_core.retry import Retry, if_exception_type
from google.api_core import exceptions


def get_llm_model(**generation_kwargs) -> BaseLlm:
    """Returns the configured LLM model based on the AI_PROVIDER env var."""
    provider = os.getenv("AI_PROVIDER", "gemini").lower()

    # Retry configuration for robust API calls
    retry_config = Retry(
        initial=1.0,
        maximum=60.0,
        multiplier=2.0,
        deadline=300.0,
        predicate=if_exception_type(
            exceptions.TooManyRequests,
            exceptions.InternalServerError,
            exceptions.ServiceUnavailable,
            exceptions.GatewayTimeout,
        ),
    )

    if provider == "groq":
        model = GroqLlm(model="openai/gpt-oss-20b")
        model.generation_kwargs = generation_kwargs
        return model
    else:
        # Provide defaults that can be overridden
        config = {"temperature": 0.2}
        config.update(generation_kwargs)

        return Gemini(
            model="gemini-2.0-flash-exp",
            retry_options=retry_config,
            generation_config=config,
        )
