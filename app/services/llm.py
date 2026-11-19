import time
from functools import lru_cache

from google import genai
from google.genai import errors, types

from app.core.config import get_settings

_RETRIES = 4


class LLMService:
    def __init__(self) -> None:
        s = get_settings()
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._model = s.llm_model

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        for attempt in range(_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                    ),
                )
                return (response.text or "").strip()
            except errors.ServerError:
                if attempt == _RETRIES - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
        return ""


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()