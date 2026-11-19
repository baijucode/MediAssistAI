from app.core.config import get_settings
from app.services.llm import get_llm_service

s = get_settings()
print("model:", s.llm_model)
print("key set:", bool(s.gemini_api_key))

client = get_llm_service()._client
r = client.models.generate_content(
    model=s.llm_model,
    contents="Say hello in one short sentence.",
)
print("text:", repr(r.text))
print("raw:", r)