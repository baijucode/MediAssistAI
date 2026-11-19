import json
from functools import lru_cache

from redis import Redis

from app.core.config import get_settings


class ChatMemory:
    def __init__(self) -> None:
        s = get_settings()
        self._redis = Redis.from_url(s.redis_url, decode_responses=True)
        self._max_messages = s.chat_history_turns * 2
        self._ttl = s.chat_ttl_seconds

    def check_connection(self) -> bool:
      return bool(self._redis.ping())

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:{session_id}"

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Oldest -> newest messages, e.g. [{"role": "user", "content": "..."}]."""
        raw = self._redis.lrange(self._key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        if self._max_messages == 0:
            self._redis.delete(key)
            return

        pipe = self._redis.pipeline()
        pipe.rpush(key, json.dumps({"role": role, "content": content}))
        pipe.ltrim(key, -self._max_messages, -1)
        pipe.expire(key, self._ttl)
        pipe.execute()


@lru_cache
def get_chat_memory() -> ChatMemory:
    return ChatMemory()