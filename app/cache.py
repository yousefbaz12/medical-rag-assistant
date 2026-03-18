import hashlib
import time

from app.config import settings

_cache_store = {}


def make_cache_key(session_id: str, question: str) -> str:
    raw = f"{session_id}:{question}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def get_cached_value(key: str):
    item = _cache_store.get(key)
    if not item:
        return None

    value, expires_at = item
    if time.time() > expires_at:
        del _cache_store[key]
        return None

    return value


def set_cached_value(key: str, value: dict, ttl: int | None = None):
    ttl = ttl or settings.cache_ttl_seconds
    _cache_store[key] = (value, time.time() + ttl)