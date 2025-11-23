"""Caching layer for models and embeddings."""

import hashlib
import threading
import time
from typing import Any

from loguru import logger


class CacheEntry:
    """Cache entry with expiration."""

    def __init__(self, value: Any, ttl: float | None = None):
        """
        Initialize cache entry.

        Args:
            value: Cached value
            ttl: Time to live in seconds (None for no expiration)
        """
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def get_age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at


class Cache:
    """Thread-safe cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: float | None = None):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time to live in seconds
        """
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()

            ttl = ttl if ttl is not None else self.default_ttl
            self._cache[key] = CacheEntry(value, ttl)

    def delete(self, key: str) -> None:
        """
        Delete entry from cache.

        Args:
            key: Cache key
        """
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def _evict_oldest(self) -> None:
        """Evict oldest entry."""
        if not self._cache:
            return

        # Use min() to find oldest entry - O(n) but acceptable for cache eviction
        # For better performance with large caches, consider using OrderedDict
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }


class ModelCache:
    """Global cache for ML models (shared across instances)."""

    def __init__(self):
        """Initialize model cache."""
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_model(self, model_name: str, device: str = "cpu") -> Any | None:
        """
        Get cached model.

        Args:
            model_name: Model name/identifier
            device: Device (cpu/cuda)

        Returns:
            Model instance or None
        """
        key = f"{model_name}:{device}"
        with self._lock:
            return self._models.get(key)

    def set_model(self, model_name: str, model: Any, device: str = "cpu") -> None:
        """
        Cache model instance.

        Args:
            model_name: Model name/identifier
            model: Model instance
            device: Device (cpu/cuda)
        """
        key = f"{model_name}:{device}"
        with self._lock:
            self._models[key] = model

    def has_model(self, model_name: str, device: str = "cpu") -> bool:
        """Check if model is cached."""
        key = f"{model_name}:{device}"
        with self._lock:
            return key in self._models

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._models.clear()
            logger.info("Model cache cleared")


class EmbeddingCache:
    """Cache for text embeddings to avoid recomputing."""

    def __init__(self, max_size: int = 2000, ttl: float | None = None):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of cached embeddings
            ttl: Time to live in seconds (None for no expiration)
        """
        self._cache = Cache(max_size=max_size, default_ttl=ttl)

    def _make_key(self, text: str, model_name: str) -> str:
        """
        Create cache key from text and model.

        Args:
            text: Text to embed
            model_name: Model name

        Returns:
            Cache key
        """
        # Normalize text
        normalized_text = text.lower().strip()
        key_string = f"{model_name}:{normalized_text}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Any | None:
        """
        Get cached embedding.

        Args:
            text: Text that was embedded
            model_name: Model name used

        Returns:
            Cached embedding or None
        """
        key = self._make_key(text, model_name)
        return self._cache.get(key)

    def set(self, text: str, model_name: str, embedding: Any) -> None:
        """
        Cache embedding.

        Args:
            text: Text that was embedded
            model_name: Model name used
            embedding: Embedding vector
        """
        key = self._make_key(text, model_name)
        self._cache.set(key, embedding)

    def clear(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


# Global cache instances with thread-safe singleton pattern
_model_cache: ModelCache | None = None
_model_cache_lock = threading.Lock()
_embedding_cache: EmbeddingCache | None = None
_embedding_cache_lock = threading.Lock()


def get_model_cache() -> ModelCache:
    """Get global model cache instance (thread-safe)."""
    global _model_cache
    if _model_cache is None:
        with _model_cache_lock:
            if _model_cache is None:  # Double-check pattern
                _model_cache = ModelCache()
    return _model_cache


def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance (thread-safe)."""
    global _embedding_cache
    if _embedding_cache is None:
        with _embedding_cache_lock:
            if _embedding_cache is None:  # Double-check pattern
                _embedding_cache = EmbeddingCache()
    return _embedding_cache
