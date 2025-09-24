"""
Intelligent Caching and Resource Optimization for Felix Framework

This module provides advanced caching strategies and resource optimization
specifically designed for HuggingFace Pro accounts and ZeroGPU deployments.

Features:
- Multi-tier caching with semantic similarity matching
- GPU memory optimization and automatic cleanup
- Predictive pre-loading of popular models
- Request deduplication and batch optimization
- Cost-aware caching strategies
- Adaptive cache sizing based on usage patterns
- Redis integration for distributed caching
- LRU with priority scoring for cache eviction
"""

import os
import json
import asyncio
import logging
import time
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import numpy as np
from abc import ABC, abstractmethod

# Optional Redis for distributed caching
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Optional sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the intelligent cache."""
    key: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    cost_to_generate: float = 0.0
    quality_score: float = 0.0
    model_id: str = ""
    agent_type: str = ""
    tokens_used: int = 0
    response_time: float = 0.0
    embedding: Optional[np.ndarray] = None
    priority_score: float = 0.0


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    semantic_hits: int = 0
    cost_savings: float = 0.0
    time_savings: float = 0.0
    storage_used: int = 0  # bytes
    evictions: int = 0
    hit_rate: float = 0.0
    semantic_hit_rate: float = 0.0
    avg_retrieval_time: float = 0.0


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        pass

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry, ttl: Optional[int] = None):
        """Set cache entry with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str):
        """Delete cache entry."""
        pass

    @abstractmethod
    async def clear(self):
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get cache size in bytes."""
        pass

    @abstractmethod
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction."""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size

    async def get(self, key: str) -> Optional[CacheEntry]:
        if key in self.cache:
            # Move to end (most recently used)
            entry = self.cache.pop(key)
            self.cache[key] = entry
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            return entry
        return None

    async def set(self, key: str, entry: CacheEntry, ttl: Optional[int] = None):
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)

        self.cache[key] = entry
        if key != list(self.cache.keys())[-1]:
            # Move to end if not already there
            self.cache.move_to_end(key)

    async def delete(self, key: str):
        self.cache.pop(key, None)

    async def clear(self):
        self.cache.clear()

    async def size(self) -> int:
        return sum(len(pickle.dumps(entry)) for entry in self.cache.values())

    async def keys(self) -> List[str]:
        return list(self.cache.keys())


class RedisCacheBackend(CacheBackend):
    """Redis-based distributed cache backend."""

    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "felix_cache:"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install: pip install redis")

        self.redis_url = redis_url
        self.prefix = prefix
        self.redis: Optional[aioredis.Redis] = None

    async def _ensure_connection(self):
        if not self.redis:
            self.redis = aioredis.from_url(self.redis_url, decode_responses=False)

    async def get(self, key: str) -> Optional[CacheEntry]:
        await self._ensure_connection()
        try:
            data = await self.redis.get(f"{self.prefix}{key}")
            if data:
                entry = pickle.loads(data)
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                return entry
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
        return None

    async def set(self, key: str, entry: CacheEntry, ttl: Optional[int] = None):
        await self._ensure_connection()
        try:
            data = pickle.dumps(entry)
            if ttl:
                await self.redis.setex(f"{self.prefix}{key}", ttl, data)
            else:
                await self.redis.set(f"{self.prefix}{key}", data)
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    async def delete(self, key: str):
        await self._ensure_connection()
        try:
            await self.redis.delete(f"{self.prefix}{key}")
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")

    async def clear(self):
        await self._ensure_connection()
        try:
            keys = await self.redis.keys(f"{self.prefix}*")
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")

    async def size(self) -> int:
        await self._ensure_connection()
        try:
            memory_info = await self.redis.info("memory")
            return memory_info.get("used_memory", 0)
        except Exception as e:
            logger.warning(f"Redis size failed: {e}")
            return 0

    async def keys(self) -> List[str]:
        await self._ensure_connection()
        try:
            keys = await self.redis.keys(f"{self.prefix}*")
            return [key.decode().replace(self.prefix, "") for key in keys]
        except Exception as e:
            logger.warning(f"Redis keys failed: {e}")
            return []


class IntelligentCache:
    """
    Intelligent caching system for Felix Framework.

    Provides multi-tier caching with semantic similarity, cost optimization,
    and adaptive resource management for HuggingFace Pro deployments.
    """

    def __init__(self,
                 backend: Optional[CacheBackend] = None,
                 enable_semantic_similarity: bool = True,
                 semantic_threshold: float = 0.85,
                 max_cache_size_mb: int = 512,
                 ttl_hours: int = 24,
                 cost_optimization: bool = True,
                 adaptive_sizing: bool = True):
        """
        Initialize intelligent cache.

        Args:
            backend: Cache backend (defaults to memory)
            enable_semantic_similarity: Enable semantic similarity matching
            semantic_threshold: Similarity threshold for semantic matches
            max_cache_size_mb: Maximum cache size in MB
            ttl_hours: Time to live for cache entries in hours
            cost_optimization: Enable cost-aware caching
            adaptive_sizing: Enable adaptive cache sizing
        """
        self.backend = backend or MemoryCacheBackend()
        self.enable_semantic_similarity = enable_semantic_similarity
        self.semantic_threshold = semantic_threshold
        self.max_cache_size_mb = max_cache_size_mb
        self.ttl_hours = ttl_hours
        self.cost_optimization = cost_optimization
        self.adaptive_sizing = adaptive_sizing

        # Semantic similarity model
        self.similarity_model = None
        if enable_semantic_similarity and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic similarity model loaded")
            except Exception as e:
                logger.warning(f"Failed to load similarity model: {e}")
                self.enable_semantic_similarity = False

        # Statistics and monitoring
        self.stats = CacheStats()
        self.embeddings_cache: Dict[str, np.ndarray] = {}

        # Request patterns for optimization
        self.request_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.popular_patterns: Dict[str, float] = {}

        logger.info(f"Intelligent cache initialized (semantic: {self.enable_semantic_similarity})")

    def _generate_cache_key(self, prompt: str, agent_type: str, model_id: str, **kwargs) -> str:
        """Generate deterministic cache key."""
        # Include key parameters that affect output
        key_params = {
            "prompt": prompt.strip(),
            "agent_type": agent_type,
            "model_id": model_id,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 0.9)
        }

        # Create hash of normalized parameters
        key_string = json.dumps(key_params, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get(self, prompt: str, agent_type: str, model_id: str, **kwargs) -> Optional[CacheEntry]:
        """Get cached response with semantic similarity fallback."""
        start_time = time.time()
        self.stats.total_requests += 1

        # Try exact match first
        cache_key = self._generate_cache_key(prompt, agent_type, model_id, **kwargs)
        entry = await self.backend.get(cache_key)

        if entry:
            self.stats.cache_hits += 1
            self.stats.time_savings += entry.response_time
            self.stats.cost_savings += entry.cost_to_generate
            self._update_hit_rate()
            retrieval_time = time.time() - start_time
            self._update_avg_retrieval_time(retrieval_time)

            # Track request pattern
            self._track_request_pattern(cache_key)

            logger.debug(f"Cache hit: {cache_key[:8]}...")
            return entry

        # Try semantic similarity if enabled
        if self.enable_semantic_similarity and self.similarity_model:
            semantic_entry = await self._find_semantic_match(prompt, agent_type, model_id, **kwargs)
            if semantic_entry:
                self.stats.semantic_hits += 1
                self.stats.time_savings += semantic_entry.response_time
                self.stats.cost_savings += semantic_entry.cost_to_generate
                self._update_semantic_hit_rate()
                retrieval_time = time.time() - start_time
                self._update_avg_retrieval_time(retrieval_time)

                logger.debug(f"Semantic cache hit: {semantic_entry.key[:8]}...")
                return semantic_entry

        # Cache miss
        self.stats.cache_misses += 1
        self._update_hit_rate()
        logger.debug(f"Cache miss: {cache_key[:8]}...")
        return None

    async def set(self, prompt: str, agent_type: str, model_id: str, content: str,
                  metadata: Dict[str, Any], cost: float, quality_score: float,
                  tokens_used: int, response_time: float, **kwargs):
        """Cache response with intelligent priority scoring."""
        cache_key = self._generate_cache_key(prompt, agent_type, model_id, **kwargs)

        # Calculate embedding for semantic similarity
        embedding = None
        if self.enable_semantic_similarity and self.similarity_model:
            try:
                embedding = self.similarity_model.encode(prompt)
                self.embeddings_cache[cache_key] = embedding
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        # Calculate priority score for cache eviction
        priority_score = self._calculate_priority_score(
            cost, quality_score, len(prompt), response_time, agent_type
        )

        entry = CacheEntry(
            key=cache_key,
            content=content,
            metadata=metadata,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            cost_to_generate=cost,
            quality_score=quality_score,
            model_id=model_id,
            agent_type=agent_type,
            tokens_used=tokens_used,
            response_time=response_time,
            embedding=embedding,
            priority_score=priority_score
        )

        # Check cache size and evict if necessary
        await self._ensure_cache_size()

        # Set with TTL
        ttl_seconds = self.ttl_hours * 3600
        await self.backend.set(cache_key, entry, ttl_seconds)

        # Track request pattern
        self._track_request_pattern(cache_key)

        logger.debug(f"Cached response: {cache_key[:8]} (priority: {priority_score:.3f})")

    async def _find_semantic_match(self, prompt: str, agent_type: str, model_id: str,
                                 **kwargs) -> Optional[CacheEntry]:
        """Find semantically similar cached response."""
        if not self.similarity_model:
            return None

        try:
            # Generate embedding for input prompt
            query_embedding = self.similarity_model.encode(prompt)

            # Check all cached embeddings
            best_similarity = 0.0
            best_entry = None

            cache_keys = await self.backend.keys()
            for cache_key in cache_keys:
                if cache_key in self.embeddings_cache:
                    cached_embedding = self.embeddings_cache[cache_key]

                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, cached_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                    )

                    if similarity > best_similarity and similarity >= self.semantic_threshold:
                        cached_entry = await self.backend.get(cache_key)
                        if (cached_entry and
                            cached_entry.agent_type == agent_type and
                            cached_entry.model_id == model_id):
                            best_similarity = similarity
                            best_entry = cached_entry

            return best_entry

        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")
            return None

    def _calculate_priority_score(self, cost: float, quality_score: float,
                                prompt_length: int, response_time: float,
                                agent_type: str) -> float:
        """Calculate priority score for cache eviction."""
        # Higher score = higher priority = keep longer
        score = 0.0

        # Cost factor (expensive to generate = higher priority)
        score += min(cost * 100, 50)  # Cap at 50 points

        # Quality factor
        score += quality_score * 30  # 0-30 points

        # Prompt complexity factor (longer prompts often more valuable)
        complexity_score = min(prompt_length / 100, 20)  # Cap at 20 points
        score += complexity_score

        # Agent type importance
        agent_weights = {
            "synthesis": 1.2,
            "analysis": 1.1,
            "research": 1.0,
            "critic": 1.0,
            "general": 0.9
        }
        score *= agent_weights.get(agent_type, 1.0)

        # Response time factor (slower = more valuable to cache)
        if response_time > 5.0:
            score += 15  # High priority for slow responses
        elif response_time > 2.0:
            score += 10
        elif response_time > 1.0:
            score += 5

        return score

    async def _ensure_cache_size(self):
        """Ensure cache doesn't exceed size limits."""
        current_size_bytes = await self.backend.size()
        max_size_bytes = self.max_cache_size_mb * 1024 * 1024

        if current_size_bytes <= max_size_bytes:
            return

        # Get all entries for eviction scoring
        cache_keys = await self.backend.keys()
        entries_with_scores = []

        for key in cache_keys:
            entry = await self.backend.get(key)
            if entry:
                # Calculate eviction score (lower = evict first)
                eviction_score = self._calculate_eviction_score(entry)
                entries_with_scores.append((eviction_score, key, entry))

        # Sort by eviction score (lowest first)
        entries_with_scores.sort(key=lambda x: x[0])

        # Evict until under size limit
        evicted_count = 0
        target_size = max_size_bytes * 0.8  # Evict to 80% capacity

        for eviction_score, key, entry in entries_with_scores:
            if current_size_bytes <= target_size:
                break

            await self.backend.delete(key)
            self.embeddings_cache.pop(key, None)

            current_size_bytes -= len(pickle.dumps(entry))
            evicted_count += 1

        if evicted_count > 0:
            self.stats.evictions += evicted_count
            logger.info(f"Evicted {evicted_count} cache entries to manage size")

    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score (lower = evict first)."""
        score = entry.priority_score

        # Recent access bonus
        hours_since_access = (datetime.now() - entry.last_accessed).total_seconds() / 3600
        if hours_since_access < 1:
            score += 20
        elif hours_since_access < 6:
            score += 10
        elif hours_since_access < 24:
            score += 5

        # Access frequency bonus
        score += min(entry.access_count * 2, 20)

        # Age penalty (older entries more likely to be evicted)
        hours_since_creation = (datetime.now() - entry.created_at).total_seconds() / 3600
        if hours_since_creation > 48:
            score -= 10
        elif hours_since_creation > 24:
            score -= 5

        return score

    def _track_request_pattern(self, cache_key: str):
        """Track request patterns for predictive optimization."""
        now = datetime.now()
        self.request_patterns[cache_key].append(now)

        # Keep only last 100 requests per key
        if len(self.request_patterns[cache_key]) > 100:
            self.request_patterns[cache_key] = self.request_patterns[cache_key][-100:]

        # Update popularity score
        recent_requests = [
            req for req in self.request_patterns[cache_key]
            if (now - req).total_seconds() < 3600  # Last hour
        ]
        self.popular_patterns[cache_key] = len(recent_requests)

    def _update_hit_rate(self):
        """Update cache hit rate."""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests

    def _update_semantic_hit_rate(self):
        """Update semantic hit rate."""
        if self.stats.total_requests > 0:
            self.stats.semantic_hit_rate = self.stats.semantic_hits / self.stats.total_requests

    def _update_avg_retrieval_time(self, retrieval_time: float):
        """Update average retrieval time."""
        total_retrievals = self.stats.cache_hits + self.stats.semantic_hits
        if total_retrievals > 0:
            self.stats.avg_retrieval_time = (
                (self.stats.avg_retrieval_time * (total_retrievals - 1) + retrieval_time) /
                total_retrievals
            )

    async def get_popular_entries(self, limit: int = 10) -> List[Tuple[str, CacheEntry, float]]:
        """Get most popular cache entries."""
        popular_items = []

        for cache_key, popularity in sorted(
            self.popular_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]:
            entry = await self.backend.get(cache_key)
            if entry:
                popular_items.append((cache_key, entry, popularity))

        return popular_items

    async def preload_popular_models(self, model_loader_callback):
        """Preload popular models based on usage patterns."""
        if not callable(model_loader_callback):
            return

        # Analyze model usage patterns
        model_usage = defaultdict(float)
        cache_keys = await self.backend.keys()

        for cache_key in cache_keys:
            entry = await self.backend.get(cache_key)
            if entry:
                popularity = self.popular_patterns.get(cache_key, 0)
                model_usage[entry.model_id] += popularity

        # Preload top 3 models
        top_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)[:3]

        for model_id, usage_score in top_models:
            if usage_score > 5:  # Threshold for preloading
                try:
                    await model_loader_callback(model_id)
                    logger.info(f"Preloaded popular model: {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to preload model {model_id}: {e}")

    async def optimize_cache(self):
        """Perform cache optimization."""
        if self.adaptive_sizing:
            await self._adaptive_size_adjustment()

        # Clean up old request patterns
        cutoff = datetime.now() - timedelta(days=7)
        for cache_key in list(self.request_patterns.keys()):
            self.request_patterns[cache_key] = [
                req for req in self.request_patterns[cache_key]
                if req > cutoff
            ]
            if not self.request_patterns[cache_key]:
                del self.request_patterns[cache_key]
                self.popular_patterns.pop(cache_key, None)

        logger.info("Cache optimization completed")

    async def _adaptive_size_adjustment(self):
        """Adaptively adjust cache size based on hit rates."""
        if self.stats.total_requests < 100:
            return  # Need more data

        # Increase size if hit rate is high and we're evicting frequently
        if (self.stats.hit_rate > 0.7 and
            self.stats.evictions > self.stats.total_requests * 0.1):
            new_size = min(self.max_cache_size_mb * 1.2, 2048)  # Max 2GB
            logger.info(f"Increasing cache size to {new_size}MB (high hit rate)")
            self.max_cache_size_mb = new_size

        # Decrease size if hit rate is low
        elif self.stats.hit_rate < 0.3 and self.max_cache_size_mb > 128:
            new_size = max(self.max_cache_size_mb * 0.8, 128)  # Min 128MB
            logger.info(f"Decreasing cache size to {new_size}MB (low hit rate)")
            self.max_cache_size_mb = new_size

    async def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        self.stats.storage_used = await self.backend.size()
        return self.stats

    async def clear(self):
        """Clear all cache data."""
        await self.backend.clear()
        self.embeddings_cache.clear()
        self.request_patterns.clear()
        self.popular_patterns.clear()
        self.stats = CacheStats()
        logger.info("Cache cleared")


class ResourceOptimizer:
    """
    Resource optimization for GPU memory and model loading.

    Manages GPU memory efficiently for ZeroGPU deployments with
    intelligent model loading and memory cleanup strategies.
    """

    def __init__(self,
                 max_gpu_memory_mb: int = 8192,
                 memory_threshold: float = 0.9,
                 cleanup_interval: int = 300,
                 enable_model_quantization: bool = True):
        """
        Initialize resource optimizer.

        Args:
            max_gpu_memory_mb: Maximum GPU memory in MB
            memory_threshold: Memory usage threshold for cleanup
            cleanup_interval: Cleanup interval in seconds
            enable_model_quantization: Enable model quantization for memory savings
        """
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        self.enable_model_quantization = enable_model_quantization

        # Memory tracking
        self.memory_usage: Dict[str, float] = {}
        self.model_access_times: Dict[str, datetime] = {}
        self.memory_pressure_events = 0

        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None

        logger.info("Resource optimizer initialized")

    async def start(self):
        """Start resource optimization background tasks."""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Resource optimizer started")

    async def stop(self):
        """Stop resource optimization background tasks."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
            logger.info("Resource optimizer stopped")

    async def _periodic_cleanup(self):
        """Periodic memory cleanup task."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_unused_resources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")

    async def cleanup_unused_resources(self):
        """Clean up unused GPU resources."""
        try:
            import torch
            if not torch.cuda.is_available():
                return

            current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
            if current_memory > self.max_gpu_memory_mb * self.memory_threshold:
                self.memory_pressure_events += 1

                # Force garbage collection
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                freed_memory = current_memory - (torch.cuda.memory_allocated() / (1024**2))
                logger.info(f"Freed {freed_memory:.1f}MB GPU memory")

        except ImportError:
            pass  # Torch not available
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")

    def track_model_usage(self, model_id: str, memory_mb: float):
        """Track model memory usage."""
        self.memory_usage[model_id] = memory_mb
        self.model_access_times[model_id] = datetime.now()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**2)
                cached = torch.cuda.memory_reserved() / (1024**2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**2)

                return {
                    "gpu_memory_allocated_mb": allocated,
                    "gpu_memory_cached_mb": cached,
                    "gpu_memory_total_mb": total,
                    "gpu_memory_utilization": allocated / total,
                    "loaded_models": dict(self.memory_usage),
                    "memory_pressure_events": self.memory_pressure_events
                }
        except ImportError:
            pass

        return {
            "gpu_memory_available": False,
            "loaded_models": dict(self.memory_usage),
            "memory_pressure_events": self.memory_pressure_events
        }


# Factory functions for easy integration
def create_intelligent_cache(use_redis: bool = False,
                           redis_url: str = "redis://localhost:6379") -> IntelligentCache:
    """
    Create intelligent cache with recommended settings.

    Args:
        use_redis: Use Redis backend for distributed caching
        redis_url: Redis connection URL

    Returns:
        Configured IntelligentCache instance
    """
    backend = None
    if use_redis and REDIS_AVAILABLE:
        try:
            backend = RedisCacheBackend(redis_url)
        except Exception as e:
            logger.warning(f"Redis backend failed, using memory: {e}")

    if not backend:
        backend = MemoryCacheBackend(max_size=1000)

    return IntelligentCache(
        backend=backend,
        enable_semantic_similarity=SENTENCE_TRANSFORMERS_AVAILABLE,
        semantic_threshold=0.85,
        max_cache_size_mb=512,
        ttl_hours=24,
        cost_optimization=True,
        adaptive_sizing=True
    )


def create_resource_optimizer() -> ResourceOptimizer:
    """
    Create resource optimizer with recommended settings.

    Returns:
        Configured ResourceOptimizer instance
    """
    return ResourceOptimizer(
        max_gpu_memory_mb=8192,  # 8GB default
        memory_threshold=0.9,
        cleanup_interval=300,  # 5 minutes
        enable_model_quantization=True
    )


# Export main classes
__all__ = [
    'IntelligentCache',
    'ResourceOptimizer',
    'CacheEntry',
    'CacheStats',
    'MemoryCacheBackend',
    'RedisCacheBackend',
    'create_intelligent_cache',
    'create_resource_optimizer'
]