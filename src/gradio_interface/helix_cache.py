"""
Optimized helix caching system for web deployment.

This module provides caching for helix geometry calculations to reduce
computational overhead in web deployments where the same helix configurations
are used repeatedly.

Key Features:
- Pre-computed position caches
- Memory-efficient storage
- Thread-safe access
- Automatic cache warming
"""

import numpy as np
import threading
import hashlib
import pickle
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging

from core.helix_geometry import HelixGeometry

logger = logging.getLogger(__name__)


@dataclass
class HelixCacheEntry:
    """Cache entry for helix geometry data."""
    helix_params: Tuple[float, float, float, int]
    positions: np.ndarray  # Pre-computed positions
    radii: np.ndarray  # Pre-computed radii
    angles: np.ndarray  # Pre-computed angles
    access_count: int = 0
    last_accessed: float = 0.0


class HelixCache:
    """
    Caching system for helix geometry calculations.

    Provides fast access to pre-computed helix positions and properties
    to reduce computational overhead in web deployments.
    """

    def __init__(self, max_cache_size: int = 50, precompute_steps: int = 100):
        """
        Initialize helix cache.

        Args:
            max_cache_size: Maximum number of helix configurations to cache
            precompute_steps: Number of steps to precompute for each helix
        """
        self.max_cache_size = max_cache_size
        self.precompute_steps = precompute_steps
        self._lock = threading.Lock()
        self._cache: Dict[str, HelixCacheEntry] = {}

        # Pre-warm cache with common configurations
        self._warm_cache()

    def _warm_cache(self):
        """Pre-populate cache with common helix configurations."""
        common_configs = [
            # ComplexityLevel.DEMO
            (10.0, 0.01, 5.0, 5),
            # ComplexityLevel.SIMPLE
            (10.0, 0.01, 10.0, 10),
            # ComplexityLevel.MEDIUM
            (10.0, 0.01, 20.0, 20),
            # ComplexityLevel.COMPLEX
            (10.0, 0.01, 30.0, 30),
            # ComplexityLevel.RESEARCH (original)
            (33.0, 0.001, 33.0, 33),
        ]

        for config in common_configs:
            self._compute_and_cache(*config)

        logger.info(f"Cache warmed with {len(common_configs)} common configurations")

    def _make_cache_key(self, top_radius: float, bottom_radius: float,
                       height: float, turns: int) -> str:
        """Generate cache key from helix parameters."""
        params = f"{top_radius}_{bottom_radius}_{height}_{turns}"
        return hashlib.md5(params.encode()).hexdigest()

    def get_helix_data(self, top_radius: float, bottom_radius: float,
                      height: float, turns: int) -> Optional[HelixCacheEntry]:
        """
        Get cached helix data if available.

        Args:
            top_radius: Radius at the top of the helix
            bottom_radius: Radius at the bottom of the helix
            height: Total vertical height
            turns: Number of complete rotations

        Returns:
            Cached helix data or None if not cached
        """
        cache_key = self._make_cache_key(top_radius, bottom_radius, height, turns)

        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.access_count += 1
                entry.last_accessed = time.time()

                # Move to end for LRU
                self._cache.move_to_end(cache_key)

                return entry

        # Not in cache, compute and store
        return self._compute_and_cache(top_radius, bottom_radius, height, turns)

    def _compute_and_cache(self, top_radius: float, bottom_radius: float,
                          height: float, turns: int) -> HelixCacheEntry:
        """Compute helix data and add to cache."""
        # Create helix geometry
        helix = HelixGeometry(top_radius, bottom_radius, height, turns)

        # Pre-compute positions at regular intervals
        t_values = np.linspace(0, 1, self.precompute_steps)
        positions = np.zeros((self.precompute_steps, 3))
        radii = np.zeros(self.precompute_steps)
        angles = np.zeros(self.precompute_steps)

        for i, t in enumerate(t_values):
            x, y, z = helix.get_position(t)
            positions[i] = [x, y, z]
            radii[i] = helix.get_radius(z)
            angles[i] = 2 * np.pi * turns * t

        # Create cache entry
        entry = HelixCacheEntry(
            helix_params=(top_radius, bottom_radius, height, turns),
            positions=positions,
            radii=radii,
            angles=angles
        )

        # Add to cache
        cache_key = self._make_cache_key(top_radius, bottom_radius, height, turns)

        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_cache_size:
                # Remove least recently used
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[cache_key] = entry

        logger.debug(f"Cached helix configuration: {turns} turns, {height} height")
        return entry

    def interpolate_position(self, entry: HelixCacheEntry, t: float) -> Tuple[float, float, float]:
        """
        Interpolate position from cached data.

        Args:
            entry: Cached helix data
            t: Parameter value (0-1)

        Returns:
            Interpolated (x, y, z) position
        """
        # Find surrounding cached points
        scaled_t = t * (self.precompute_steps - 1)
        idx_low = int(np.floor(scaled_t))
        idx_high = min(idx_low + 1, self.precompute_steps - 1)

        # Interpolation weight
        weight = scaled_t - idx_low

        # Linear interpolation
        pos_low = entry.positions[idx_low]
        pos_high = entry.positions[idx_high]

        interpolated = pos_low * (1 - weight) + pos_high * weight

        return tuple(interpolated)

    def get_cached_helix(self, top_radius: float, bottom_radius: float,
                        height: float, turns: int) -> 'CachedHelixGeometry':
        """
        Get a cached helix geometry object.

        Returns a helix-like object that uses cached data for fast lookups.
        """
        entry = self.get_helix_data(top_radius, bottom_radius, height, turns)
        return CachedHelixGeometry(self, entry)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(e.access_count for e in self._cache.values())
            avg_accesses = total_accesses / len(self._cache) if self._cache else 0

            return {
                "cache_size": len(self._cache),
                "max_size": self.max_cache_size,
                "total_accesses": total_accesses,
                "average_accesses": avg_accesses,
                "precompute_steps": self.precompute_steps
            }

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
        logger.info("Helix cache cleared")


class CachedHelixGeometry:
    """
    Helix geometry wrapper that uses cached data.

    Provides the same interface as HelixGeometry but uses
    pre-computed cached data for fast lookups.
    """

    def __init__(self, cache: HelixCache, entry: HelixCacheEntry):
        """
        Initialize cached helix geometry.

        Args:
            cache: Parent cache instance
            entry: Cache entry with pre-computed data
        """
        self.cache = cache
        self.entry = entry

        # Extract parameters
        self.top_radius = entry.helix_params[0]
        self.bottom_radius = entry.helix_params[1]
        self.height = entry.helix_params[2]
        self.turns = entry.helix_params[3]

    def get_position(self, t: float) -> Tuple[float, float, float]:
        """
        Get position at parameter t using cached data.

        Args:
            t: Parameter value (0-1)

        Returns:
            (x, y, z) position
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0 and 1")

        return self.cache.interpolate_position(self.entry, t)

    def get_radius(self, z: float) -> float:
        """
        Get radius at height z using cached data.

        Args:
            z: Height value

        Returns:
            Radius at height z
        """
        # Convert z to t parameter
        t = 1.0 - (z / self.height) if self.height > 0 else 0.0
        t = max(0.0, min(1.0, t))

        # Interpolate from cached radii
        scaled_t = t * (len(self.entry.radii) - 1)
        idx_low = int(np.floor(scaled_t))
        idx_high = min(idx_low + 1, len(self.entry.radii) - 1)

        weight = scaled_t - idx_low
        return self.entry.radii[idx_low] * (1 - weight) + self.entry.radii[idx_high] * weight

    def get_velocity(self, t: float) -> Tuple[float, float, float]:
        """
        Get velocity vector at parameter t.

        Uses finite differences on cached data.
        """
        dt = 0.01
        t1 = max(0, t - dt / 2)
        t2 = min(1, t + dt / 2)

        pos1 = self.get_position(t1)
        pos2 = self.get_position(t2)

        actual_dt = t2 - t1
        if actual_dt > 0:
            velocity = tuple((p2 - p1) / actual_dt for p1, p2 in zip(pos1, pos2))
        else:
            velocity = (0.0, 0.0, 0.0)

        return velocity

    def calculate_arc_length(self, t1: float = 0.0, t2: float = 1.0,
                            steps: int = 100) -> float:
        """Calculate arc length between two parameter values."""
        # Use cached positions for fast calculation
        t_values = np.linspace(t1, t2, steps)
        total_length = 0.0

        for i in range(1, len(t_values)):
            pos1 = self.get_position(t_values[i - 1])
            pos2 = self.get_position(t_values[i])
            segment_length = np.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(pos1, pos2)))
            total_length += segment_length

        return total_length


# Global cache instance
_global_helix_cache: Optional[HelixCache] = None


def get_helix_cache() -> HelixCache:
    """Get or create global helix cache instance."""
    global _global_helix_cache
    if _global_helix_cache is None:
        _global_helix_cache = HelixCache()
    return _global_helix_cache


# Import time at the end to avoid circular import
import time