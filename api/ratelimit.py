"""In-memory sliding-window rate limiter.

State lives in this process only: with several workers or replicas each one
counts separately, so the effective limit multiplies. The app is deployed with
one worker and max one replica (SQLite state), which keeps this honest; a
multi-replica deployment would move the counters to Redis or the gateway.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, bucket: str, key: str, limit: int, window_seconds: float) -> float | None:
        """Record one request. Returns None if allowed, else seconds until a slot frees."""

        now = self._clock()
        with self._lock:
            if len(self._hits) > 10_000:
                # Drop idle keys so per-IP buckets cannot grow without bound.
                for stale in [k for k, v in self._hits.items() if not v or v[-1] <= now - window_seconds]:
                    del self._hits[stale]
            hits = self._hits[(bucket, key)]
            while hits and hits[0] <= now - window_seconds:
                hits.popleft()
            if len(hits) >= limit:
                return max(0.0, hits[0] + window_seconds - now)
            hits.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
