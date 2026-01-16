"""
Rate limiter for API calls.
Implements token bucket algorithm to respect API rate limits.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from ..observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    calls_per_minute: int = 15
    burst_limit: int = 5  # Max burst above rate


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Kraken rate limits:
    - Public endpoints: ~1 call/second
    - Private endpoints: ~15-20 calls/minute (varies by tier)
    """

    def __init__(
        self,
        calls_per_minute: int = 15,
        burst_limit: int = 5
    ):
        """
        Initialize the rate limiter.

        Args:
            calls_per_minute: Maximum sustained calls per minute.
            burst_limit: Maximum burst calls allowed.
        """
        self.calls_per_minute = calls_per_minute
        self.burst_limit = burst_limit

        # Token bucket parameters
        self.tokens = float(burst_limit)
        self.max_tokens = float(burst_limit)
        self.refill_rate = calls_per_minute / 60.0  # Tokens per second
        self.last_refill = time.monotonic()

        # Statistics
        self.total_calls = 0
        self.total_waits = 0
        self.total_wait_time = 0.0

        self._lock = asyncio.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens for an API call, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        async with self._lock:
            self._refill_tokens()

            wait_time = 0.0

            while self.tokens < tokens:
                # Calculate wait time
                deficit = tokens - self.tokens
                sleep_time = deficit / self.refill_rate

                logger.debug(
                    f"Rate limit: waiting {sleep_time:.2f}s for {deficit:.2f} tokens",
                    tokens_needed=tokens,
                    tokens_available=self.tokens,
                    sleep_time=sleep_time
                )

                await asyncio.sleep(sleep_time)
                wait_time += sleep_time
                self._refill_tokens()

            self.tokens -= tokens
            self.total_calls += 1

            if wait_time > 0:
                self.total_waits += 1
                self.total_wait_time += wait_time

            return wait_time

    def acquire_sync(self, tokens: int = 1) -> float:
        """
        Synchronous version of acquire for non-async code.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        self._refill_tokens()

        wait_time = 0.0

        while self.tokens < tokens:
            deficit = tokens - self.tokens
            sleep_time = deficit / self.refill_rate

            logger.debug(
                f"Rate limit: waiting {sleep_time:.2f}s for {deficit:.2f} tokens",
                tokens_needed=tokens,
                tokens_available=self.tokens,
                sleep_time=sleep_time
            )

            time.sleep(sleep_time)
            wait_time += sleep_time
            self._refill_tokens()

        self.tokens -= tokens
        self.total_calls += 1

        if wait_time > 0:
            self.total_waits += 1
            self.total_wait_time += wait_time

        return wait_time

    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "total_calls": self.total_calls,
            "total_waits": self.total_waits,
            "total_wait_time": self.total_wait_time,
            "average_wait_time": (
                self.total_wait_time / self.total_waits
                if self.total_waits > 0
                else 0.0
            ),
            "current_tokens": self.tokens,
            "max_tokens": self.max_tokens,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_calls = 0
        self.total_waits = 0
        self.total_wait_time = 0.0


class MultiEndpointRateLimiter:
    """
    Rate limiter that tracks separate limits for different endpoint types.
    """

    def __init__(
        self,
        public_calls_per_minute: int = 60,
        private_calls_per_minute: int = 15
    ):
        """
        Initialize multi-endpoint rate limiter.

        Args:
            public_calls_per_minute: Rate limit for public endpoints.
            private_calls_per_minute: Rate limit for private endpoints.
        """
        self.public_limiter = RateLimiter(
            calls_per_minute=public_calls_per_minute,
            burst_limit=10
        )
        self.private_limiter = RateLimiter(
            calls_per_minute=private_calls_per_minute,
            burst_limit=3
        )

    async def acquire(self, endpoint_type: str = "private", tokens: int = 1) -> float:
        """
        Acquire tokens for the appropriate endpoint type.

        Args:
            endpoint_type: "public" or "private".
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        if endpoint_type == "public":
            return await self.public_limiter.acquire(tokens)
        else:
            return await self.private_limiter.acquire(tokens)

    def acquire_sync(self, endpoint_type: str = "private", tokens: int = 1) -> float:
        """
        Synchronous version of acquire.

        Args:
            endpoint_type: "public" or "private".
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        if endpoint_type == "public":
            return self.public_limiter.acquire_sync(tokens)
        else:
            return self.private_limiter.acquire_sync(tokens)

    def get_stats(self) -> dict:
        """
        Get combined stats for all limiters.

        Returns:
            Dictionary with stats for each endpoint type.
        """
        return {
            "public": self.public_limiter.get_stats(),
            "private": self.private_limiter.get_stats(),
        }
