# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Dict, Optional
import asyncio
import hashlib
import threading


class _DualSemaphore:
    """A semaphore that works in both threading and asyncio contexts.

    - Sync callers (the main thread running ``LMProvider.run_async``) acquire
      via the threading semaphore.
    - Async callers (coroutines running on the persistent event loop) acquire
      via the asyncio semaphore.

    Both semaphores are initialised with the same ``value`` so the effective
    concurrency limit is ``value`` regardless of which interface is used.

    In practice, DGT's current sync runner uses only the threading path.
    The async path is available for future ``acall``-based consumers (Phase 2).
    """

    def __init__(self, value: int) -> None:
        self._value = value
        self._threading_sem = threading.Semaphore(value)
        # asyncio.Semaphore must be created on the event loop that will use it;
        # we defer creation until first async acquisition.
        self._async_sem: Optional[asyncio.Semaphore] = None
        self._async_sem_lock = threading.Lock()

    @property
    def value(self) -> int:
        return self._value

    def _get_async_sem(self) -> asyncio.Semaphore:
        """Return (lazily creating) the asyncio semaphore."""
        if self._async_sem is None:
            with self._async_sem_lock:
                if self._async_sem is None:
                    self._async_sem = asyncio.Semaphore(self._value)
        return self._async_sem

    # ------------------------------------------------------------------
    # Sync interface (threading)
    # ------------------------------------------------------------------

    def acquire(self) -> bool:
        return self._threading_sem.acquire()

    def release(self) -> None:
        self._threading_sem.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Async interface (asyncio)
    # ------------------------------------------------------------------

    async def async_acquire(self) -> None:
        await self._get_async_sem().acquire()

    async def async_release(self) -> None:
        self._get_async_sem().release()

    async def __aenter__(self):
        await self.async_acquire()
        return self

    async def __aexit__(self, *_):
        await self.async_release()


class CredentialPool:
    """Maintains one ``_DualSemaphore`` per unique credential.

    The pool key is a SHA-256 hash of the credential string (typically an API
    key), so the raw credential is never stored.  All LLM blocks that share
    the same credential share a single semaphore, enforcing a global
    ``max_concurrent_requests`` limit across all blocks for that credential.

    Usage (sync)::

        pool = CredentialPool.get_instance()
        sem = pool.get(api_key, max_concurrent_requests=20)
        with sem:
            # at most 20 concurrent threads reach here for this credential
            ...

    Usage (async)::

        pool = CredentialPool.get_instance()
        sem = pool.get(api_key, max_concurrent_requests=20)
        async with sem:
            ...

    The ``max_concurrent_requests`` argument is only respected on first
    creation of the semaphore for a given credential.  Subsequent calls with
    the same credential return the existing semaphore regardless of the
    ``max_concurrent_requests`` value passed.
    """

    _instance: Optional["CredentialPool"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._pool: Dict[str, _DualSemaphore] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "CredentialPool":
        """Return the process-wide singleton CredentialPool."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton (useful in tests)."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Pool access
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(credential: str) -> str:
        return hashlib.sha256(credential.encode()).hexdigest()

    def get(self, credential: str, max_concurrent_requests: int = 10) -> _DualSemaphore:
        """Return the semaphore for ``credential``, creating it if necessary.

        Args:
            credential: The API key or other credential string.  Only its hash
                is stored; the raw value is not retained.
            max_concurrent_requests: Concurrency limit.  Only applied when the
                semaphore is first created for this credential.

        Returns:
            A ``_DualSemaphore`` shared by all callers with the same
            credential.
        """
        key = self._hash(credential)
        if key not in self._pool:
            with self._lock:
                if key not in self._pool:
                    self._pool[key] = _DualSemaphore(max_concurrent_requests)
        return self._pool[key]
