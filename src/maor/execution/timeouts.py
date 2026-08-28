"""Bounded execution: nothing runs without a deadline.

``stage_timeout_s`` and ``model_load_timeout_s`` existed in the configuration and
were described in the runbook as "nothing runs unbounded", but no code read them.
This module makes that true.

A note on what can and cannot be interrupted. A CUDA kernel already dispatched to
the device cannot be pre-empted from Python; neither can a C extension holding
the GIL. So there are two mechanisms here:

* :func:`run_with_timeout` runs the callable on a worker thread and stops
  *waiting* at the deadline. The work may continue in the background, so this is
  used where the caller needs control back and the process is going to exit or
  the resource is going to be reset regardless.
* :class:`TimeoutGuard` watches a block and raises in the main thread at the
  deadline via a watchdog. It cannot interrupt a blocking C call, but it does
  fire the moment control returns to the interpreter, which covers generation
  loops, dataloaders and retry loops — the places that actually hang.

Neither can guarantee interruption of a wedged driver call. What they guarantee
is that the *process* does not wait forever without reporting why, which is the
difference between a run that fails at 3am and one that is still hung at 9am.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutError_(TimeoutError):
    """Raised when a bounded operation exceeds its deadline.

    Named with a trailing underscore to avoid shadowing the builtin while
    remaining a subclass of it, so ``except TimeoutError`` still catches it.
    """

    def __init__(
        self, label: str = "operation", seconds: float = 0.0, advice: str = ""
    ) -> None:
        # Both arguments default because the watchdog raises this class into the
        # main thread via PyThreadState_SetAsyncExc, which instantiates it with
        # no arguments. Requiring them turned every watchdog firing into a
        # TypeError, and the timeout was then misreported as a generic failure.
        message = f"{label!r} exceeded its {seconds:.3g}s deadline"
        if advice:
            message += f". {advice}"
        super().__init__(message)
        self.label = label
        self.seconds = seconds


@dataclass
class _Result:
    value: Any = None
    error: BaseException | None = None
    done: bool = False


def run_with_timeout(
    fn: Callable[[], T],
    seconds: float,
    *,
    label: str = "operation",
    advice: str = "",
) -> T:
    """Run ``fn`` and stop waiting after ``seconds``.

    The worker thread is a daemon: if the call is genuinely wedged the process
    can still exit. The work is not killed, so callers must treat the associated
    resource as unusable afterwards — which for a wedged CUDA context means the
    process, not just the model.
    """
    if seconds <= 0:
        return fn()

    result = _Result()

    def _target() -> None:
        try:
            result.value = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            result.error = exc
        finally:
            result.done = True

    thread = threading.Thread(target=_target, name=f"timeout:{label}", daemon=True)
    started = time.perf_counter()
    thread.start()
    thread.join(timeout=seconds)

    if not result.done:
        raise TimeoutError_(
            label,
            seconds,
            advice
            or (
                "The work is still running on a background thread and cannot be "
                "killed from Python. Treat the GPU context as unusable and restart "
                "the process."
            ),
        )
    if result.error is not None:
        raise result.error

    log.debug("%s completed in %.1fs", label, time.perf_counter() - started)
    return result.value  # type: ignore[return-value]


class TimeoutGuard:
    """Watchdog that reports, and optionally interrupts, an overrunning block.

    Used where the work must run on the calling thread — CUDA work is bound to
    the thread that owns the context — but must not be allowed to hang silently.
    """

    def __init__(
        self,
        label: str,
        seconds: float,
        *,
        on_timeout: Callable[[], None] | None = None,
        interrupt: bool = True,
    ) -> None:
        self.label = label
        self.seconds = seconds
        self.on_timeout = on_timeout
        self.interrupt = interrupt
        self.timed_out = False
        self.elapsed_s = 0.0
        self._timer: threading.Timer | None = None
        self._started = 0.0

    def _fire(self) -> None:
        self.timed_out = True
        log.error(
            "%s exceeded its %.3gs deadline and is still running",
            self.label,
            self.seconds,
        )
        if self.on_timeout is not None:
            try:
                self.on_timeout()
            except Exception as exc:
                log.warning("timeout handler for %s raised: %s", self.label, exc)
        if self.interrupt:
            # Raises in the main thread at the next bytecode boundary. Cannot
            # interrupt a blocking C call, but fires as soon as control returns.
            import ctypes

            try:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(threading.main_thread().ident or 0),
                    ctypes.py_object(TimeoutError_),
                )
            except Exception as exc:  # pragma: no cover - platform-dependent
                log.debug("could not signal main thread: %s", exc)

    def __enter__(self) -> "TimeoutGuard":
        if self.seconds > 0:
            self._started = time.perf_counter()
            self._timer = threading.Timer(self.seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._timer is not None:
            self._timer.cancel()
        self.elapsed_s = time.perf_counter() - self._started if self._started else 0.0
        if self.timed_out and exc_type is None:
            raise TimeoutError_(self.label, self.seconds)
        return False


@contextmanager
def bounded(label: str, seconds: float, **kwargs: Any) -> Iterator[TimeoutGuard]:
    """Convenience wrapper around :class:`TimeoutGuard`."""
    guard = TimeoutGuard(label, seconds, **kwargs)
    with guard:
        yield guard


@dataclass
class RetryPolicy:
    """Bounded retries. Unbounded retry on a resource error is an infinite loop.

    ``retry_on_oom`` defaults to False deliberately: retrying an out-of-memory
    failure with identical parameters will fail identically, and the loop is a
    common way for a run to appear hung. Set it only alongside a
    ``on_retry`` that changes something — a smaller batch, a released model.
    """

    max_attempts: int = 3
    initial_delay_s: float = 1.0
    backoff: float = 2.0
    max_delay_s: float = 30.0
    retry_on: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, OSError)
    retry_on_oom: bool = False
    on_retry: Callable[[int, BaseException], None] | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def _should_retry(self, exc: BaseException) -> bool:
        if _is_oom(exc):
            return self.retry_on_oom
        return isinstance(exc, self.retry_on)

    def run(self, fn: Callable[[], T], *, label: str = "operation") -> T:
        delay = self.initial_delay_s
        last: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except BaseException as exc:  # noqa: BLE001
                last = exc
                if not self._should_retry(exc) or attempt == self.max_attempts:
                    raise
                log.warning(
                    "%s failed (attempt %d/%d): %s: %s — retrying in %.1fs",
                    label,
                    attempt,
                    self.max_attempts,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                if self.on_retry is not None:
                    try:
                        self.on_retry(attempt, exc)
                    except Exception as hook_exc:
                        log.warning("retry hook raised: %s", hook_exc)
                time.sleep(delay)
                delay = min(delay * self.backoff, self.max_delay_s)
        assert last is not None
        raise last


def _is_oom(exc: BaseException) -> bool:
    """Recognise a CUDA out-of-memory failure across torch versions."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def is_oom(exc: BaseException) -> bool:
    """Public predicate for out-of-memory failures."""
    return _is_oom(exc)


def with_timeout(seconds: float, label: str | None = None):
    """Decorator form of :func:`run_with_timeout`."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return run_with_timeout(
                lambda: fn(*args, **kwargs), seconds, label=label or fn.__name__
            )

        return wrapper

    return decorator
