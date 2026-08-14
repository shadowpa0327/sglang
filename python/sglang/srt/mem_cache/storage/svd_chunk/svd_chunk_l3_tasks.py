# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Bounded asynchronous task manager for SVD chunk L3 storage.

The manager deliberately owns only scheduling and completion semantics.  Blob
validation, cache admission, and durability policy remain the connector's
responsibility.

Each submission returns a caller-owned :class:`concurrent.futures.Future`.
Submissions for the same operation and key share one backend task, but never
share that public future.  Consequently, cancelling one caller does not cancel
the backend operation or poison any other caller waiting on the same key.  This
is particularly important for puts: once accepted, a durability write runs to
completion even if every observer goes away.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Protocol, TypeVar

T = TypeVar("T")
Operation = Literal["read", "put"]


class SVDChunkL3Backend(Protocol[T]):
    """Blocking backend surface scheduled by :class:`SVDChunkL3TaskManager`."""

    def get(self, key: str) -> Optional[T]: ...

    def put(self, key: str, value: T) -> bool: ...


class SVDChunkL3TaskManagerError(RuntimeError):
    """Base class for task-manager lifecycle and scheduling errors."""


class SVDChunkL3QueueFull(SVDChunkL3TaskManagerError):
    """Raised through a submission future when the bounded queue is full."""


class SVDChunkL3TaskManagerClosed(SVDChunkL3TaskManagerError):
    """Raised through a submission future after the manager has been closed."""


@dataclass(frozen=True)
class SVDChunkL3TaskFailure:
    """One backend exception retained for drain/shutdown observability."""

    operation: Operation
    key: str
    exception: BaseException


class SVDChunkL3TaskFailures(SVDChunkL3TaskManagerError):
    """Aggregate raised by ``drain`` or ``shutdown`` on request."""

    def __init__(self, failures: tuple[SVDChunkL3TaskFailure, ...]) -> None:
        if not failures:
            raise ValueError("failures must not be empty")
        self.failures = failures
        summary = ", ".join(
            f"{failure.operation}({failure.key!r}): "
            f"{type(failure.exception).__name__}"
            for failure in failures
        )
        super().__init__(f"{len(failures)} SVD L3 task(s) failed: {summary}")


@dataclass(frozen=True)
class SVDChunkL3TaskManagerStats:
    """Atomic lifetime counters plus a live queue snapshot."""

    max_workers: int
    max_pending_tasks: int
    closed: bool
    pending_tasks: int
    pending_reads: int
    pending_puts: int
    max_observed_pending_tasks: int
    read_requests: int
    read_tasks_submitted: int
    read_requests_coalesced: int
    read_tasks_completed: int
    read_tasks_failed: int
    put_requests: int
    put_tasks_submitted: int
    put_requests_coalesced: int
    put_tasks_completed: int
    put_tasks_failed: int
    queue_full_rejections: int
    closed_rejections: int
    subscriber_cancellations: int


@dataclass
class _SharedTask:
    operation: Operation
    key: str
    subscribers: list[concurrent.futures.Future[Any]]
    backend_future: Optional[concurrent.futures.Future[Any]] = None


def _failed_future(exception: BaseException) -> concurrent.futures.Future[Any]:
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()
    future.set_exception(exception)
    return future


class SVDChunkL3TaskManager:
    """Run blocking L3 reads and puts on separate bounded thread pools.

    ``max_pending_tasks`` bounds the total number of distinct backend tasks,
    including both running and queued work.  A duplicate operation for an
    already-pending key coalesces even when this bound has been reached because
    it adds no backend work.  New work rejected by the bound is represented by
    an already-failed future; submission itself never blocks.

    Reads and puts each receive ``max_workers`` workers.  Keeping their
    executors separate prevents background durability writes from occupying
    every worker needed by latency-sensitive prefix reads, while the shared
    pending-task bound continues to cap their aggregate queued work.

    Put coalescing relies on the SVD store's content-addressed invariant: the
    first accepted value for a pending key is authoritative.  Callers must not
    submit different bytes under the same key.
    """

    def __init__(
        self,
        backend: SVDChunkL3Backend[Any],
        *,
        max_workers: int = 2,
        max_pending_tasks: int = 64,
        thread_name_prefix: str = "SVDChunkL3",
        read_transform: Optional[Callable[[str, Any], Any]] = None,
    ) -> None:
        if not callable(getattr(backend, "get", None)):
            raise TypeError("backend must provide get(key)")
        if not callable(getattr(backend, "put", None)):
            raise TypeError("backend must provide put(key, value)")
        self._validate_positive_int("max_workers", max_workers)
        self._validate_positive_int("max_pending_tasks", max_pending_tasks)
        if max_pending_tasks < max_workers:
            raise ValueError("max_pending_tasks must be at least max_workers")
        if not isinstance(thread_name_prefix, str) or not thread_name_prefix:
            raise ValueError("thread_name_prefix must be a nonempty string")
        if read_transform is not None and not callable(read_transform):
            raise TypeError("read_transform must be callable or None")

        self._backend = backend
        self._read_transform = read_transform
        self._max_workers = max_workers
        self._max_pending_tasks = max_pending_tasks
        self._read_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{thread_name_prefix}Read",
        )
        self._put_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{thread_name_prefix}Put",
        )
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._reads: dict[str, _SharedTask] = {}
        self._puts: dict[str, _SharedTask] = {}
        self._pending_tasks = 0
        self._max_observed_pending_tasks = 0
        self._failures: list[SVDChunkL3TaskFailure] = []

        self._read_requests = 0
        self._read_tasks_submitted = 0
        self._read_requests_coalesced = 0
        self._read_tasks_completed = 0
        self._read_tasks_failed = 0
        self._put_requests = 0
        self._put_tasks_submitted = 0
        self._put_requests_coalesced = 0
        self._put_tasks_completed = 0
        self._put_tasks_failed = 0
        self._queue_full_rejections = 0
        self._closed_rejections = 0
        self._subscriber_cancellations = 0

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a nonempty string")

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def submit_read(self, key: str) -> concurrent.futures.Future[Optional[Any]]:
        """Schedule ``backend.get(key)`` or join its pending keyed read."""

        self._validate_key(key)
        return self._submit("read", key, None)

    def submit_put(self, key: str, value: Any) -> concurrent.futures.Future[bool]:
        """Schedule ``backend.put(key, value)`` or join its pending keyed put."""

        self._validate_key(key)
        return self._submit("put", key, value)

    def _submit(
        self, operation: Operation, key: str, value: Any
    ) -> concurrent.futures.Future[Any]:
        subscriber: concurrent.futures.Future[Any] = concurrent.futures.Future()
        subscriber.add_done_callback(self._subscriber_done)
        tasks = self._reads if operation == "read" else self._puts

        with self._condition:
            if operation == "read":
                self._read_requests += 1
            else:
                self._put_requests += 1

            existing = tasks.get(key)
            if existing is not None:
                existing.subscribers.append(subscriber)
                if operation == "read":
                    self._read_requests_coalesced += 1
                else:
                    self._put_requests_coalesced += 1
                return subscriber

            if self._closed:
                self._closed_rejections += 1
                return _failed_future(
                    SVDChunkL3TaskManagerClosed("SVD L3 task manager is closed")
                )
            if self._pending_tasks >= self._max_pending_tasks:
                self._queue_full_rejections += 1
                return _failed_future(
                    SVDChunkL3QueueFull(
                        "SVD L3 task queue is full "
                        f"({self._pending_tasks}/{self._max_pending_tasks})"
                    )
                )

            shared = _SharedTask(operation, key, [subscriber])
            tasks[key] = shared
            self._pending_tasks += 1
            self._max_observed_pending_tasks = max(
                self._max_observed_pending_tasks, self._pending_tasks
            )
            if operation == "read":
                self._read_tasks_submitted += 1
                callable_ = self._run_read
                args = (key,)
                executor = self._read_executor
            else:
                self._put_tasks_submitted += 1
                callable_ = self._backend.put
                args = (key, value)
                executor = self._put_executor

            try:
                backend_future = executor.submit(callable_, *args)
            except BaseException as exc:
                # ThreadPoolExecutor.submit can fail during interpreter or
                # executor shutdown. Roll the task back atomically and surface
                # the exact exception to this caller.
                tasks.pop(key, None)
                self._pending_tasks -= 1
                self._condition.notify_all()
                subscriber.set_exception(exc)
                return subscriber
            shared.backend_future = backend_future
            backend_future.add_done_callback(
                lambda completed, task=shared: self._complete(task, completed)
            )
            return subscriber

    def _run_read(self, key: str) -> Any:
        value = self._backend.get(key)
        if self._read_transform is None:
            return value
        return self._read_transform(key, value)

    def _subscriber_done(self, future: concurrent.futures.Future[Any]) -> None:
        if future.cancelled():
            with self._condition:
                self._subscriber_cancellations += 1

    def _complete(
        self,
        shared: _SharedTask,
        backend_future: concurrent.futures.Future[Any],
    ) -> None:
        exception = backend_future.exception()
        result = None if exception is not None else backend_future.result()

        with self._condition:
            tasks = self._reads if shared.operation == "read" else self._puts
            if tasks.get(shared.key) is shared:
                tasks.pop(shared.key)
            self._pending_tasks -= 1
            if exception is None:
                if shared.operation == "read":
                    self._read_tasks_completed += 1
                else:
                    self._put_tasks_completed += 1
            else:
                self._failures.append(
                    SVDChunkL3TaskFailure(
                        operation=shared.operation,
                        key=shared.key,
                        exception=exception,
                    )
                )
                if shared.operation == "read":
                    self._read_tasks_failed += 1
                else:
                    self._put_tasks_failed += 1
            subscribers = tuple(shared.subscribers)
            shared.subscribers.clear()
            self._condition.notify_all()

        for subscriber in subscribers:
            if subscriber.cancelled():
                continue
            try:
                if exception is None:
                    subscriber.set_result(result)
                else:
                    subscriber.set_exception(exception)
            except concurrent.futures.InvalidStateError:
                # A caller may cancel after the check above.  Its cancellation
                # must not affect backend work or any other subscriber.
                continue

    def drain(
        self,
        timeout: Optional[float] = None,
        *,
        raise_on_error: bool = False,
    ) -> tuple[SVDChunkL3TaskFailure, ...]:
        """Wait for accepted work and consume failures completed since last drain.

        ``TimeoutError`` leaves the manager and its retained failures intact.
        Backend exceptions are always delivered through the corresponding
        submission futures.  ``raise_on_error`` additionally raises one
        aggregate after all work has settled.
        """

        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a non-negative number or None")
            if timeout < 0:
                raise ValueError("timeout must be non-negative")
            deadline = time.monotonic() + timeout
        else:
            deadline = None

        with self._condition:
            while self._pending_tasks:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"timed out with {self._pending_tasks} SVD L3 task(s) pending"
                    )
                self._condition.wait(remaining)
            failures = tuple(self._failures)
            self._failures.clear()

        if failures and raise_on_error:
            raise SVDChunkL3TaskFailures(failures)
        return failures

    def close(self) -> None:
        """Reject new distinct work while allowing already-accepted work to finish."""

        with self._condition:
            self._closed = True

    def shutdown(
        self,
        *,
        wait: bool = True,
        raise_on_error: bool = False,
    ) -> tuple[SVDChunkL3TaskFailure, ...]:
        """Close the manager and optionally wait for all accepted work.

        Pending tasks are never cancelled, including when ``wait`` is false.
        A later ``drain`` or ``shutdown(wait=True)`` may be used to join them.
        """

        self.close()
        if not wait:
            self._read_executor.shutdown(wait=False, cancel_futures=False)
            self._put_executor.shutdown(wait=False, cancel_futures=False)
            return ()

        failures = self.drain(raise_on_error=False)
        self._read_executor.shutdown(wait=True, cancel_futures=False)
        self._put_executor.shutdown(wait=True, cancel_futures=False)
        if failures and raise_on_error:
            raise SVDChunkL3TaskFailures(failures)
        return failures

    def stats_snapshot(self) -> SVDChunkL3TaskManagerStats:
        """Return a self-consistent view of counters and pending work."""

        with self._condition:
            return SVDChunkL3TaskManagerStats(
                max_workers=self._max_workers,
                max_pending_tasks=self._max_pending_tasks,
                closed=self._closed,
                pending_tasks=self._pending_tasks,
                pending_reads=len(self._reads),
                pending_puts=len(self._puts),
                max_observed_pending_tasks=self._max_observed_pending_tasks,
                read_requests=self._read_requests,
                read_tasks_submitted=self._read_tasks_submitted,
                read_requests_coalesced=self._read_requests_coalesced,
                read_tasks_completed=self._read_tasks_completed,
                read_tasks_failed=self._read_tasks_failed,
                put_requests=self._put_requests,
                put_tasks_submitted=self._put_tasks_submitted,
                put_requests_coalesced=self._put_requests_coalesced,
                put_tasks_completed=self._put_tasks_completed,
                put_tasks_failed=self._put_tasks_failed,
                queue_full_rejections=self._queue_full_rejections,
                closed_rejections=self._closed_rejections,
                subscriber_cancellations=self._subscriber_cancellations,
            )

    def __enter__(self) -> SVDChunkL3TaskManager:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown(wait=True, raise_on_error=exc_type is None)


__all__ = [
    "SVDChunkL3Backend",
    "SVDChunkL3QueueFull",
    "SVDChunkL3TaskFailure",
    "SVDChunkL3TaskFailures",
    "SVDChunkL3TaskManager",
    "SVDChunkL3TaskManagerClosed",
    "SVDChunkL3TaskManagerError",
    "SVDChunkL3TaskManagerStats",
]
