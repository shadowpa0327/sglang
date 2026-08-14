"""CPU tests for the bounded asynchronous SVD chunk L3 task manager.

python test/registered/unit/mem_cache/test_hicache_svd_l3_tasks.py -v
"""

from __future__ import annotations

import threading
import time
import unittest
from collections import Counter
from typing import Any, Optional

from sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_l3_tasks import (
    SVDChunkL3QueueFull,
    SVDChunkL3TaskFailures,
    SVDChunkL3TaskManager,
    SVDChunkL3TaskManagerClosed,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ControlledBackend:
    def __init__(self, *, blocked: bool = True) -> None:
        self.release = threading.Event()
        if not blocked:
            self.release.set()
        self.started = threading.Event()
        self._lock = threading.Lock()
        self.read_calls: Counter[str] = Counter()
        self.put_calls: Counter[str] = Counter()
        self.values: dict[str, Any] = {}
        self.read_failures: dict[str, BaseException] = {}
        self.put_failures: dict[str, BaseException] = {}
        self.put_results: dict[str, bool] = {}
        self.active = 0
        self.max_active = 0

    def _enter(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test backend was not released")

    def _leave(self) -> None:
        with self._lock:
            self.active -= 1

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self.read_calls[key] += 1
        self._enter()
        try:
            failure = self.read_failures.get(key)
            if failure is not None:
                raise failure
            return self.values.get(key, f"read:{key}")
        finally:
            self._leave()

    def put(self, key: str, value: Any) -> bool:
        with self._lock:
            self.put_calls[key] += 1
        self._enter()
        try:
            failure = self.put_failures.get(key)
            if failure is not None:
                raise failure
            result = self.put_results.get(key, True)
            if not result:
                return False
            self.values.setdefault(key, value)
            return True
        finally:
            self._leave()


class _PutBlockingBackend:
    """Block writes independently while allowing reads to finish immediately."""

    def __init__(self) -> None:
        self.put_started = threading.Event()
        self.release_put = threading.Event()

    def get(self, key: str) -> str:
        return f"read:{key}"

    def put(self, key: str, value: Any) -> bool:
        del key, value
        self.put_started.set()
        if not self.release_put.wait(timeout=5):
            raise TimeoutError("test put was not released")
        return True


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        time.sleep(0.001)


class TestSVDChunkL3TaskManager(unittest.TestCase):
    def test_configuration_validation(self):
        backend = _ControlledBackend()
        for name, value in (("max_workers", 0), ("max_pending_tasks", 0)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                SVDChunkL3TaskManager(backend, **{name: value})
        with self.assertRaises(TypeError):
            SVDChunkL3TaskManager(object())
        with self.assertRaises(ValueError):
            SVDChunkL3TaskManager(backend, max_workers=2, max_pending_tasks=1)
        with self.assertRaises(TypeError):
            SVDChunkL3TaskManager(backend, read_transform=object())

    def test_read_transform_runs_once_inside_the_coalesced_backend_task(self):
        backend = _ControlledBackend()
        calls = []

        def transform(key, value):
            calls.append((key, value, threading.current_thread().name))
            return f"parsed:{value}"

        manager = SVDChunkL3TaskManager(
            backend,
            max_workers=1,
            max_pending_tasks=1,
            read_transform=transform,
        )
        first = manager.submit_read("same")
        self.assertTrue(backend.started.wait(timeout=2))
        second = manager.submit_read("same")
        backend.release.set()
        self.assertEqual(first.result(timeout=2), "parsed:read:same")
        self.assertEqual(second.result(timeout=2), "parsed:read:same")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][2].startswith("SVDChunkL3"))
        manager.shutdown()

    def test_reads_coalesce_and_one_subscriber_can_cancel(self):
        backend = _ControlledBackend()
        manager = SVDChunkL3TaskManager(backend, max_workers=1, max_pending_tasks=2)
        first = manager.submit_read("same")
        self.assertTrue(backend.started.wait(timeout=2))
        second = manager.submit_read("same")

        self.assertIsNot(first, second)
        self.assertTrue(first.cancel())
        backend.release.set()
        self.assertEqual(second.result(timeout=2), "read:same")
        self.assertTrue(first.cancelled())
        self.assertEqual(manager.drain(), ())

        stats = manager.stats_snapshot()
        self.assertEqual(backend.read_calls, Counter({"same": 1}))
        self.assertEqual(stats.read_requests, 2)
        self.assertEqual(stats.read_tasks_submitted, 1)
        self.assertEqual(stats.read_requests_coalesced, 1)
        self.assertEqual(stats.read_tasks_completed, 1)
        self.assertEqual(stats.subscriber_cancellations, 1)
        manager.shutdown()

    def test_puts_coalesce_and_continue_after_every_subscriber_cancels(self):
        backend = _ControlledBackend()
        manager = SVDChunkL3TaskManager(backend, max_workers=1, max_pending_tasks=2)
        first = manager.submit_put("immutable", b"first")
        self.assertTrue(backend.started.wait(timeout=2))
        second = manager.submit_put("immutable", b"first")

        self.assertIsNot(first, second)
        self.assertTrue(first.cancel())
        self.assertTrue(second.cancel())
        backend.release.set()
        self.assertEqual(manager.drain(), ())

        stats = manager.stats_snapshot()
        self.assertEqual(backend.put_calls, Counter({"immutable": 1}))
        self.assertEqual(backend.values["immutable"], b"first")
        self.assertEqual(stats.put_requests, 2)
        self.assertEqual(stats.put_tasks_submitted, 1)
        self.assertEqual(stats.put_requests_coalesced, 1)
        self.assertEqual(stats.put_tasks_completed, 1)
        self.assertEqual(stats.subscriber_cancellations, 2)
        manager.shutdown()

    def test_false_put_result_is_completion_not_failure(self):
        backend = _ControlledBackend(blocked=False)
        backend.put_results["rejected"] = False
        manager = SVDChunkL3TaskManager(backend, max_workers=1, max_pending_tasks=1)

        self.assertFalse(manager.submit_put("rejected", b"blob").result(timeout=2))
        self.assertEqual(manager.drain(), ())
        stats = manager.stats_snapshot()
        self.assertEqual(stats.put_tasks_completed, 1)
        self.assertEqual(stats.put_tasks_failed, 0)
        self.assertNotIn("rejected", backend.values)
        manager.shutdown()

    def test_blocked_put_does_not_block_read_with_one_worker(self):
        backend = _PutBlockingBackend()
        manager = SVDChunkL3TaskManager(
            backend,
            max_workers=1,
            max_pending_tasks=2,
        )
        put = manager.submit_put("slow-put", b"blob")
        self.assertTrue(backend.put_started.wait(timeout=2))

        try:
            read = manager.submit_read("latency-sensitive")
            self.assertEqual(read.result(timeout=2), "read:latency-sensitive")
            self.assertFalse(put.done())
        finally:
            backend.release_put.set()
            manager.shutdown(wait=True)

        self.assertTrue(put.result(timeout=0))

    def test_workers_and_distinct_pending_tasks_are_bounded(self):
        backend = _ControlledBackend()
        manager = SVDChunkL3TaskManager(backend, max_workers=1, max_pending_tasks=2)
        first = manager.submit_read("a")
        self.assertTrue(backend.started.wait(timeout=2))
        second = manager.submit_read("b")
        duplicate = manager.submit_read("a")
        rejected = manager.submit_read("c")

        with self.assertRaises(SVDChunkL3QueueFull):
            rejected.result(timeout=0)
        before_release = manager.stats_snapshot()
        self.assertEqual(before_release.pending_tasks, 2)
        self.assertEqual(before_release.read_requests_coalesced, 1)
        self.assertEqual(before_release.queue_full_rejections, 1)

        backend.release.set()
        self.assertEqual(first.result(timeout=2), "read:a")
        self.assertEqual(duplicate.result(timeout=2), "read:a")
        self.assertEqual(second.result(timeout=2), "read:b")
        manager.drain()
        self.assertEqual(backend.max_active, 1)
        self.assertEqual(manager.stats_snapshot().max_observed_pending_tasks, 2)
        manager.shutdown()

    def test_backend_exceptions_reach_every_subscriber_and_drain(self):
        backend = _ControlledBackend()
        read_error = OSError("read exploded")
        put_error = RuntimeError("put exploded")
        backend.read_failures["read-error"] = read_error
        backend.put_failures["put-error"] = put_error
        manager = SVDChunkL3TaskManager(backend, max_workers=2, max_pending_tasks=4)

        read_a = manager.submit_read("read-error")
        read_b = manager.submit_read("read-error")
        put = manager.submit_put("put-error", b"blob")
        backend.release.set()

        for future, expected in (
            (read_a, read_error),
            (read_b, read_error),
            (put, put_error),
        ):
            with self.subTest(future=future), self.assertRaises(
                type(expected)
            ) as caught:
                future.result(timeout=2)
            self.assertIs(caught.exception, expected)

        failures = manager.drain()
        self.assertEqual(
            {(failure.operation, failure.key) for failure in failures},
            {("read", "read-error"), ("put", "put-error")},
        )
        stats = manager.stats_snapshot()
        self.assertEqual(stats.read_tasks_failed, 1)
        self.assertEqual(stats.put_tasks_failed, 1)
        self.assertEqual(stats.read_requests_coalesced, 1)

        backend.release.clear()
        backend.read_failures["another"] = ValueError("another failure")
        another = manager.submit_read("another")
        _wait_until(lambda: backend.read_calls["another"] == 1)
        backend.release.set()
        with self.assertRaises(ValueError):
            another.result(timeout=2)
        with self.assertRaises(SVDChunkL3TaskFailures) as aggregate:
            manager.drain(raise_on_error=True)
        self.assertEqual(len(aggregate.exception.failures), 1)
        manager.shutdown()

    def test_drain_timeout_does_not_cancel_work(self):
        backend = _ControlledBackend()
        manager = SVDChunkL3TaskManager(backend, max_workers=1, max_pending_tasks=1)
        future = manager.submit_read("slow")
        self.assertTrue(backend.started.wait(timeout=2))
        with self.assertRaises(TimeoutError):
            manager.drain(timeout=0.001)
        self.assertFalse(future.done())

        backend.release.set()
        self.assertEqual(future.result(timeout=2), "read:slow")
        self.assertEqual(manager.drain(), ())
        manager.shutdown()

    def test_shutdown_waits_rejects_new_work_and_is_idempotent(self):
        backend = _ControlledBackend()
        manager = SVDChunkL3TaskManager(backend, max_workers=1, max_pending_tasks=1)
        accepted = manager.submit_put("durable", b"blob")
        self.assertTrue(backend.started.wait(timeout=2))
        shutdown_done = threading.Event()

        def shutdown() -> None:
            manager.shutdown(wait=True)
            shutdown_done.set()

        thread = threading.Thread(target=shutdown)
        thread.start()
        _wait_until(lambda: manager.closed)
        rejected = manager.submit_read("too-late")
        with self.assertRaises(SVDChunkL3TaskManagerClosed):
            rejected.result(timeout=0)
        self.assertFalse(shutdown_done.is_set())

        backend.release.set()
        self.assertTrue(accepted.result(timeout=2))
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(shutdown_done.is_set())
        self.assertEqual(backend.values["durable"], b"blob")
        self.assertEqual(manager.stats_snapshot().closed_rejections, 1)
        self.assertEqual(manager.shutdown(wait=True), ())

    def test_shutdown_without_wait_does_not_cancel_accepted_put(self):
        backend = _ControlledBackend()
        manager = SVDChunkL3TaskManager(backend, max_workers=1, max_pending_tasks=1)
        future = manager.submit_put("background", b"blob")
        self.assertTrue(backend.started.wait(timeout=2))
        self.assertEqual(manager.shutdown(wait=False), ())
        self.assertFalse(future.cancelled())

        backend.release.set()
        self.assertTrue(future.result(timeout=2))
        self.assertEqual(manager.drain(), ())
        self.assertEqual(manager.shutdown(wait=True), ())


if __name__ == "__main__":
    unittest.main()
