"""Boundary tests for the scheduler's waiting / running request timeouts.

Both paths are pure bookkeeping over timestamps -- no model, no GPU, no draft
worker -- so they are driven here directly instead of through a server. The
e2e side (503 reaching the client, server stays up) is covered by
scheduler/test_scheduler_control.py.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.disaggregation.utils import DisaggregationMode

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _FakeReq:
    """Must stay hashable: the waiting-timeout path collects drops in a set."""

    def __init__(self, rid, wait_entry=0.0, forward_entry=0.0, is_finished=False):
        self.rid = rid
        self.to_finish = None
        self._finished = is_finished
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=wait_entry,
            forward_entry_time=forward_entry,
            trace_ctx=MagicMock(),
        )

    def finished(self):
        return self._finished


def _req(
    rid: str, *, wait_entry: float = 0.0, forward_entry: float = 0.0, finished=False
):
    return _FakeReq(rid, wait_entry, forward_entry, finished)


def _scheduler(waiting_queue):
    s = Scheduler.__new__(Scheduler)
    s.waiting_queue = waiting_queue
    s.enable_hicache_storage = False
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    return s


class TestWaitingTimeout(CustomTestCase):
    def test_drops_only_reqs_past_the_deadline(self):
        now = time.perf_counter()
        stale = _req("stale", wait_entry=now - 10)
        fresh = _req("fresh", wait_entry=now)
        s = _scheduler([stale, fresh])

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            s._abort_on_waiting_timeout()

        self.assertEqual([r.rid for r in s.waiting_queue], ["fresh"])
        self.assertEqual(s.ipc_channels.send_to_tokenizer.send_output.call_count, 1)

    def test_unset_entry_time_is_never_dropped(self):
        # 0 is the "not yet stamped" sentinel; the guard is `0 < entry_time`.
        s = _scheduler([_req("unstamped", wait_entry=0.0)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1e-9):
            s._abort_on_waiting_timeout()
        self.assertEqual(len(s.waiting_queue), 1)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_disabled_timeout_is_a_no_op(self):
        s = _scheduler([_req("stale", wait_entry=time.perf_counter() - 100)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0):
            s._abort_on_waiting_timeout()
        self.assertEqual(len(s.waiting_queue), 1)

    def test_hierarchical_cache_without_generic_storage_releases_lookup(self):
        stale = _req("svd-stale", wait_entry=time.perf_counter() - 10)
        s = _scheduler([stale])
        s.enable_hierarchical_cache = True
        s.tree_cache = MagicMock()

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            s._abort_on_waiting_timeout()

        s.tree_cache.terminate_prefetch.assert_called_once_with("svd-stale")


class TestRunningTimeout(CustomTestCase):
    @staticmethod
    def _batch(reqs):
        return SimpleNamespace(reqs=reqs, is_empty=lambda: not reqs)

    def test_marks_only_stale_unfinished_reqs(self):
        now = time.perf_counter()
        stale = _req("stale", forward_entry=now - 10)
        fresh = _req("fresh", forward_entry=now)
        done = _req("done", forward_entry=now - 10, finished=True)
        s = _scheduler([])

        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            s._abort_on_running_timeout(self._batch([stale, fresh, done]))

        self.assertIsNotNone(stale.to_finish)
        self.assertIsNone(fresh.to_finish)
        self.assertIsNone(done.to_finish, "a finished req must not be aborted")

    def test_unset_forward_entry_time_is_never_marked(self):
        s = _scheduler([])
        req = _req("unstamped", forward_entry=0.0)
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1e-9):
            s._abort_on_running_timeout(self._batch([req]))
        self.assertIsNone(req.to_finish)

    def test_empty_batch_and_disabled_timeout_are_no_ops(self):
        s = _scheduler([])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            s._abort_on_running_timeout(self._batch([]))
        req = _req("stale", forward_entry=time.perf_counter() - 100)
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(0):
            s._abort_on_running_timeout(self._batch([req]))
        self.assertIsNone(req.to_finish)


class TestSchedulerHiCacheIdle(CustomTestCase):
    @staticmethod
    def _idle_scheduler(tree_cache):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = SimpleNamespace(is_empty=lambda: True)
        scheduler.chunked_req = None
        scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler.result_queue = []
        scheduler._pp_microbatches_drained = lambda: True
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = True
        scheduler.tree_cache = tree_cache
        return scheduler

    def test_svd_write_may_sleep_but_does_not_allow_destructive_idle(self):
        tree_cache = SimpleNamespace(
            hicache_writes_allow_idle_sleep=True,
            ongoing_write_through={1: object()},
            ongoing_load_back={},
            enable_storage=False,
        )
        scheduler = self._idle_scheduler(tree_cache)

        self.assertFalse(scheduler.is_fully_idle())
        self.assertTrue(scheduler.is_fully_idle(allow_background_hicache=True))

        # Stock cache controllers keep their existing polling contract.
        tree_cache.hicache_writes_allow_idle_sleep = False
        self.assertFalse(scheduler.is_fully_idle(allow_background_hicache=True))

    def test_background_svd_write_yields_without_idle_sleeper(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.idle_sleeper = None

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler.time.sleep"
        ) as sleep_mock:
            scheduler.maybe_sleep_on_idle(timeout_ms=5, yield_if_missing=True)
            sleep_mock.assert_called_once_with(0.005)

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler.time.sleep"
        ) as sleep_mock:
            scheduler.maybe_sleep_on_idle(timeout_ms=5)
            sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
