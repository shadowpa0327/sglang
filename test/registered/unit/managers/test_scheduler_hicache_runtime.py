import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerHiCacheRuntime(CustomTestCase):
    def test_generic_and_svd_snapshots_coexist(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = SimpleNamespace(
            hicache_runtime_snapshot=lambda: {
                "quiescent": True,
                "backend": "hiradix",
            },
            runtime_metrics_snapshot=lambda: {"connector": {"restored_chunks": 7}},
        )
        state = {}

        scheduler._add_hicache_runtime_snapshots(state)

        self.assertTrue(state["hicache_runtime"]["quiescent"])
        self.assertEqual(state["hicache_runtime"]["backend"], "hiradix")
        self.assertEqual(
            state["hicache_svd_runtime"]["connector"]["restored_chunks"], 7
        )

    def test_unsupported_cache_adds_no_runtime_fields(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = SimpleNamespace()
        state = {"existing": "preserved"}

        scheduler._add_hicache_runtime_snapshots(state)

        self.assertEqual(state, {"existing": "preserved"})


if __name__ == "__main__":
    unittest.main()
