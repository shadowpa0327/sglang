"""Compressed-only restores start only for admitted requests (PrefillAdder + linker)."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGE = 16
BOUNDARY = 64
PROMPT = 200
RESTORED = 192  # whole pages of the 199-token match limit
STATE = 7.0


class _StoreLinker(UnifiedCacheLinker):
    """Compressed-only store: every lookup hits, and a load scatters STATE into
    the transfer's recurrent-state slot when the batch starts loading."""

    compressed_only = True

    def __init__(self, pool):
        self.pool = pool
        self.layer_done_counter = object()
        self.queued = {}
        self.restored = []
        self.events = []

    def lookup(self, rid, transfers):
        return [len(transfers[0].keys)]

    def load(self, rid, transfers):
        self.queued[rid] = transfers
        return True

    def start_layer_wise_loading(self):
        cache = self.pool.mamba_pool.mamba_cache
        for rid, transfers in self.queued.items():
            state = next(t for t in transfers if t.name == PoolName.MAMBA)
            slot = int(state.device_indices[0])
            cache.temporal[:, slot] = STATE
            for conv in cache.conv:
                conv[:, slot] = STATE
            self.restored.append(rid)
        self.queued.clear()
        return 1

    def cancel_queued_load(self, rid):
        return self.queued.pop(rid, None) is not None

    def num_completed_loads(self):
        return 0

    def pop_completed_load(self):
        raise AssertionError("not drained in this test")

    def offload(self, transfers):
        return False

    def num_completed_offloads(self):
        return 0

    def pop_completed_offload(self):
        raise AssertionError("never offloads")

    def record(self, event):
        self.events.append(event)

    def reset(self):
        pass

    def close(self):
        pass


class _KvSlots:
    def __init__(self, size):
        self.free_slots = list(range(1, size + 1))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, num_tokens):
        if num_tokens > len(self.free_slots):
            return None
        slots, self.free_slots = (
            self.free_slots[:num_tokens],
            self.free_slots[num_tokens:],
        )
        return torch.tensor(slots, dtype=torch.int64)

    def free(self, slots):
        self.free_slots += slots.tolist()


def _mamba_pool():
    layers = [0, 1]
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=1,
        num_heads=2,
        head_dim=32,
        state_size=8,
        conv_kernel=4,
    )
    return HybridReqToTokenPool(
        size=4,
        mamba_size=8,
        mamba_spec_state_size=4,
        max_context_len=PROMPT + 16,
        device="cpu",
        enable_memory_saver=False,
        cache_params=Mamba2CacheParams(shape=shape, layers=layers),
        mamba_layer_ids=layers,
        enable_mamba_extra_buffer=False,
    )


def _component(cls, cache):
    component = cls.__new__(cls)
    component.cache = cache
    return component


def _req(rid, prompt_len):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", range(prompt_len)),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    req.init_next_round_input()
    return req


class TestPrivateRestoreAdmission(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.pool = _mamba_pool()
        self.store = _StoreLinker(self.pool)
        empty = MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=0,
            last_host_node=0,
            best_match_node=0,
        )
        cache = SimpleNamespace(
            tree_core=SimpleNamespace(empty_match_result=empty),
            write_through_threshold=256,
            disable=True,
            page_size=PAGE,
            is_swa_enabled=False,
            req_to_token_pool=self.pool,
            token_to_kv_pool_allocator=_KvSlots(4096),
            inc_lock_ref=lambda node: IncLockRefResult(),
            dec_lock_ref=lambda node, params: DecLockRefResult(),
            _all_reduce_attn_groups=lambda mask, op: None,
        )
        cache._components_tuple = (
            _component(FullComponent, cache),
            _component(MambaComponent, cache),
        )
        self.linker = UnifiedCacheLinkerWrapper(cache, self.store)
        # UnifiedRadixCache in compressed-only mode: match and load-back go
        # straight to the linker.
        tree_cache = MagicMock()
        tree_cache.disable = True
        tree_cache.linker = self.linker
        tree_cache.supports_mamba.return_value = True
        tree_cache.swa_reprefill_tail_tokens.return_value = 0
        tree_cache.full_evictable_size.return_value = 0
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        tree_cache.match_prefix.side_effect = lambda params: self.linker.match(
            params.key, params.req, empty
        )
        tree_cache.init_load_back.side_effect = (
            lambda params: UnifiedRadixCache.init_load_back(
                SimpleNamespace(buffer_pipeline=None, linker=self.linker), params
            )
        )
        self.tree_cache = tree_cache

    def _adder(self, rem_chunk_tokens):
        allocator = MagicMock()
        allocator.available_size.return_value = 100_000
        running = MagicMock()
        running.reqs = []
        return PrefillAdder(
            page_size=PAGE,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=running,
            new_token_ratio=1.0,
            rem_input_tokens=10_000,
            rem_chunk_tokens=rem_chunk_tokens,
            prefill_boundary_tokens=BOUNDARY,
        )

    def _restore_and_forward(self, req):
        """Run the admitting batch: queued loads, prepare_for_extend, deferred clear."""
        self.linker.start_layer_wise_loading()
        self.pool.alloc([req])
        batch = SimpleNamespace()
        ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])
        if batch.mamba_clear_indices is not None:
            self.pool.mamba_pool.clear_slots(
                self.pool.translate_mamba_indices(batch.mamba_clear_indices)
            )
        slot = self.pool.get_mamba_indices(torch.tensor([req.kv.req_pool_idx]))[0]
        return float(self.pool.mamba_pool.mamba_cache.temporal[:, slot].max())

    def _restored(self, rid):
        """A request whose private restore was started by an admitting round."""
        req = _req(rid, PROMPT)
        req.init_next_round_input(self.tree_cache)
        indices, _ = self.linker.load_back(req)
        self.assertEqual(len(indices), RESTORED)
        return req

    def test_deferred_restore_reaches_the_first_forward(self):
        """A compressed-only restore writes the recurrent state into a slot bound
        to the request. When a round rejects the request (another request is
        mid-chunk), the forward of the round that admits it must still read the
        restored state, not a zeroed slot."""
        scored = _req("scored", PROMPT)
        inflight = _req("inflight", 1000)
        inflight.prefix_indices = torch.arange(128)
        mamba_free = self.pool.mamba_allocator.available_size()

        adder = self._adder(rem_chunk_tokens=BOUNDARY)
        self.assertIs(adder.add_chunked_req(inflight), inflight)
        scored.init_next_round_input(self.tree_cache)
        self.assertEqual(scored.host_hit_length, RESTORED)
        self.assertEqual(
            adder.add_one_req(scored, has_chunked_req=True, truncation_align_size=None),
            AddReqResult.OTHER,
        )
        self.assertEqual(adder.can_run_list, [inflight])
        # The scheduler's rejection branch releases a slot the request holds.
        if scored.kv.holds_mamba:
            self.pool.mamba_allocator.free(scored.kv.mamba_pool_idx.unsqueeze(-1))
            scored.kv.mamba_pool_idx = None
        self.linker.start_layer_wise_loading()
        self.assertEqual(self.pool.mamba_allocator.available_size(), mamba_free)

        adder = self._adder(rem_chunk_tokens=BOUNDARY)
        scored.init_next_round_input(self.tree_cache)
        adder.add_one_req(scored, has_chunked_req=False, truncation_align_size=None)
        self.assertEqual(adder.can_run_list, [scored])
        self.assertEqual(len(scored.prefix_indices), RESTORED)
        self.assertEqual(self._restore_and_forward(scored), STATE)
        self.assertEqual(
            self.linker.restore_integrity(),
            {
                "private_restores": 1,
                "restored_states_checked": 1,
                "restored_states_lost": 0,
            },
        )

    def test_tail_that_fits_beside_a_chunked_request_restores_this_round(self):
        """The projection must not reject on another request being mid-chunk
        alone: a compressed request whose restored tail fits the chunk budget
        left by that request's last chunk is admitted and restored this round."""
        scored = _req("scored", PROMPT)
        inflight = _req("inflight", 160)
        inflight.prefix_indices = torch.arange(128)

        adder = self._adder(rem_chunk_tokens=BOUNDARY)
        self.assertIsNone(adder.add_chunked_req(inflight))
        scored.init_next_round_input(self.tree_cache)
        adder.add_one_req(scored, has_chunked_req=True, truncation_align_size=None)

        self.assertEqual(adder.can_run_list, [inflight, scored])
        self.assertEqual(len(scored.prefix_indices), RESTORED)
        self.assertEqual(self._restore_and_forward(scored), STATE)

    def test_slot_handed_back_zeroed_counts_as_lost(self):
        """The mamba allocator is FIFO and can hand a freed slot index straight
        back with a pending clear; the integrity check must count that restore as
        lost even though the request holds the same index again."""
        req = self._restored("moved")
        slot = req.kv.mamba_pool_idx
        allocator = self.pool.mamba_allocator
        allocator.free(slot.unsqueeze(-1))
        allocator.alloc(allocator.available_size() - 1)
        req.kv.mamba_pool_idx = allocator.alloc(1)[0]
        req.kv.mamba_needs_clear = True
        self.assertEqual(int(req.kv.mamba_pool_idx), int(slot))

        self.assertEqual(self._restore_and_forward(req), 0.0)

        self.assertEqual(self.linker.restored_states_checked, 1)
        self.assertEqual(self.linker.restored_states_lost, 1)

    def test_restore_returning_fewer_tokens_than_projected_raises(self):
        """Admission was decided on the projected restored prefix; a restore that
        comes back short (here: no KV slots, so nothing) must raise instead of
        admitting the request on a plan it was never checked against."""
        req = _req("short", PROMPT)
        self.linker.cache.token_to_kv_pool_allocator = _KvSlots(0)
        self.linker.cache.evict = lambda params: None
        adder = self._adder(rem_chunk_tokens=BOUNDARY)
        req.init_next_round_input(self.tree_cache)

        with self.assertRaisesRegex(RuntimeError, "loaded 0 tokens"):
            adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

    def test_post_load_plan_must_equal_the_projection(self):
        """The queued restore runs with this round's batch, so a post-load
        verdict that differs from the projected one must raise rather than
        reject the request and strand the restore."""
        req = _req("drift", PROMPT)
        adder = self._adder(rem_chunk_tokens=BOUNDARY)
        plan_extend = adder._plan_extend
        plans = []

        def drifting_plan(req, **kwargs):
            plans.append(plan_extend(req, **kwargs))
            if len(plans) == 1:
                return plans[0]
            return plans[-1]._replace(rejection=AddReqResult.OTHER)

        adder._plan_extend = drifting_plan
        req.init_next_round_input(self.tree_cache)

        with self.assertRaisesRegex(RuntimeError, "changed its admission plan"):
            adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertEqual(len(plans), 2)

    def test_reentering_load_back_after_a_restore_raises(self):
        """Restores start only for admitted requests; a second load-back for a
        restored request must count a loss and raise, never reuse or redo the
        restore, even while the request still holds its slot."""
        req = self._restored("reentry")

        with self.assertRaisesRegex(RuntimeError, "re-entered load-back"):
            self.linker.load_back(req)
        self.assertEqual(self.linker.restored_states_lost, 1)
        self.assertEqual(self.linker.private_restores, 1)
        self.assertEqual(list(self.store.queued), ["reentry"])


if __name__ == "__main__":
    unittest.main()
