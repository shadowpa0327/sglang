"""Unit tests for SMC helper state and resampling utilities."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.smc_info import SMCScoreInput
from sglang.srt.speculative.smc_info import (
    SMCParticleState,
    SMCRequestState,
    effective_sample_size,
    multinomial_resample,
    normalize_log_weights,
    resolve_smc_proposal_length,
    resolve_smc_seed_output_ids,
    systematic_resample,
    validate_smc_parent_req,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


def _make_particle(output_len: int, log_weight: float, finished: bool = False):
    target_req = MagicMock()
    target_req.output_ids = list(range(output_len))
    target_req.finished.return_value = finished
    target_req.finished_reason = None
    target_req.finished_len = None

    draft_req = MagicMock()
    draft_req.output_ids = list(range(output_len))
    draft_req.finished.return_value = finished

    return SMCParticleState(
        target_req=target_req,
        draft_req=draft_req,
        log_weight=log_weight,
    )


class TestSMCWeightHelpers(TestCase):
    def test_normalize_log_weights(self):
        normalized = normalize_log_weights([0.0, 0.0, 0.0])
        self.assertTrue(
            torch.allclose(normalized, torch.full((3,), 1.0 / 3.0, dtype=torch.float64))
        )

    def test_effective_sample_size(self):
        self.assertAlmostEqual(effective_sample_size([0.5, 0.5]), 2.0)
        self.assertAlmostEqual(effective_sample_size([1.0, 0.0]), 1.0)

    def test_systematic_resample_with_degenerate_weight(self):
        self.assertEqual(systematic_resample([1.0, 0.0, 0.0]), [0, 0, 0])

    def test_multinomial_resample_with_degenerate_weight(self):
        self.assertEqual(multinomial_resample([1.0, 0.0, 0.0]), [0, 0, 0])


class TestSMCScoreInput(TestCase):
    def _make_score_input(self, draft_token_num: int = 4) -> SMCScoreInput:
        return SMCScoreInput(
            draft_token=torch.tensor([17, 23, 29, 29], dtype=torch.int64),
            draft_lengths=torch.tensor([2], dtype=torch.int32),
            draft_logprobs=torch.tensor([0.5], dtype=torch.float32),
            draft_finished=torch.tensor([False], dtype=torch.bool),
            positions=torch.tensor([3, 4, 5, 6], dtype=torch.int64),
            custom_mask=torch.ones((28,), dtype=torch.bool),
            draft_token_num=draft_token_num,
        )

    @patch("sglang.srt.speculative.smc_info.ForwardBatch.init_new")
    @patch("sglang.srt.speculative.smc_info.assign_extend_cache_locs_func")
    def test_prepare_for_v2_verify_uses_graph_runner_when_available(
        self, mock_assign_cache_locs, mock_init_forward_batch
    ):
        score_input = self._make_score_input()
        req = SimpleNamespace(req_pool_idx=3, kv_allocated_len=5, rid="r-1")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([3], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            input_ids=None,
            reqs=[req],
            capture_hidden_mode=None,
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((8, 32), dtype=torch.int32),
            write=MagicMock(),
        )
        allocator = MagicMock()
        allocator.alloc.return_value = torch.tensor([101, 102, 103], dtype=torch.int64)
        graph_runner = MagicMock()
        graph_runner.can_run.return_value = True
        attn_backend = MagicMock()
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool_allocator=allocator,
                graph_runner=graph_runner,
                attn_backend=attn_backend,
            )
        )
        mock_assign_cache_locs.return_value = torch.tensor(
            [11, 12, 13, 14], dtype=torch.int64
        )
        fake_forward_batch = object()
        mock_init_forward_batch.return_value = fake_forward_batch

        verify_forward_batch, can_run_cuda_graph = score_input.prepare_for_v2_verify(
            req_to_token_pool,
            batch,
            target_worker,
        )

        self.assertIs(verify_forward_batch, fake_forward_batch)
        self.assertTrue(can_run_cuda_graph)
        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(batch.capture_hidden_mode, CaptureHiddenMode.NULL)
        self.assertTrue(torch.equal(batch.input_ids, score_input.draft_token))
        self.assertTrue(
            torch.equal(batch.out_cache_loc, mock_assign_cache_locs.return_value)
        )
        self.assertEqual(req.kv_allocated_len, 8)
        write_call = req_to_token_pool.write.call_args
        self.assertEqual(write_call.args[0], (3, slice(5, 8)))
        self.assertTrue(
            torch.equal(write_call.args[1], torch.tensor([101, 102, 103], dtype=torch.int32))
        )
        graph_runner.replay_prepare.assert_called_once_with(fake_forward_batch)
        attn_backend.init_forward_metadata.assert_not_called()

    @patch("sglang.srt.speculative.smc_info.ForwardBatch.init_new")
    @patch("sglang.srt.speculative.smc_info.assign_extend_cache_locs_func")
    def test_prepare_for_v2_verify_falls_back_to_attn_backend_without_graph(
        self, mock_assign_cache_locs, mock_init_forward_batch
    ):
        score_input = self._make_score_input()
        req = SimpleNamespace(req_pool_idx=2, kv_allocated_len=8, rid="r-2")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([2], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            input_ids=None,
            reqs=[req],
            capture_hidden_mode=None,
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((8, 32), dtype=torch.int32),
            write=MagicMock(),
        )
        graph_runner = MagicMock()
        graph_runner.can_run.return_value = False
        attn_backend = MagicMock()
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool_allocator=MagicMock(),
                graph_runner=graph_runner,
                attn_backend=attn_backend,
            )
        )
        mock_assign_cache_locs.return_value = torch.tensor(
            [21, 22, 23, 24], dtype=torch.int64
        )
        fake_forward_batch = object()
        mock_init_forward_batch.return_value = fake_forward_batch

        verify_forward_batch, can_run_cuda_graph = score_input.prepare_for_v2_verify(
            req_to_token_pool,
            batch,
            target_worker,
        )

        self.assertIs(verify_forward_batch, fake_forward_batch)
        self.assertFalse(can_run_cuda_graph)
        attn_backend.init_forward_metadata.assert_called_once_with(fake_forward_batch)
        graph_runner.replay_prepare.assert_not_called()


class TestSMCRequestState(TestCase):
    def test_get_best_particle_prefers_higher_weight(self):
        short_heavier = _make_particle(output_len=2, log_weight=10.0)
        long_lighter = _make_particle(output_len=3, log_weight=1.0)
        state = SMCRequestState(
            parent_req_id="parent",
            particles=[short_heavier, long_lighter],
            n_particles=2,
            gamma=2,
            resample_threshold=0.5,
            resample_method="systematic",
        )

        best = state.get_best_particle()
        self.assertIs(best, short_heavier)
        self.assertEqual(state.best_particle_idx, 0)

    def test_active_particles_excludes_finished(self):
        active = _make_particle(output_len=2, log_weight=0.0, finished=False)
        finished = _make_particle(output_len=3, log_weight=1.0, finished=True)
        state = SMCRequestState(
            parent_req_id="parent",
            particles=[active, finished],
            n_particles=2,
            gamma=2,
            resample_threshold=0.5,
            resample_method="systematic",
        )

        self.assertEqual(state.active_particles(), [active])
        self.assertFalse(state.is_terminal())


class TestValidateSMCParentReq(TestCase):
    def test_validate_rejects_stop_strings_and_hidden_states(self):
        req = MagicMock()
        req.grammar = None
        req.return_logprob = False
        req.return_hidden_states = True
        req.return_routed_experts = False
        req.sampling_params.stop_strs = []
        req.sampling_params.stop_regex_strs = []
        self.assertIn("return_hidden_states", validate_smc_parent_req(req))

        req.return_hidden_states = False
        req.sampling_params.stop_strs = ["stop"]
        self.assertIn("stop strings", validate_smc_parent_req(req))


class TestResolveSMCSeedOutputIds(TestCase):
    def test_uses_overlap_token_when_parent_lags_by_one(self):
        req = MagicMock()
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = []

        seeded = resolve_smc_seed_output_ids(
            req,
            overlap_last_token_id=99,
            overlap_new_seq_len=4,
        )

        self.assertEqual(seeded, [99])

    def test_accepts_already_synchronized_parent(self):
        req = MagicMock()
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [99]

        seeded = resolve_smc_seed_output_ids(
            req,
            overlap_last_token_id=99,
            overlap_new_seq_len=4,
        )

        self.assertEqual(seeded, [99])

    def test_rejects_large_overlap_gap(self):
        req = MagicMock()
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = []

        with self.assertRaisesRegex(ValueError, "at most one missing output token"):
            resolve_smc_seed_output_ids(
                req,
                overlap_last_token_id=99,
                overlap_new_seq_len=5,
            )


class TestResolveSMCProposalLength(TestCase):
    def _make_req(
        self,
        *,
        output_len: int = 0,
        max_new_tokens: int = 16,
        ignore_eos: bool = False,
        stop_token_ids=None,
        eos_token_ids=None,
        tokenizer_eos_id=None,
        additional_stop_token_ids=None,
        vocab_size: int = 32000,
        to_finish=None,
    ):
        req = MagicMock()
        req.finished.return_value = False
        req.output_ids = [1] * output_len
        req.sampling_params.max_new_tokens = max_new_tokens
        req.sampling_params.ignore_eos = ignore_eos
        req.sampling_params.stop_token_ids = stop_token_ids
        req.eos_token_ids = eos_token_ids
        req.tokenizer = (
            None
            if tokenizer_eos_id is None and additional_stop_token_ids is None
            else SimpleNamespace(
                eos_token_id=tokenizer_eos_id,
                additional_stop_token_ids=additional_stop_token_ids,
            )
        )
        req.vocab_size = vocab_size
        req.to_finish = to_finish
        return req

    def test_truncates_at_eos_without_changing_fixed_width_contract(self):
        req = self._make_req(
            eos_token_ids={99},
            tokenizer_eos_id=100,
            additional_stop_token_ids=[101],
        )

        proposal_len, proposal_finished = resolve_smc_proposal_length(
            req, [7, 99, 88, 77]
        )

        self.assertEqual(proposal_len, 2)
        self.assertTrue(proposal_finished)

    def test_ignore_eos_keeps_full_proposal_length(self):
        req = self._make_req(ignore_eos=True, eos_token_ids={99})

        proposal_len, proposal_finished = resolve_smc_proposal_length(
            req, [7, 99, 88, 77]
        )

        self.assertEqual(proposal_len, 4)
        self.assertFalse(proposal_finished)

    def test_max_new_tokens_is_counted_in_sample_phase(self):
        req = self._make_req(output_len=3, max_new_tokens=5)

        proposal_len, proposal_finished = resolve_smc_proposal_length(req, [7, 8, 9])

        self.assertEqual(proposal_len, 2)
        self.assertTrue(proposal_finished)
