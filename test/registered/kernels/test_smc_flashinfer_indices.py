import unittest

import torch

from sglang.srt.speculative.spec_utils import generate_smc_draft_decode_kv_indices
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, suite="stage-b-test-large-1-gpu")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestSMCFlashInferIndices(unittest.TestCase):
    def test_generate_smc_draft_decode_kv_indices_matches_chain_expectation(self):
        device = torch.device("cuda")
        gamma = 3
        raw_bs = 2
        padded_bs = 4
        fill_value = 1

        req_pool_indices = torch.tensor([1, 2, 0, 0], dtype=torch.int64, device=device)
        base_seq_lens = torch.tensor([3, 4, fill_value, fill_value], dtype=torch.int32, device=device)

        req_to_token = torch.zeros((3, 16), dtype=torch.int32, device=device)
        req_to_token[1, :6] = torch.tensor([10, 11, 12, 100, 101, 102], dtype=torch.int32, device=device)
        req_to_token[2, :7] = torch.tensor([20, 21, 22, 23, 200, 201, 202], dtype=torch.int32, device=device)
        req_to_token[0, :4] = torch.tensor([7, 70, 71, 72], dtype=torch.int32, device=device)

        max_kv_len = int(base_seq_lens[:raw_bs].sum().item()) + raw_bs * (gamma - 1) + (padded_bs - raw_bs) * fill_value
        kv_indices = torch.full((gamma, max_kv_len + 8), -1, dtype=torch.int32, device=device)
        kv_indptr = torch.zeros((gamma, padded_bs + 1), dtype=torch.int32, device=device)

        generate_smc_draft_decode_kv_indices[(gamma, padded_bs)](
            req_pool_indices,
            req_to_token,
            base_seq_lens,
            kv_indices,
            kv_indptr,
            raw_bs,
            req_to_token.shape[1],
            kv_indices.shape[1],
            kv_indptr.shape[1],
            4,
            4,
        )

        expected_indptr = torch.tensor(
            [
                [0, 3, 7, 8, 9],
                [0, 4, 9, 10, 11],
                [0, 5, 11, 12, 13],
            ],
            dtype=torch.int32,
            device=device,
        )
        self.assertTrue(torch.equal(kv_indptr, expected_indptr))

        expected_rows = [
            [10, 11, 12, 20, 21, 22, 23, 7, 7],
            [10, 11, 12, 100, 20, 21, 22, 23, 200, 7, 7],
            [10, 11, 12, 100, 101, 20, 21, 22, 23, 200, 201, 7, 7],
        ]
        for step, expected in enumerate(expected_rows):
            actual = kv_indices[step, : len(expected)].cpu()
            self.assertTrue(
                torch.equal(actual, torch.tensor(expected, dtype=torch.int32)),
                msg=f"step={step} actual={actual.tolist()} expected={expected}",
            )
