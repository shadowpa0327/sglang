"""CUDA graph runner for the SMC verify post-processing.

Captures the accept-all, verified-id extraction, and logprob-diff operations
that follow the target-model forward inside ``smc_verify()``.  All shapes are
fixed for a given ``(bs, spec_steps, draft_token_num, vocab_size)`` so a
CUDA graph can replay them without kernel-launch overhead.

Memory: only ONE set of buffers is allocated at ``max_bs``.  Every captured
graph writes into the same buffers (safe because only one graph replays at a
time per decode step).
"""

from __future__ import annotations

import bisect
import logging
from typing import Tuple

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.cuda_graph_runner import (
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
)

logger = logging.getLogger(__name__)


class SmcVerifyCudaGraphRunner:
    """Captures post-target-forward ops of ``smc_verify`` as CUDA graphs.

    For each captured batch-size *bs* the graph replays:

    1. ``argmax`` on the last-position target logit  →  bonus token
    2. Construct ``predict`` (draft tokens + bonus)
    3. Extract ``verified_id`` (= bonus, the last accepted token)
    4. ``log_softmax`` + ``gather`` for target logprobs of draft tokens
    """

    def __init__(
        self,
        spec_steps: int,
        draft_token_num: int,
        vocab_size: int,
        device: torch.device,
        capture_bs: list[int],
        disable_padding: bool = False,
        logits_dtype: torch.dtype = torch.float32,
    ):
        self.spec_steps = spec_steps
        self.draft_token_num = draft_token_num
        self.vocab_size = vocab_size
        self.device = device
        self.capture_bs = sorted(capture_bs)
        self.max_bs = max(self.capture_bs)
        self.disable_padding = disable_padding
        self.logits_dtype = logits_dtype

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

        # Single set of pre-allocated buffers at max_bs (shared across graphs)
        max_total = self.max_bs * draft_token_num
        d = device

        # ── inputs (caller copies into before replay) ──
        self.target_logits = torch.zeros(
            max_total, vocab_size, dtype=logits_dtype, device=d
        )
        self.draft_token = torch.zeros(max_total, dtype=torch.int64, device=d)
        self.temperatures = torch.ones(self.max_bs, 1, dtype=torch.float32, device=d)

        # ── outputs (read after replay) ──
        self.predict = torch.zeros(max_total, dtype=torch.int32, device=d)
        self.accept_length = torch.full(
            (self.max_bs,), spec_steps + 1, dtype=torch.int32, device=d
        )
        self.accept_index = (
            torch.arange(spec_steps + 1, device=d, dtype=torch.int32)
            .unsqueeze(0)
            .expand(self.max_bs, -1)
            .contiguous()
        )
        self.verified_id = torch.zeros(self.max_bs, dtype=torch.int32, device=d)
        self.target_lp = torch.zeros(
            self.max_bs, spec_steps, dtype=torch.float32, device=d
        )

        try:
            with model_capture_mode():
                self._capture_all()
        except RuntimeError as e:
            raise RuntimeError(
                f"SmcVerifyCudaGraphRunner capture failed: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    # capture
    # ------------------------------------------------------------------ #
    def _run_body(self, bs: int) -> None:
        """The actual ops — called during warmup AND capture.

        All tensors come from ``self.*`` (pre-allocated), so no allocation
        happens during graph capture.
        """
        dtn = self.draft_token_num
        ss = self.spec_steps
        total = bs * dtn

        # 1. bonus = argmax of last-position logits per request
        logits_3d = self.target_logits[:total].reshape(bs, dtn, -1)
        bonus = torch.argmax(logits_3d[:, -1, :], dim=-1).to(torch.int32)

        # 2. construct predict  [draft_1 … draft_r, bonus]  per request
        draft_2d = self.draft_token[:total].reshape(bs, dtn)
        pv = self.predict[:total].reshape(bs, dtn)
        pv[:, :ss] = draft_2d[:, 1:].to(torch.int32)
        pv[:, ss] = bonus

        # 3. verified_id = bonus (last accepted token per request)
        #    Since accept_length is always ss+1, the bonus is at column ss.
        self.verified_id[:bs].copy_(pv[:, ss])

        # 4. target logprob of each draft token
        temp_3d = self.temperatures[:bs].unsqueeze(1).expand(-1, dtn, -1)
        target_log_probs = F.log_softmax(
            logits_3d / temp_3d.to(logits_3d.dtype), dim=-1
        )
        r_tokens = draft_2d[:, 1 : ss + 1].long()  # (bs, ss)
        self.target_lp[:bs].copy_(
            target_log_probs[:, :ss, :]
            .gather(2, r_tokens.unsqueeze(-1))
            .squeeze(-1)
            .float()
        )

    def _capture_all(self) -> None:
        for bs in self.capture_bs:
            graph = torch.cuda.CUDAGraph()

            # warmup
            for _ in range(2):
                torch.cuda.synchronize()
                self._run_body(bs)

            # capture
            pool = get_global_graph_memory_pool()
            with torch.cuda.graph(graph, pool=pool):
                self._run_body(bs)
            set_global_graph_memory_pool(graph.pool())

            self.graphs[bs] = graph
            logger.debug(f"SmcVerifyCudaGraphRunner: captured bs={bs}")

    # ------------------------------------------------------------------ #
    # replay
    # ------------------------------------------------------------------ #
    def can_run(self, bs: int) -> bool:
        if self.disable_padding:
            return bs in self.graphs
        return 0 < bs <= self.max_bs

    def replay(
        self,
        target_logits: torch.Tensor,
        draft_token: torch.Tensor,
        temperatures: torch.Tensor,
        draft_logprobs: torch.Tensor,
        bs: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Replay the captured verify post-processing graph.

        Returns ``(predict, accept_length, accept_index, verified_id,
        logprob_diff_sum)`` — all sliced to the real *bs*.
        """
        # pick captured bs (with optional padding)
        if self.disable_padding:
            graph_bs = bs
        else:
            idx = bisect.bisect_left(self.capture_bs, bs)
            graph_bs = self.capture_bs[idx]

        total_real = bs * self.draft_token_num
        total_cap = graph_bs * self.draft_token_num

        # copy inputs into pre-allocated buffers
        self.target_logits[:total_real].copy_(target_logits[:total_real])
        self.draft_token[:total_real].copy_(draft_token[:total_real])
        self.temperatures[:bs].copy_(temperatures[:bs])
        if bs < graph_bs:
            # zero-pad so the padded rows don't produce NaN / garbage
            self.target_logits[total_real:total_cap].zero_()
            self.draft_token[total_real:total_cap].zero_()
            self.temperatures[bs:graph_bs].fill_(1.0)

        # replay
        self.graphs[graph_bs].replay()

        # compute logprob_diff_sum (needs draft_logprobs which lives outside the graph)
        target_lp = self.target_lp[:bs]
        logprob_diff_sum = (target_lp - draft_logprobs).sum(dim=1)

        return (
            self.predict[:total_real],
            self.accept_length[:bs],
            self.accept_index[:bs],
            self.verified_id[:bs],
            logprob_diff_sum,
        )
