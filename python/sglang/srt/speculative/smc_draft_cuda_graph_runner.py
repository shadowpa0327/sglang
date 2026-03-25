"""CUDA graph runner for SMC draft phase.

Extends EAGLEDraftCudaGraphRunner to capture `smc_draft_forward` which returns
4 values (parent_list, top_scores_index, draft_tokens, draft_logprobs) instead
of the standard 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    get_global_graph_memory_pool,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput

if TYPE_CHECKING:
    from sglang.srt.speculative.smc_worker_v2 import SmcDraftWorker


class SmcDraftCudaGraphRunner(EAGLEDraftCudaGraphRunner):
    """CUDA graph runner for SMC multi-step draft with logprob collection.

    Inherits all buffer setup, input copying, attention backend, and replay
    machinery from EAGLEDraftCudaGraphRunner. Only overrides capture and
    output post-processing to handle the extra draft_logprobs tensor.
    """

    def capture_one_batch_size(
        self, num_seqs: int, forward: Callable, stream_idx: int = 0
    ):
        buffers = self.buffers
        graph = self._create_graph()
        stream = self.stream
        num_tokens = num_seqs * self.num_tokens_per_bs

        # Graph inputs (identical to parent)
        req_pool_indices = buffers.req_pool_indices[:num_seqs]
        seq_lens = buffers.seq_lens[:num_seqs]
        seq_lens_cpu = buffers.seq_lens_cpu[:num_seqs]
        extend_seq_lens = buffers.extend_seq_lens[:num_seqs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:num_seqs]
        out_cache_loc = buffers.out_cache_loc[
            : num_tokens * self.speculative_num_steps
        ]
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        hidden_states = buffers.hidden_states[:num_seqs]
        topk_p = buffers.topk_p[:num_seqs]
        topk_index = buffers.topk_index[:num_seqs]

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_dp_buffer_len = num_tokens * self.dp_size
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_dp_buffer_len = num_tokens
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
        else:
            global_num_tokens = None
            global_dp_buffer_len = None
            global_num_tokens_for_logprob = None

        spec_info = EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        # Forward batch
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=num_seqs,
            input_ids=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=(
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            ),
        )

        # Attention backend
        self.model_runner.draft_attn_backend.init_forward_metadata_capture_cuda_graph(
            forward_batch
        )

        # Run and capture — KEY CHANGE: call smc_draft_forward instead of draft_forward
        def run_once():
            forward_batch.dp_local_start_pos = (
                forward_batch.dp_local_num_tokens
            ) = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.eagle_worker.smc_draft_forward(forward_batch)

            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        self.deepep_adapter.capture(is_extend_in_batch=False)

        self._capture_init(run_once)

        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        # Handle 4 output tensors instead of 3
        parent_list, top_scores_index, draft_tokens, draft_logprobs = (
            t[:raw_bs] for t in out
        )
        return parent_list, top_scores_index, draft_tokens, draft_logprobs
