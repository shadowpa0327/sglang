"""SMC speculative decoding worker (V2 overlap scheduler).

SMC accepts ALL draft tokens (no rejection) and computes log-probability
differences between target and draft models. Uses topk=1 (linear chain).
"""

import contextlib
import logging
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import fill_new_verified_id
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
)
from sglang.srt.speculative.smc_info_v2 import SmcLogprobDiff, compute_smc_logprob_diff
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    fast_topk,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.speculative.standalone_worker_v2 import (
    StandaloneDraftWorker,
    StandaloneWorkerV2,
    _get_plan_stream,
)
from sglang.srt.utils import empty_context, get_available_gpu_memory, is_cuda

logger = logging.getLogger(__name__)


class SmcDraftWorker(StandaloneDraftWorker):
    """Draft worker for SMC that also returns per-step draft logprobs."""

    def init_cuda_graphs(self):
        """Override to use SmcDraftCudaGraphRunner for draft phase."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None
        self.smc_cuda_graph_runner = None

        if self.server_args.disable_cuda_graph:
            return

        if self.server_args.model_impl == "mindspore":
            return

        # Capture SMC draft CUDA graph (replaces EAGLE draft graph)
        if self.speculative_num_steps > 1:
            from sglang.srt.speculative.smc_draft_cuda_graph_runner import (
                SmcDraftCudaGraphRunner,
            )

            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture SMC draft cuda graph begin. avail mem={before_mem:.2f} GB"
            )
            self.smc_cuda_graph_runner = SmcDraftCudaGraphRunner(self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture SMC draft cuda graph end. Time: {time.perf_counter() - tic:.2f}s. "
                f"mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # Capture draft extend CUDA graph (same conditions as parent)
        from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
            EAGLEDraftExtendNpuGraphRunner,
        )
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLAMultiStepDraftBackend,
        )
        from sglang.srt.utils import is_hip as _is_hip
        from sglang.srt.utils import is_npu as _is_npu

        _is_npu_val = _is_npu()
        _is_hip_val = _is_hip()

        supports_hip_aiter_draft_extend_graph = False
        if _is_hip_val:
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        supports_cuda_draft_extend_graph = is_cuda() and (
            isinstance(self.draft_attn_backend, TritonMultiStepDraftBackend)
            or isinstance(self.draft_attn_backend, TRTLLMMLAMultiStepDraftBackend)
        )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
        }

        if self.draft_extend_attn_backend and (
            _is_npu_val
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time: {time.perf_counter() - tic:.2f}s. "
                f"mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    def smc_draft_forward(self, forward_batch: ForwardBatch):
        """Multi-step draft forward that also collects per-step draft logprobs.

        Same structure as EagleDraftWorker.draft_forward but appends log(topk_p)
        at each step before select_top_k_tokens consumes it.

        Returns:
            (parent_list, top_scores_index, draft_tokens, draft_logprobs)
            where draft_logprobs is (bs, spec_steps) log-probabilities of each
            draft token under the draft model.
        """
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "smc_draft_forward: NaN in initial topk_p")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        step_draft_logprobs: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            # Collect draft logprob BEFORE select_top_k_tokens.
            # With topk=1, topk_p shape (bs, 1): the prob of the token about to be selected.
            step_draft_logprobs.append(
                torch.log(topk_p.clamp(min=1e-10))
            )  # (bs, topk) where topk=1

            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # Last step: no forward needed
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(
                logits_output.next_token_logits, f"smc_draft_forward step {i}"
            )
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"smc_draft_forward step {i}: topk_index OOB",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results (same as draft_forward)
        score_list = torch.cat(score_list, dim=1).flatten(1)
        ss_token_list = torch.cat(token_list, dim=1)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "smc_draft_forward: top_scores_index OOB",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        # Stack draft logprobs and reorder to match top_scores_index ordering
        draft_logprobs = torch.cat(step_draft_logprobs, dim=1)  # (bs, spec_steps)
        draft_logprobs = torch.gather(
            draft_logprobs, 1, top_scores_index
        )  # (bs, num_draft_tokens - 1)

        return parent_list, top_scores_index, draft_tokens, draft_logprobs

    def smc_draft(self, model_worker_batch: ModelWorkerBatch):
        """SMC draft phase: propose tokens AND return draft logprobs.

        Returns:
            (EagleVerifyInput, draft_logprobs) where draft_logprobs is (bs, spec_steps)
            or (EagleVerifyInput, None) for idle batches.
        """
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            # Pass the smc_cuda_graph_runner for can_run check
            self.smc_cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft with logprob collection
        if can_cuda_graph and self.smc_cuda_graph_runner:
            parent_list, top_scores_index, draft_tokens, draft_logprobs = (
                self.smc_cuda_graph_runner.replay(forward_batch)
            )
        else:
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens, draft_logprobs = (
                self.smc_draft_forward(forward_batch)
            )

        if model_worker_batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            ), None

        # Build tree mask (trivial chain with topk=1)
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        verify_input = EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

        return verify_input, draft_logprobs


class SmcWorkerV2(StandaloneWorkerV2):
    """SMC speculative decoding worker with V2 overlap scheduler."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments (same as StandaloneWorkerV2)
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Create SMC draft worker (instead of StandaloneDraftWorker)
        self._draft_worker = SmcDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        # Capture verify post-processing CUDA graph
        self.verify_cuda_graph_runner = None
        if not server_args.disable_cuda_graph and is_cuda():
            self._init_verify_cuda_graph()

    def _init_verify_cuda_graph(self):
        from sglang.srt.model_executor.cuda_graph_runner import (
            get_batch_sizes_to_capture,
        )
        from sglang.srt.speculative.smc_verify_cuda_graph_runner import (
            SmcVerifyCudaGraphRunner,
        )

        capture_bs, _ = get_batch_sizes_to_capture(
            self.target_worker.model_runner
        )
        vocab_size = self.target_worker.model_runner.model_config.vocab_size
        logits_dtype = self.target_worker.model_runner.model_config.dtype

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture SMC verify cuda graph begin. "
            f"bs={capture_bs} vocab={vocab_size} dtype={logits_dtype} "
            f"avail mem={before_mem:.2f} GB"
        )
        try:
            self.verify_cuda_graph_runner = SmcVerifyCudaGraphRunner(
                spec_steps=self.speculative_num_steps,
                draft_token_num=self.speculative_num_draft_tokens,
                vocab_size=vocab_size,
                device=self.device,
                capture_bs=capture_bs,
                disable_padding=self.server_args.disable_cuda_graph_padding,
                logits_dtype=logits_dtype,
            )
        except Exception as e:
            logger.warning(
                f"SMC verify cuda graph capture failed, falling back to eager: {e}"
            )
            self.verify_cuda_graph_runner = None
            return
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture SMC verify cuda graph end. Time: {time.perf_counter() - tic:.2f}s. "
            f"bs={sorted(self.verify_cuda_graph_runner.graphs.keys())} "
            f"mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Prefill path: identical to EAGLE/STANDALONE
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        model_worker_batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:
            # Decode path: 4-step SMC flow
            if model_worker_batch.spec_info is None:
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.hidden_size,
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )

            # Step 1: Draft with logprob collection
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                verify_input, draft_logprobs = self.draft_worker.smc_draft(
                    model_worker_batch
                )

            assert verify_input.is_verify_input()
            model_worker_batch.spec_info = verify_input

            # Step 2: Verify (no rejection)
            batch_output = self.smc_verify(model_worker_batch, draft_logprobs)

            # Step 3: Draft extend (reuse from parent)
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.draft_worker._draft_extend_for_decode(
                    model_worker_batch, batch_output
                )

            return batch_output

    def smc_verify(
        self,
        batch: ModelWorkerBatch,
        draft_logprobs: Optional[torch.Tensor],
    ) -> GenerationBatchResult:
        """Verify phase for SMC: accept ALL draft tokens, compute logprob diff.

        Mirrors EAGLEWorkerV2.verify() but replaces rejection sampling with
        unconditional acceptance.
        """
        # Record stream for seq_lens (same as EAGLE)
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)
        device = self.device

        # Prepare for target verify (same as EAGLE)
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Correct buffers due to overlap plan (same as EAGLE)
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Run target verify (same as EAGLE)
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # --- CORE CHANGE: No rejection — accept all tokens ---
        maybe_detect_nan(logits_output.next_token_logits, "smc_verify: target logits")

        draft_token_num = verify_input.draft_token_num
        spec_steps = verify_input.spec_steps

        use_verify_graph = (
            not batch.forward_mode.is_idle()
            and self.verify_cuda_graph_runner is not None
            and self.verify_cuda_graph_runner.can_run(bs)
        )

        if not batch.forward_mode.is_idle():
            if use_verify_graph:
                # ---- CUDA-graph path for accept-all + logprob diff ----
                temperatures = batch.sampling_info.temperatures  # (bs, 1)
                (
                    predict,
                    accept_length,
                    accept_index,
                    verified_id,
                    logprob_diff_sum,
                ) = self.verify_cuda_graph_runner.replay(
                    target_logits=logits_output.next_token_logits,
                    draft_token=verify_input.draft_token,
                    temperatures=temperatures,
                    draft_logprobs=(
                        draft_logprobs
                        if draft_logprobs is not None
                        else torch.zeros(bs, spec_steps, device=device)
                    ),
                    bs=bs,
                )
            else:
                # ---- Fallback: plain-tensor path ----
                accept_length = torch.full(
                    (bs,), spec_steps + 1, dtype=torch.int32, device=device
                )

                draft_reshaped = verify_input.draft_token.reshape(bs, draft_token_num)
                target_logits_3d = logits_output.next_token_logits.reshape(
                    bs, draft_token_num, -1
                )

                bonus = torch.argmax(target_logits_3d[:, -1, :], dim=-1).to(
                    torch.int32
                )

                predict = torch.zeros(
                    bs * draft_token_num, dtype=torch.int32, device=device
                )
                predict_view = predict.reshape(bs, draft_token_num)
                predict_view[:, :spec_steps] = draft_reshaped[:, 1:].to(torch.int32)
                predict_view[:, spec_steps] = bonus

                accept_index = (
                    torch.arange(spec_steps + 1, device=device, dtype=torch.int32)
                    .unsqueeze(0)
                    .expand(bs, -1)
                    .contiguous()
                )

                # verified_id
                all_verified_id = predict[accept_index]
                verified_id = torch.empty_like(accept_length, dtype=torch.int32)
                fill_new_verified_id[(bs,)](
                    all_verified_id,
                    accept_length,
                    verified_id,
                    self.speculative_num_draft_tokens,
                )
                logprob_diff_sum = None
        else:
            accept_length = torch.empty(0, dtype=torch.int32, device=device)
            predict = torch.empty(0, dtype=torch.int32, device=device)
            accept_index = torch.empty(0, dtype=torch.int32, device=device)
            verified_id = torch.empty((0,), device=device, dtype=torch.int32)
            logprob_diff_sum = None

        new_seq_lens = batch.seq_lens + accept_length

        # Record verify_done event (same as EAGLE)
        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        # Compute logprobs if requested (same as EAGLE)
        if batch.return_logprob and not batch.forward_mode.is_idle():
            self._compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index
            )

        # Step 4: Compute / store SMC logprob diff
        if not batch.forward_mode.is_idle():
            if logprob_diff_sum is None and draft_logprobs is not None:
                # Fallback path: compute diff without graph
                temperatures = batch.sampling_info.temperatures
                smc_diff = compute_smc_logprob_diff(
                    target_logits=logits_output.next_token_logits,
                    draft_token=verify_input.draft_token,
                    draft_logprobs=draft_logprobs,
                    spec_steps=spec_steps,
                    draft_token_num=draft_token_num,
                    temperatures=temperatures,
                )
                logits_output.smc_logprob_diff = smc_diff
            elif logprob_diff_sum is not None:
                logits_output.smc_logprob_diff = SmcLogprobDiff(
                    draft_logprobs=draft_logprobs,
                    target_logprobs=None,  # already consumed in graph
                    logprob_diff_sum=logprob_diff_sum,
                )

        # Construct the next draft input (same as EAGLE)
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
        )

    def _compute_spec_v2_logprobs(
        self,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
    ):
        """Compute logprobs for accepted tokens on GPU in the forward stream.

        Identical to EAGLEWorkerV2._compute_spec_v2_logprobs.
        """
        bs = len(batch.seq_lens)
        max_accept = self.speculative_num_steps + 1
        device = predict.device

        flat_accept_idx = accept_index.long().reshape(-1)
        gathered_logits = logits_output.next_token_logits[flat_accept_idx]

        if (
            batch.sampling_info.is_all_greedy
            or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get()
        ):
            gathered_logprobs = torch.nn.functional.log_softmax(
                gathered_logits, dim=-1
            )
        else:
            temperatures = torch.repeat_interleave(
                batch.sampling_info.temperatures,
                max_accept,
                dim=0,
            )
            gathered_logprobs = torch.nn.functional.log_softmax(
                gathered_logits / temperatures, dim=-1
            )
        gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

        accepted_token_ids = predict[flat_accept_idx]
        token_logprobs = gathered_logprobs[
            torch.arange(bs * max_accept, device=device),
            accepted_token_ids.long(),
        ]
        logits_output.next_token_logprobs = token_logprobs.reshape(bs, max_accept)

        if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
            top_logprobs_nums_expanded = [
                num for num in batch.top_logprobs_nums for _ in range(max_accept)
            ]
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
                gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True
            )

        if batch.token_ids_logprobs and any(
            x is not None for x in batch.token_ids_logprobs
        ):
            token_ids_logprobs_expanded = [
                ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
            ]
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True
            )
