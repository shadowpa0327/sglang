import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Union, Tuple

import numpy as np
import torch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode, CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import broadcast_pyobj, fast_topk
from sglang.srt.speculative.eagle_utils import assign_draft_cache_locs, EagleDraftInput, EagleVerifyInput, select_top_k_tokens
if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
)

logger = logging.getLogger(__name__)


class SSPECWorker:
    """
    SSPEC (Simple Speculative) Worker implementation for speculative decoding.
    
    This is a simplified speculative decoding approach that uses the same model
    for both drafting and verification phases. Unlike EAGLE which uses a separate
    draft model and tree-based verification, SSPEC performs sequential speculative
    decoding with the target model.
    
    Key characteristics:
    - Uses the same model_runner for both draft and target phases
    - Performs sequential token verification (not tree-based)
    - Supports only top-k=1 and page_size=1 for simplicity
    - Uses FlashInfer attention backend
    - Reuses EAGLE's spec_info structures for consistency
    
    The workflow is:
    1. Draft Phase: Generate speculative_num_steps tokens using the target model
    2. Verify Phase: Run target model on all draft tokens and verify sequentially
    3. Accept tokens until first mismatch, add bonus token if all accepted
    """
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.speculative_num_steps = server_args.speculative_num_steps
        self.page_size = 1 
        self.topk = 1
        self.init_attention_backend()

    def init_attention_backend(self):
        if self.server_args.attention_backend == "flashinfer":
            self.sparse_attn_backend = FlashInferAttnBackend(
                self.model_runner,
                skip_prefill=False,
            )
        else:
            raise NotImplementedError(f"SSPEC only support flashinfer backend. {self.server_args.attention_backend} do not support")

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Draft tokens using the same model as the target model."""
        num_seqs = batch.batch_size()
        if num_seqs == 0:
            # Return empty verify input
            return EagleVerifyInput.create(
                torch.tensor([], dtype=torch.long),
                [], [], [],
                torch.tensor([], dtype=torch.long),
                0, self.topk, self.speculative_num_steps, 0
            )
            
        # Get spec_info from batch (similar to EAGLE)
        spec_info = batch.spec_info
        
        # Set up cache allocation similar to EAGLE
        # For SSPEC with topk=1, we allocate cache for all speculative steps
        extend_num_tokens = num_seqs * self.topk * self.speculative_num_steps
        
        # Allocate cache locations for draft tokens
        out_cache_loc, token_to_kv_pool_state_backup = batch.alloc_token_slots(
            extend_num_tokens, backup_state=True
        )
        
        # Set up batch for drafting (similar to EAGLE)
        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        
        # Get forward batch
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
                
        # Run forward steps (simplified version of EAGLE's draft_forward)
        score_list, token_list, parents_list = self.draft_forward(forward_batch)
        
        # Restore KV cache state
        self.target_worker.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)
        
        # Create EagleVerifyInput using EAGLE's structure
        ret = EagleVerifyInput.create(
            spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.server_args.speculative_num_draft_tokens,
        )
        return ret

    def draft_forward(self, forward_batch: ForwardBatch) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass for drafting - simplified version of EAGLE's approach."""
        spec_info = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        num_seqs = forward_batch.batch_size
        
        # Get initial tokens from spec_info (last verified tokens)
        if spec_info.verified_id is not None:
            current_tokens = torch.tensor(spec_info.verified_id, device=forward_batch.seq_lens.device)
        else:
            # Fallback: get last token from each sequence
            current_tokens = torch.zeros(num_seqs, dtype=torch.long, device=forward_batch.seq_lens.device)
        
        positions = spec_info.positions.clone()
        
        # For SSPEC, we create simple linear tree structure (no branching)
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        
        # Initialize hidden states for the first step
        hidden_states = spec_info.hidden_states
        scores = None
        
        # Generate speculative tokens step by step
        for step in range(self.speculative_num_steps):
            # Set up input for this step
            forward_batch.input_ids = current_tokens
            forward_batch.positions = positions
            
            # Set cache location for this step
            step_cache_loc = out_cache_loc[step * num_seqs:(step + 1) * num_seqs]
            forward_batch.out_cache_loc = step_cache_loc
            
            # Initialize sparse attention backend
            # backup the spec_info
            forward_batch.spec_info = None
            self.sparse_attn_backend.init_forward_metadata(forward_batch)
            forward_batch.attn_backend = self.sparse_attn_backend
            # restore the spec_info
            forward_batch.spec_info = spec_info

            # Forward pass
            logits_output = self.model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            
            # Sample next tokens using top-k selection (even though topk=1)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            
            # Use EAGLE's select_top_k_tokens function for consistency
            current_tokens, hidden_states, scores, tree_info = select_top_k_tokens(
                step, topk_p, topk_index, hidden_states, scores, self.topk
            )
            
            # Extract tree info components
            step_scores, step_tokens, step_parents = tree_info
            
            # Store tree info for EAGLE compatibility
            score_list.append(step_scores)
            token_list.append(step_tokens)
            parents_list.append(step_parents)
            
            # Update positions for next iteration
            positions += 1
        
        return score_list, token_list, parents_list

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput) -> Tuple[LogitsProcessorOutput, List[int], List[int], bool]:
        """Verify draft tokens against target model using EAGLE's verify structure."""
        # Use EAGLE's prepare_for_verify method
        spec_info.prepare_for_verify(batch, self.page_size)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        model_worker_batch = batch.get_model_worker_batch()
        
        # Forward
        logits_output, _, can_run_cuda_graph = (
            self.target_worker.forward_batch_generation(
                model_worker_batch, skip_sample=True
            )
        )
        
        # Use EAGLE's verify method
        spec_info.hidden_states = logits_output.hidden_states
        res = spec_info.verify(
            batch,
            logits_output,
            self.target_worker.token_to_kv_pool_allocator,
            self.page_size,
            None,  # No vocab_mask for SSPEC
        )
        
        # Extract results in SSPEC format
        verified_tokens = res.verified_id.tolist() if res.verified_id is not None else []
        draft_accept_lengths = res.accept_length_per_req_cpu
        
        # Prepare the batch for the next draft forwards (like EAGLE)
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = res.draft_input
        
        return logits_output, verified_tokens, draft_accept_lengths, can_run_cuda_graph

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: List[int],
    ):
        """Set up spec_info for next draft - using EAGLE's EagleDraftInput."""
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
        )
        #batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

    def forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int, int, bool]:
        print("SSPEC_WORKER: forward_batch_speculative_generation")
        from fpdb import ForkedPdb
        ForkedPdb().set_trace()

        if batch.forward_mode.is_decode():
            print("SSPEC_WORKER: forward_batch_speculative_generation (draft)")
            from fpdb import ForkedPdb
            ForkedPdb().set_trace()
            # Run Draft then Verify (similar to EAGLE pattern)
            spec_info = self.draft(batch)
            print("SSPEC_WORKER: forward_batch_speculative_generation (verify)")
            from fpdb import ForkedPdb
            ForkedPdb().set_trace()
            logits_output, verified_tokens, draft_accept_lengths, can_run_cuda_graph = self.verify(batch, spec_info)
            
            # Calculate total accepted draft tokens (exclude bonus tokens)
            total_accepted = sum(draft_accept_lengths)
            
            # Set up spec_info for next iteration (if there are verified tokens)
            if batch.spec_info.verified_id is not None:
                self.forward_draft_extend(batch, logits_output.hidden_states, verified_tokens)
            
            model_worker_batch = batch.get_model_worker_batch()
            print("SSPEC_WORKER: forward_batch_speculative_generation (decode)")
            from fpdb import ForkedPdb
            ForkedPdb().set_trace()
            return logits_output, verified_tokens, model_worker_batch.bid, total_accepted, can_run_cuda_graph
        else: # IDLE/Extend
            # Forward with the target model - similar to EAGLE
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            logits_output, next_token_ids, _ = (
                self.target_worker.forward_batch_generation(model_worker_batch)
            )
            
            # Set up spec_info for next draft
            self.forward_draft_extend(batch, logits_output.hidden_states, next_token_ids)
            
            print("SSPEC_WORKER: forward_batch_speculative_generation (idle/extend)")
            from fpdb import ForkedPdb
            ForkedPdb().set_trace()
            return logits_output, next_token_ids, model_worker_batch.bid, 0, False