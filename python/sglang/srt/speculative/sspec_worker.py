import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Union, Tuple

import numpy as np
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import broadcast_pyobj
from sglang.srt.speculative.eagle_utils import assign_draft_cache_locs
if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class SSPECWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.speculative_num_steps = server_args.speculative_num_steps
        self.page_size = 1 #FIXME Hard coded now.

        # Override context length with target model's context length
    def draft(self, batch: ScheduleBatch):
        #TODO
        # Parse args
        num_seqs = batch.batch_size()
        # Allocate cache locations
        out_cache_loc, token_to_kv_pool_state_backup = batch.alloc_token_slots(
            num_seqs * self.speculative_num_steps, backup_state=True
        )
        # Allocate 
        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            topk=1,
            speculative_num_steps=self.speculative_num_steps,
            page_size=1
        )
        # Set FlashInfer Metadata
        #TODO
        # Forward Multiple Steps
        for i in range(self.speculative_num_steps):
            #TODO
            pass



    def forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int, int]:

        if batch.forward_mode.is_decode():
            pass
        else: # IDLE/Extend
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids, _ = (
                self.target_worker.forward_batch_generation(model_worker_batch)
            )
            return logits_output, next_token_ids, model_worker_batch.bid, 0, False