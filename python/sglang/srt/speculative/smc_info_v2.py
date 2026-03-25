"""SMC speculative decoding data structures and logprob-diff computation.

SMC (Sequential Monte Carlo) accepts ALL draft tokens (no rejection) and computes
the log-probability difference between target and draft models per token.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

# Type aliases — SMC reuses EAGLE data structures for FutureMap/overlap compatibility.
SmcDraftInput = EagleDraftInput
SmcVerifyInput = EagleVerifyInput


@dataclass
class SmcLogprobDiff:
    """Per-request logprob difference information for SMC."""

    draft_logprobs: torch.Tensor  # (bs, spec_steps) — draft model logprobs for r draft tokens
    target_logprobs: torch.Tensor  # (bs, spec_steps) — target model logprobs for r draft tokens
    logprob_diff_sum: torch.Tensor  # (bs,) — sum of (target_lp - draft_lp) per request


def compute_smc_logprob_diff(
    target_logits: torch.Tensor,
    draft_token: torch.Tensor,
    draft_logprobs: torch.Tensor,
    spec_steps: int,
    draft_token_num: int,
    temperatures: torch.Tensor,
) -> SmcLogprobDiff:
    """Compute per-request logprob diff between target and draft models.

    Args:
        target_logits: (bs * draft_token_num, vocab_size) from target forward.
        draft_token: (bs * draft_token_num,) = [verified_id, d1, ..., dr] per request.
        draft_logprobs: (bs, spec_steps) from draft phase.
        spec_steps: number of draft tokens (r).
        draft_token_num: spec_steps + 1 (includes verified_id root).
        temperatures: (bs, 1) sampling temperatures.

    Returns:
        SmcLogprobDiff with per-request diff information.
    """
    bs = target_logits.shape[0] // draft_token_num
    device = target_logits.device

    # Reshape to (bs, draft_token_num, vocab)
    target_logits_3d = target_logits.reshape(bs, draft_token_num, -1)

    # Apply temperature and compute log-softmax
    expanded_temp = temperatures.unsqueeze(1).expand(-1, draft_token_num, -1)  # (bs, draft_token_num, 1)
    target_log_probs = F.log_softmax(target_logits_3d / expanded_temp, dim=-1)

    # The r draft tokens are at positions 1..spec_steps in draft_token
    draft_token_reshaped = draft_token.reshape(bs, draft_token_num)
    r_draft_tokens = draft_token_reshaped[:, 1 : spec_steps + 1].long()  # (bs, spec_steps)

    # Target logprobs at positions 0..spec_steps-1 (logits predicting the next token)
    # Position j predicts what comes at position j+1, i.e., draft_token[j+1]
    target_lp = target_log_probs[:, :spec_steps, :]  # (bs, spec_steps, vocab)
    target_lp_gathered = target_lp.gather(2, r_draft_tokens.unsqueeze(-1)).squeeze(
        -1
    )  # (bs, spec_steps)

    # Logprob diff: target - draft. Bonus token contributes 0 (not included here).
    diff = target_lp_gathered - draft_logprobs
    diff_sum = diff.sum(dim=1)  # (bs,)

    return SmcLogprobDiff(
        draft_logprobs=draft_logprobs,
        target_logprobs=target_lp_gathered,
        logprob_diff_sum=diff_sum,
    )
