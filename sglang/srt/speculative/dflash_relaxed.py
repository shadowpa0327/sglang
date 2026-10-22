"""Relaxed greedy verification for DFLASH (research patch, not lossless).

A drafted token y at chain slot j is accepted when it equals the target argmax,
or when the target ranks it within its top `k` and within `margin` logits of
the argmax (abdelfattah-lab/sharar, engine/accept.py). Acceptance stays the
cumulative prefix and the bonus stays the target argmax at the first rejected
slot, so the rule is applied by rewriting the greedy `target_predict` rows
before the existing accept kernels run.

Per request: `custom_params: {"accept_top_k": K, "accept_margin": M}`.
Server default: SGLANG_DFLASH_RELAXED_TOP_K / SGLANG_DFLASH_RELAXED_MARGIN
(top-k <= 1 is exact). SGLANG_DFLASH_RELAXED_LOG=<path> appends one JSON line
per verify step that relaxed-accepted a token (rank 0 only), and
SGLANG_DFLASH_RELAXED_STATS=<path> keeps cumulative counters per rule.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import torch

_DEFAULT_K = int(os.environ.get("SGLANG_DFLASH_RELAXED_TOP_K", "0"))
_DEFAULT_M = float(os.environ.get("SGLANG_DFLASH_RELAXED_MARGIN", "0"))
_LOG_PATH = os.environ.get("SGLANG_DFLASH_RELAXED_LOG") or None
_STATS_PATH = os.environ.get("SGLANG_DFLASH_RELAXED_STATS") or None


def request_rules(reqs) -> Optional[tuple[list[int], list[float]]]:
    """Per-row (top-k, margin), or None when every row verifies exactly."""
    ks, ms = [], []
    for r in reqs:
        cp = getattr(r.sampling_params, "custom_params", None) or {}
        ks.append(int(cp.get("accept_top_k", _DEFAULT_K)))
        ms.append(float(cp.get("accept_margin", _DEFAULT_M)))
    if all(k <= 1 for k in ks):
        return None
    return ks, ms


def relaxed_mask(logits: torch.Tensor, drafts: torch.Tensor, ks: torch.Tensor,
                 ms: torch.Tensor) -> torch.Tensor:
    """[b, n] bool: drafts within each row's top-k and margin of the argmax.

    logits [b, n, V] for the drafted slots, drafts [b, n], ks/ms [b].
    """
    ly = logits.gather(-1, drafts.unsqueeze(-1)).squeeze(-1).float()
    lmax = logits.max(-1).values.float()
    rank = (logits > ly.unsqueeze(-1)).sum(-1)
    return (rank < ks.unsqueeze(1)) & ((lmax - ly) <= ms.unsqueeze(1))


def apply_relaxed(target_predict: torch.Tensor, candidates: torch.Tensor,
                  next_token_logits: torch.Tensor, rules) -> tuple[torch.Tensor, torch.Tensor]:
    """Rewrite target_predict[:, :-1] in place to the draft where relaxed.

    candidates[:, 1:] are the drafts checked against target_predict[:, :-1].
    Returns (exact, relaxed) masks [b, block-1], computed before the rewrite.
    """
    bs, block = target_predict.shape
    dev = target_predict.device
    ks = torch.tensor(rules[0], device=dev, dtype=torch.int64)
    ms = torch.tensor(rules[1], device=dev, dtype=torch.float32)
    logits = next_token_logits.view(bs, block, -1)[:, : block - 1]
    drafts = candidates[:, 1:]
    exact = drafts == target_predict[:, :-1]
    relaxed = relaxed_mask(logits, drafts, ks, ms) & ~exact
    target_predict[:, :-1] = torch.where(relaxed, drafts, target_predict[:, :-1])
    return exact, relaxed


class RelaxedStats:
    """Cumulative counters per rule and the per-step relaxed-accept log (rank 0)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled and (_LOG_PATH is not None or _STATS_PATH is not None)
        self.stats: dict[str, dict] = {}
        self.steps = 0

    def record(self, reqs, rules, exact, relaxed, accept_len, argmax_before, drafts) -> None:
        if not self.enabled:
            return
        block1 = exact.shape[1]
        within = torch.arange(block1, device=exact.device).unsqueeze(0) < accept_len.unsqueeze(1)
        rx = relaxed & within
        acc = accept_len.tolist()
        rx_counts = rx.sum(1).tolist()
        for i in range(len(reqs)):
            key = f"{rules[0][i]}:{rules[1][i]:g}"
            s = self.stats.setdefault(key, {"verify_steps": 0, "accepted_drafts": 0,
                                            "relaxed_accepts": 0,
                                            "relaxed_by_slot": [0] * block1})
            s["verify_steps"] += 1
            s["accepted_drafts"] += int(acc[i])
            s["relaxed_accepts"] += int(rx_counts[i])
        if any(rx_counts):
            idx = rx.nonzero().tolist()
            am = argmax_before.tolist()
            dr = drafts.tolist()
            lines = []
            for i, j in idx:
                key = f"{rules[0][i]}:{rules[1][i]:g}"
                self.stats[key]["relaxed_by_slot"][j] += 1
                if _LOG_PATH is not None:
                    r = reqs[i]
                    lines.append(json.dumps({
                        "rid": r.rid, "rule": key, "slot": j,
                        "out_len": len(r.output_ids), "accept_len": int(acc[i]),
                        "draft": int(dr[i][j]), "argmax": int(am[i][j]),
                    }))
            if lines:
                with open(_LOG_PATH, "a") as f:
                    f.write("\n".join(lines) + "\n")
        self.steps += 1
        if _STATS_PATH is not None and (any(rx_counts) or self.steps % 10 == 0):
            self.flush()

    def flush(self) -> None:
        if _STATS_PATH is None:
            return
        tmp = _STATS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"time": time.time(), "rules": self.stats}, f)
        os.replace(tmp, _STATS_PATH)
