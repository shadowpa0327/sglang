"""Relaxed greedy verification for DFLASH, and EAGLE / MTP chains (research patch, not lossless).

A drafted token y at chain slot j is accepted when it equals the target argmax,
or when the target ranks it within its top `k` and within `margin` logits of
the argmax (abdelfattah-lab/sharar, engine/accept.py). Acceptance stays the
cumulative prefix and the bonus stays the target argmax at the first rejected
slot, so the rule is applied by rewriting the greedy `target_predict` rows
before the existing accept kernels run.

Per request: `custom_params: {"accept_top_k": K, "accept_margin": M}`, plus two harm gates
that refuse a relaxed accept (the slot then verifies exactly): `"relax_code": false` inside a
fenced code block (an odd number of ``` tokens in the request's output so far), and
`"repeat_veto_ngram": n` when the drafted token would complete an n-gram the output already
contains (the start of a loop). Sampled rows only: `"relax_top": false` keeps the target's own
top token on the ordinary rejection test (only near misses get the plausible pass), which removes
the mode-sharpening an over-confident draft otherwise passes into the output. Sampled rows only:
`"accept_soft": s` keeps the coin of a plausible draft but relaxes it by e^s (accepted when
coin * q < e^s * p, instead of always): the output law then satisfies P(x) <= e^s * p(x) for
every token, s = 0 is exact sampling, and the rejection residual (p - q)+ needs no change.
Unset (s = inf) is the forced accept.
Server default: SGLANG_DFLASH_RELAXED_TOP_K / SGLANG_DFLASH_RELAXED_MARGIN / SGLANG_DFLASH_RELAXED_TOP
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
# server default for requests that do not set relax_top: SGLANG_DFLASH_RELAXED_TOP=0 keeps the target's top token on the
# ordinary test
_DEFAULT_TOP = os.environ.get("SGLANG_DFLASH_RELAXED_TOP", "1") != "0"
# server default for accept_soft (inf: a plausible draft's coin is zeroed, the forced accept)
_DEFAULT_SOFT = float(os.environ.get("SGLANG_DFLASH_RELAXED_SOFT", "inf"))


def request_rules(reqs) -> Optional[tuple]:
    """Per-row (top-k, margin, relax_code, repeat_veto_ngram, relax_top, accept_soft), or None when all rows are exact."""
    ks, ms, code_ok, rep_n, top_ok, soft = [], [], [], [], [], []
    for r in reqs:
        cp = getattr(r.sampling_params, "custom_params", None) or {}
        ks.append(int(cp.get("accept_top_k", _DEFAULT_K)))
        ms.append(float(cp.get("accept_margin", _DEFAULT_M)))
        code_ok.append(bool(cp.get("relax_code", True)))
        rep_n.append(int(cp.get("repeat_veto_ngram", 0)))
        top_ok.append(bool(cp.get("relax_top", _DEFAULT_TOP)))
        soft.append(float(cp.get("accept_soft", _DEFAULT_SOFT)))
    if all(k <= 1 for k in ks):
        return None
    return ks, ms, code_ok, rep_n, top_ok, soft


def gates_active(rules) -> bool:
    return rules is not None and (not all(rules[2]) or any(n > 0 for n in rules[3]))


class HarmGate:
    """Host-side per-request state for the code-block and repetition gates (every TP rank)."""

    def __init__(self, tokenizer_path: Optional[str]):
        self.tokenizer_path = tokenizer_path
        self._fence = None
        self.state: dict = {}
        self.step = 0

    def fence(self) -> set:
        if self._fence is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.tokenizer_path)
            toks = tok.convert_ids_to_tokens(list(range(len(tok))))
            self._fence = {i for i, t in enumerate(toks) if t is not None and "```" in t}
        return self._fence

    def _st(self, rid, n):
        st = self.state.get(rid)
        if st is None or st["n"] != n:
            st = {"n": n, "tail": [], "grams": set(), "fences": 0, "seen": self.step}
            self.state[rid] = st
        st["seen"] = self.step
        return st

    def filter(self, reqs, rules, candidates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """mask [b, n] (relaxed-eligible drafts) with the gated slots cleared."""
        fence = self.fence()
        anchors = candidates[:, 0].tolist()
        drafts = candidates[:, 1:].tolist()
        m = mask.tolist()
        for i, r in enumerate(reqs):
            code_ok, n = rules[2][i], rules[3][i]
            if code_ok and n <= 0:
                continue
            st = self._st(r.rid, n)
            fences = st["fences"] + (anchors[i] in fence)
            seq = st["tail"] + [anchors[i]]
            local = set()
            for j, y in enumerate(drafts[i]):
                if m[i][j]:
                    veto = (not code_ok) and fences % 2 == 1
                    if not veto and n > 0 and len(seq) >= n - 1:
                        g = hash(tuple(seq[len(seq) - (n - 1):] + [y]))
                        veto = g in st["grams"] or g in local
                    if veto:
                        m[i][j] = False
                if n > 0 and len(seq) >= n:
                    local.add(hash(tuple(seq[len(seq) - n:])))
                seq.append(y)
                fences += y in fence
        return torch.tensor(m, dtype=torch.bool, device=mask.device)

    def commit(self, reqs, rules, candidates: torch.Tensor, accept_len: torch.Tensor) -> None:
        """Append each gated row's anchor and accepted drafts to its history."""
        fence = self.fence()
        self.step += 1
        anchors = candidates[:, 0].tolist()
        drafts = candidates[:, 1:].tolist()
        acc = accept_len.tolist()
        for i, r in enumerate(reqs):
            code_ok, n = rules[2][i], rules[3][i]
            if code_ok and n <= 0:
                continue
            st = self._st(r.rid, n)
            for t in [anchors[i]] + drafts[i][: int(acc[i])]:
                st["tail"].append(t)
                st["fences"] += t in fence
                if n > 0 and len(st["tail"]) >= n:
                    st["grams"].add(hash(tuple(st["tail"][-n:])))
                if len(st["tail"]) > max(n, 1):
                    del st["tail"][0]
        if self.step % 200 == 0:
            stale = [k for k, v in self.state.items() if self.step - v["seen"] > 2000]
            for k in stale:
                del self.state[k]


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
                  next_token_logits: torch.Tensor, rules, gate: Optional["HarmGate"] = None,
                  reqs=None) -> tuple[torch.Tensor, torch.Tensor]:
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
    if gate is not None and gates_active(rules):
        relaxed = gate.filter(reqs, rules, candidates, relaxed)
    target_predict[:, :-1] = torch.where(relaxed, drafts, target_predict[:, :-1])
    return exact, relaxed


def _coin_scale(rules, dev) -> torch.Tensor:
    """[b, 1] factor for a plausible draft's coin: e^-s (accept_soft s), 0 for the forced accept (s = inf)."""
    soft = rules[5] if len(rules) > 5 else [float("inf")] * len(rules[0])
    return torch.exp(-torch.tensor(soft, device=dev, dtype=torch.float32)).unsqueeze(1)


def force_plausible_coins(out: dict, candidates, target_probs, draft_probs, uniform_samples, gamma: int):
    """Sampled rows: accept a plausible draft by zeroing its coin (sharar verify_rows), or with accept_soft s by
    scaling it by e^-s (accepted when coin * q < e^s * p).

    target_probs [b, block, V] is the filtered target law, draft_probs [b, gamma, V]
    the draft's q, uniform_samples [b, gamma] the chain kernel's coins (slot j for
    draft j+1, accepted when coin * q < p). Plausible: p_y > 0, rank of y under p
    below the row's top-k, and log p_max - log p_y <= margin; rows with top-k <= 1
    keep exact rejection sampling. Fills out["exact"/"relaxed"/"argmax"] [b, gamma]:
    the original test's outcome, plausible-only acceptances, and the target argmax.
    """
    ks_l, ms_l = out["rules"][0], out["rules"][1]
    dev = candidates.device
    ks = torch.tensor(ks_l, device=dev, dtype=torch.int64)
    ms = torch.tensor(ms_l, device=dev, dtype=torch.float32)
    p = target_probs[:, :gamma].float()
    y = candidates[:, 1 : gamma + 1].long()
    p_y = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    q_y = draft_probs[:, :gamma].float().gather(-1, y.unsqueeze(-1)).squeeze(-1)
    p_max, argmax = p.max(-1)
    rank = (p > p_y.unsqueeze(-1)).sum(-1)
    gap = torch.log(p_max.clamp_min(1e-30)) - torch.log(p_y.clamp_min(1e-30))
    plausible = (p_y > 0) & (rank < ks.unsqueeze(1)) & (gap <= ms.unsqueeze(1)) & (ks.unsqueeze(1) > 1)
    reqs = out.get("reqs")
    if reqs is not None:
        top_ok = [bool((getattr(r.sampling_params, "custom_params", None) or {}).get("relax_top", _DEFAULT_TOP)) for r in reqs]
        if not all(top_ok):
            keep_top = torch.tensor(top_ok, device=dev).unsqueeze(1)
            plausible = plausible & (keep_top | (rank > 0))
    if out.get("gate") is not None and gates_active(out["rules"]):
        plausible = out["gate"].filter(out["reqs"], out["rules"], candidates, plausible)
    coins = uniform_samples[:, :gamma]
    exact = (q_y > 0) & (coins * q_y < p_y)
    relaxed_coins = coins * _coin_scale(out["rules"], dev)
    uniform_samples[:, :gamma] = torch.where(plausible, relaxed_coins, coins)
    out["exact"] = exact
    out["relaxed"] = plausible & ~exact & (q_y > 0) & (relaxed_coins * q_y < p_y)
    out["argmax"] = argmax


_CHAIN_COUNTS = {"steps": 0, "relaxed": 0}


def relax_chain_coins(rules, candidates: torch.Tensor, target_probs: torch.Tensor, coins: torch.Tensor,
                      reqs=None) -> None:
    """The sampled rule for an EAGLE / MTP chain (argmax drafts) verified by the tree kernel with zero draft probs:
    zero the coin of every plausible draft, or scale it by e^-s with accept_soft s (coin j decides draft j+1, as in force_plausible_coins). Plausible: p_y > 0,
    rank of y under the filtered target law below the row's top-k, log p_max - log p_y <= margin. No harm gates."""
    gamma = candidates.shape[1] - 1
    dev = candidates.device
    ks = torch.tensor(rules[0], device=dev, dtype=torch.int64).unsqueeze(1)
    ms = torch.tensor(rules[1], device=dev, dtype=torch.float32).unsqueeze(1)
    p = target_probs[:, :gamma].float()
    y = candidates[:, 1 : gamma + 1].long()
    p_y = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    p_max = p.max(-1).values
    rank = (p > p_y.unsqueeze(-1)).sum(-1)
    gap = torch.log(p_max.clamp_min(1e-30)) - torch.log(p_y.clamp_min(1e-30))
    plausible = (p_y > 0) & (rank < ks) & (gap <= ms) & (ks > 1)
    if reqs is not None:
        top_ok = [bool((getattr(r.sampling_params, "custom_params", None) or {}).get("relax_top", _DEFAULT_TOP)) for r in reqs]
        if not all(top_ok):
            plausible = plausible & (torch.tensor(top_ok, device=dev).unsqueeze(1) | (rank > 0))
    # the plausible drafts the ordinary test (coin < p_y) would have rejected: what relaxing adds
    relaxed_coins = coins[:, :gamma] * _coin_scale(rules, dev)
    _CHAIN_COUNTS["relaxed"] += int((plausible & ~(coins[:, :gamma] < p_y) & (relaxed_coins < p_y)).sum())
    _CHAIN_COUNTS["steps"] += candidates.shape[0]
    coins[:, :gamma] = torch.where(plausible, relaxed_coins, coins[:, :gamma])


def count_chain(relaxed: torch.Tensor, bs: int) -> None:
    """Chain counters for sampled MTP / EAGLE drafts (force_plausible_coins' relaxed mask)."""
    _CHAIN_COUNTS["relaxed"] += int(relaxed.sum())
    _CHAIN_COUNTS["steps"] += bs


def chain_counts() -> dict:
    return dict(_CHAIN_COUNTS)


def rule_key(rules, i: int) -> str:
    key = f"{rules[0][i]}:{rules[1][i]:g}"
    if len(rules) > 4 and not rules[4][i]:
        key += "|no-top"
    if len(rules) > 5 and rules[5][i] != float("inf"):
        key += f"|soft{rules[5][i]:g}"
    if len(rules) > 2:
        if not rules[2][i]:
            key += "|code-exact"
        if rules[3][i] > 0:
            key += f"|rep{rules[3][i]}"
    return key


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
            key = rule_key(rules, i)
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
                key = rule_key(rules, i)
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
        tmp = f"{_STATS_PATH}.{os.getpid()}.tmp"  # per process: a shared path must never crash a server
        try:
            with open(tmp, "w") as f:
                json.dump({"time": time.time(), "rules": self.stats}, f)
            os.replace(tmp, _STATS_PATH)
        except OSError:
            pass
