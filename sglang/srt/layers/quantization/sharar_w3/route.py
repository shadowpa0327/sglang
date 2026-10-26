# Vendored from abdelfattah-lab/sharar @ 0ecb67f (sharar/kernels/flashnext_fused.py: route_budget, Apache-2.0):
# Flash-Next's budgeted expert selection (expert sharing) for decode and verify forwards, unchanged.
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---- budgeted expert selection (SHARAR_MOE_KEEP / SHARAR_MOE_BUDGET): a cap on the distinct experts a decode or verify
# forward reads, chosen for all its rows; each row takes its guaranteed top `keep`, then its other candidates in the set ----
ROUTE_CANDIDATES = 20  # a row takes its experts from its router ranks 1..20


@triton.jit
def _or(a, b):
    return a | b


@triton.jit
def _route_rank_kernel(L, Slots, Cand, P, Bits, Dense, stride_l, E: tl.constexpr, BLOCK: tl.constexpr, KEEP: tl.constexpr,
                       C: tl.constexpr, CP: tl.constexpr, STEPS: tl.constexpr, HAS_SLOTS: tl.constexpr, K: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < E
    logits = tl.load(L + t * stride_l + cols, mask=mask, other=float("-inf")).to(tl.float32)
    m = tl.max(logits, 0)
    s = tl.sum(tl.exp(logits - m), 0)
    # rank key, distinct per column: the bf16 bits as an order-preserving int (-0.0 as +0.0) above BLOCK - 1 - column,
    # so the int32 maximum is the largest logit and, among equal logits, the lowest expert
    hi = logits.to(tl.int32, bitcast=True) >> 16
    hi = tl.where(hi == -32768, 0, hi)
    key = tl.where(mask, (hi ^ ((hi >> 15) & 0x7FFF)) * 65536 + (BLOCK - 1 - cols), -2147483648)
    j = tl.arange(0, CP)
    ids = tl.zeros([CP], dtype=tl.int32)
    vals = tl.full([CP], float("-inf"), tl.float32)
    w = tl.arange(0, BLOCK // 32)
    bits = tl.zeros([BLOCK // 32], dtype=tl.int32)
    dense = tl.zeros([BLOCK], dtype=tl.float32)
    for c in tl.static_range(C):  # rank c: the largest key left
        top = tl.max(key, 0)
        e = BLOCK - 1 - (top & 0xFFFF)
        o = top >> 16
        v = ((o ^ ((o >> 15) & 0x7FFF)) << 16).to(tl.float32, bitcast=True)  # its logit, exactly
        ids = tl.where(j == c, e, ids)
        vals = tl.where(j == c, v, vals)
        if c < K:
            dense = tl.where(cols == e, tl.exp(v - m) / s, dense)
        if c < KEEP:
            bits = tl.where(w == e // 32, bits | (1 << (e % 32)), bits)
        key = tl.where(cols == e, -2147483648, key)
    p = tl.exp(vals - m) / s  # route_topk's p of each candidate (0 past rank C)
    tl.store(Cand + t * CP + j, ids)
    tl.store(P + t * CP + j, p)
    real = True
    if HAS_SLOTS:  # a padded row (dummy slot 0) adds nothing to the core or the scores
        real = tl.load(Slots + t // STEPS) != 0
        bits = tl.where(real, bits, 0)
    tl.store(Bits + t * (BLOCK // 32) + w, bits)
    tl.store(Dense + t * BLOCK + cols, tl.where(real, dense, 0.0))  # its p within its top K by expert, for the scores


@triton.jit
def _route_pick_kernel(Slots, Cand, P, Set, Ids, W, R, K: tl.constexpr, C: tl.constexpr,
                       CP: tl.constexpr, NW: tl.constexpr, RB: tl.constexpr, STEPS: tl.constexpr,
                       HAS_SLOTS: tl.constexpr, KP: tl.constexpr, NONE: tl.constexpr, REPEAT: tl.constexpr):
    rr = tl.arange(0, RB)
    core = tl.load(Set + tl.arange(0, NW))  # the call's set, from `_budget_select_kernel`
    rows = tl.program_id(0) * RB + rr
    live = rows < R
    j = tl.arange(0, CP)[None, :]
    cand = tl.load(Cand + rows[:, None] * CP + j, mask=live[:, None], other=0)
    p = tl.load(P + rows[:, None] * CP + j, mask=live[:, None], other=1.0)  # rows past R: no 0 / 0
    word = tl.reshape(tl.gather(core, tl.reshape(cand // 32, [RB * CP]), 0), [RB, CP])
    pull = (j < C) & (((word >> (cand % 32)) & 1) != 0)  # a candidate in the set (the row's guaranteed ones are)
    real = tl.full([RB], 1, tl.int1)
    if HAS_SLOTS:  # a padded row takes no expert
        real = tl.load(Slots + rows // STEPS, mask=live, other=0) != 0
    a = pull.to(tl.int32)
    slot = tl.cumsum(a, 1) - 1  # output slot of each rank: the candidates in the set, in rank order
    chosen = pull & (slot < K) & real[:, None]
    used = tl.where(real, tl.minimum(tl.sum(a, 1), K), 0)
    total = tl.sum(tl.where(chosen, p, 0.0), 1)
    smask = live[:, None] & chosen
    tl.store(Ids + rows[:, None] * K + slot, cand, mask=smask)
    tl.store(W + rows[:, None] * K + slot, p / total[:, None], mask=smask)
    # the slots left over name expert NONE (= E) at weight 0: the w3 kernel reads nothing for them
    q = tl.arange(0, KP)[None, :]
    hole = live[:, None] & (q >= used[:, None]) & (q < K)
    if REPEAT:  # an expert the call already reads, at weight 0: the row's top 1, a padded row row 0's
        first = tl.where(real, tl.load(Cand + rows * CP, mask=live, other=0), tl.load(Cand))
        fill_id = tl.broadcast_to(first[:, None], [RB, KP])
    else:
        fill_id = tl.full([RB, KP], NONE, tl.int32)
    tl.store(Ids + rows[:, None] * K + q, fill_id, mask=hole)
    tl.store(W + rows[:, None] * K + q, tl.zeros([RB, KP], dtype=tl.float32), mask=hole)


@triton.jit
def _budget_score_kernel(Dense, Score, R, BLOCK: tl.constexpr, EB: tl.constexpr, RB: tl.constexpr):
    """Score[e]: the router probability expert e takes within the top K of the call's real rows (`_route_rank_kernel`'s
    dense rows), summed in row order (one program per EB experts; no atomics, so the same bits on every run)."""
    ex = tl.program_id(0) * EB + tl.arange(0, EB)
    rr = tl.arange(0, RB)
    acc = tl.zeros([EB], dtype=tl.float32)
    for r0 in range(0, R, RB):  # RB rows a load, added one row at a time (rows past R add 0)
        tile = tl.load(Dense + (r0 + rr)[:, None] * BLOCK + ex[None, :], mask=(r0 + rr)[:, None] < R, other=0.0)
        for i in tl.static_range(RB):
            acc += tl.sum(tl.where(rr[:, None] == i, tile, 0.0), 0)
    tl.store(Score + ex, acc)


@triton.jit
def _budget_select_kernel(Score, Bits, Slots, Set, R, Budget, PerRequest, MinCap, E: tl.constexpr, BLOCK: tl.constexpr,
                          NW: tl.constexpr, RB: tl.constexpr, STEPS: tl.constexpr, HAS_SLOTS: tl.constexpr):
    """Word `program_id` of the selected set: every real row's top KEEP (the OR of Bits), then the highest-scoring other
    experts up to the cap in all (equal scores: the lower expert). The cap is Budget, or with PerRequest,
    min(Budget, max(MinCap, PerRequest x the call's real requests)). One program per 32 experts, each ranking its own
    against all."""
    g = tl.program_id(0)
    if PerRequest > 0:
        reqs = R // STEPS
        if HAS_SLOTS:  # a graph's padded requests (dummy slot 0) do not count
            reqs = 0
            for q0 in range(0, R // STEPS, RB):
                q = q0 + tl.arange(0, RB)
                reqs += tl.sum((tl.load(Slots + q, mask=q < R // STEPS, other=0) != 0).to(tl.int32), 0)
        Budget = tl.minimum(Budget, tl.maximum(MinCap, PerRequest * reqs))
    w = tl.arange(0, NW)
    rr = tl.arange(0, RB)
    core = tl.zeros([NW], dtype=tl.int32)
    for r0 in range(0, R, RB):
        b = tl.load(Bits + (r0 + rr)[:, None] * NW + w[None, :], mask=(r0 + rr)[:, None] < R, other=0)
        core = core | tl.reduce(b, 0, _or)
    cols = tl.arange(0, BLOCK)
    incore = ((tl.gather(core, cols // 32, 0) >> (cols % 32)) & 1) != 0
    s = tl.load(Score + cols, mask=cols < E, other=0.0)
    key = tl.where(incore | (s <= 0.0) | (cols >= E), -1.0, s)
    need = Budget - tl.sum(incore.to(tl.int32), 0)
    f = g * 32 + tl.arange(0, 32)  # this program's experts
    sf = tl.load(Score + f, mask=f < E, other=0.0)
    inf = ((tl.gather(core, f // 32, 0) >> (f % 32)) & 1) != 0
    kf = tl.where(inf | (sf <= 0.0) | (f >= E), -1.0, sf)
    ahead = (key[None, :] > kf[:, None]) | ((key[None, :] == kf[:, None]) & (cols[None, :] < f[:, None]))
    rank = tl.sum(ahead.to(tl.int32), 1)  # other experts ahead of each: higher score, or equal and lower id
    sel = inf | ((kf > 0.0) & (rank < need))
    tl.store(Set + g, tl.sum(tl.where(sel, 1 << (f % 32), 0), 0))  # distinct bits: the sum is the OR


def route_budget(logits: torch.Tensor, k: int, keep: int, budget: int, slots: torch.Tensor | None = None, steps: int = 1,
                 repeat: bool = False, per_request: int = 0, min_cap: int = 0):
    """Budgeted top-k: logits bf16 [R, E] contiguous -> (int32 ids [R, k], fp32 weights [R, k]). The experts the call
    reads are chosen for the whole call, under a cap:
      - the set: every real row's top `keep` (guaranteed), then the experts with the most router probability within the
        real rows' top k, summed over the rows, until it holds the cap (more only when the guaranteed experts alone
        exceed it). The cap is `budget`, or with `per_request`, min(budget, max(min_cap, per_request x the real
        requests));
      - each real row takes its candidates (its ROUTE_CANDIDATES largest logits, largest first, an exact logit tie to
        the lower expert) that are in the set, in rank order, up to k: its top `keep` first.
    Slots left over (all of a padded row's) hold expert id E at weight 0, which the 3-bit experts skip; with `repeat`
    they name the row's top 1 (a padded row: row 0's) at weight 0, for kernels that need a real expert in every slot
    (FlashInfer's NVFP4 MoE: the same bits, no further expert read). Row r is real iff slots[r // steps] != 0 (slot 0
    is the dummy padded graph rows use); without `slots` every row is. Weights: the fp32 softmax over E at the chosen
    experts, renormalized over them. Four launches, no atomics (the same bits on every run), no host sync:
    graph-capturable."""
    r, e = logits.shape
    block, cp = triton.next_power_of_2(e), triton.next_power_of_2(ROUTE_CANDIDATES)
    nw = block // 32
    dev = logits.device
    cand = torch.empty((r, cp), dtype=torch.int32, device=dev)
    p = torch.empty((r, cp), dtype=torch.float32, device=dev)
    bits = torch.empty((r, nw), dtype=torch.int32, device=dev)
    dense = torch.empty((r, block), dtype=torch.float32, device=dev)
    score = torch.empty((block,), dtype=torch.float32, device=dev)
    chosen = torch.empty((nw,), dtype=torch.int32, device=dev)
    ids = torch.empty((r, k), dtype=torch.int32, device=dev)
    w = torch.empty((r, k), dtype=torch.float32, device=dev)
    sl = cand if slots is None else slots
    common = dict(C=ROUTE_CANDIDATES, CP=cp, STEPS=steps, HAS_SLOTS=slots is not None)
    # one warp per row: each of the ROUTE_CANDIDATES rank passes reduces with warp shuffles alone
    _route_rank_kernel[(r,)](logits, sl, cand, p, bits, dense, logits.stride(0), e, BLOCK=block, KEEP=keep, K=k,
                             num_warps=1, **common)
    _budget_score_kernel[(block // 32,)](dense, score, r, BLOCK=block, EB=32, RB=16, num_warps=1)  # 16 rows a load
    rb = min(64, max(16, triton.next_power_of_2(r)))
    _budget_select_kernel[(nw,)](score, bits, sl, chosen, r, budget, per_request, min_cap, E=e, BLOCK=block, NW=nw, RB=rb, STEPS=steps,
                                 HAS_SLOTS=slots is not None, num_warps=4)
    # the rows take their candidates in the set in rank order: a row's top `keep` are in it, so they come first
    _route_pick_kernel[(triton.cdiv(r, rb),)](sl, cand, p, chosen, ids, w, r, k, NW=nw, RB=rb, num_warps=4,
                                              KP=triton.next_power_of_2(k), NONE=e, REPEAT=repeat, **common)
    return ids, w


