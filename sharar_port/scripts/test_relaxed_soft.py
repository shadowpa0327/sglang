"""Tests for the soft relaxed acceptance rule (dflash_relaxed accept_soft): the output law of one verify slot, by
Monte Carlo through force_plausible_coins and the chain kernel's test (u * q < p) and leftover (p - q)+, against
    P(x) = min(q(x), c_x * p(x)) + (1 - alpha) * (p(x) - q(x))+ / Z,   c_x = e^s on the plausible set, else 1,
and the guarantee P(x) <= c_x * p(x). The CPU tests emulate the kernel in torch; test_chain_kernel_gpu runs the real
Triton kernel (chain_speculative_sampling_triton) and is skipped without a GPU.
usage: PYTHONPATH=<checkout> python sharar_port/scripts/test_relaxed_soft.py   (or pytest)"""
import math
import os
import types

import torch

os.environ.pop("SGLANG_DFLASH_RELAXED_SOFT", None)  # the module reads its server default at import
import sglang.srt.speculative.dflash_relaxed as R  # noqa: E402

P = torch.tensor([0.60, 0.30, 0.07, 0.03])  # target law: top-2 within 1.0 are plausible (0.6, 0.3)
Q = torch.tensor([0.90, 0.06, 0.02, 0.02])  # a drafter sharper than the target
PLAUS = torch.tensor([True, True, False, False])
B = 400_000
TOL = 0.005  # ~6 standard errors at B draws


def law(p, q, plaus, s):
    c = torch.where(plaus, torch.tensor(math.exp(s) if math.isfinite(s) else math.inf), torch.tensor(1.0))
    a = torch.minimum(q, c * p)
    left = (p - q).clamp(min=0)
    return a + (1 - a.sum()).clamp(min=0) * left / left.sum()


def rules(s, n=B, k=2, m=1.0, top=True):
    return ([k] * n, [m] * n, [True] * n, [0] * n, [top] * n, [s] * n)


def coins_after(s, y, device="cpu", n=B):
    cand = torch.stack([torch.zeros(n, dtype=torch.long, device=device), y], 1)
    tp = P.to(device).expand(n, 2, 4).contiguous()
    dp = Q.to(device).expand(n, 1, 4).contiguous()
    coins = torch.rand(n, 1, device=device)
    out = {"rules": rules(s, n), "gate": None, "reqs": None}
    R.force_plausible_coins(out, cand, tp, dp, coins, 1)
    return cand, tp, dp, coins, out


def sample_cpu(s, seed=0):
    torch.manual_seed(seed)
    y = torch.multinomial(Q, B, replacement=True)
    _, _, _, coins, out = coins_after(s, y)
    accept = (Q[y] > 0) & (coins[:, 0] * Q[y] < P[y])
    left = (P - Q).clamp(min=0)
    tok = torch.where(accept, y, torch.multinomial(left / left.sum(), B, replacement=True))
    return torch.bincount(tok, minlength=4).float() / B, accept.float().mean().item(), out


def test_law_and_bound_cpu():
    for s in [0.0, 0.1, 0.25, 0.5, 1.0, math.inf]:
        emp, acc, _ = sample_cpu(s)
        th = law(P, Q, PLAUS, s)
        a_th = torch.minimum(Q, torch.where(PLAUS, torch.tensor(math.exp(s) if math.isfinite(s) else math.inf), torch.tensor(1.0)) * P).sum().item()
        assert (emp - th).abs().max() < TOL, (s, emp, th)
        assert abs(acc - a_th) < TOL, (s, acc, a_th)
        bound = torch.where(PLAUS, P * (math.exp(s) if math.isfinite(s) else math.inf), P)
        assert (emp <= bound + TOL).all(), (s, emp, bound)


def test_s0_is_exact_and_inf_is_forced():
    emp0, _, _ = sample_cpu(0.0, seed=1)
    assert (emp0 - P).abs().max() < TOL, emp0
    empinf, accinf, _ = sample_cpu(math.inf, seed=2)
    # forced accept: plausible tokens come out at the drafter's own probability
    assert abs(empinf[0] - Q[0]) < TOL and abs(empinf[1] - Q[1]) < TOL and abs(accinf - 1.0) < TOL, (empinf, accinf)


def test_default_is_the_forced_rule():
    r = types.SimpleNamespace(sampling_params=types.SimpleNamespace(custom_params={"accept_top_k": 2, "accept_margin": 1.0}))
    got = R.request_rules([r])
    assert got is not None and got[5] == [math.inf], got
    coins = torch.full((3, 1), 0.5)
    cand = torch.tensor([[0, 0], [0, 1], [0, 2]])
    out = {"rules": rules(math.inf, n=3), "gate": None, "reqs": None}
    R.force_plausible_coins(out, cand, P.expand(3, 2, 4).contiguous(), Q.expand(3, 1, 4).contiguous(), coins, 1)
    assert coins[:, 0].tolist() == [0.0, 0.0, 0.5], coins  # plausible drafts zeroed, the third untouched


def test_request_param_and_rule_key():
    rs = [types.SimpleNamespace(sampling_params=types.SimpleNamespace(custom_params=cp)) for cp in
          [{"accept_top_k": 2, "accept_margin": 1.0, "accept_soft": 0.25}, {"accept_top_k": 2, "accept_margin": 1.0}]]
    got = R.request_rules(rs)
    assert got[5] == [0.25, math.inf], got
    assert R.rule_key(got, 0) == "2:1|soft0.25" and R.rule_key(got, 1) == "2:1", (R.rule_key(got, 0), R.rule_key(got, 1))
    scale = R._coin_scale(got, "cpu")[:, 0].tolist()
    assert abs(scale[0] - math.exp(-0.25)) < 1e-6 and scale[1] == 0.0, scale


def test_relaxed_stats_mask():
    # out["relaxed"] marks exactly the slots the boost accepted and the plain test would have rejected
    torch.manual_seed(3)
    y = torch.multinomial(Q, B, replacement=True)
    coins_plain = torch.rand(B, 1)
    cand = torch.stack([torch.zeros(B, dtype=torch.long), y], 1)
    out = {"rules": rules(0.25), "gate": None, "reqs": None}
    coins = coins_plain.clone()
    R.force_plausible_coins(out, cand, P.expand(B, 2, 4).contiguous(), Q.expand(B, 1, 4).contiguous(), coins, 1)
    plain = coins_plain[:, 0] * Q[y] < P[y]
    soft = coins[:, 0] * Q[y] < P[y]
    assert torch.equal(out["exact"][:, 0], plain) and torch.equal(out["relaxed"][:, 0], soft & ~plain)


def test_chain_kernel_gpu():
    if not torch.cuda.is_available():
        print("  (no GPU: skipped)")
        return
    from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton
    dev = "cuda"
    for s in [0.0, 0.1, 0.25, 0.5, math.inf]:
        torch.manual_seed(10)
        y = torch.multinomial(Q.to(dev), B, replacement=True)
        cand, tp, dp, coins, _ = coins_after(s, y, device=dev)
        n_slots = 2
        retrieve_index = torch.arange(B * n_slots, dtype=torch.int64, device=dev).view(B, n_slots)
        next_token = torch.tensor([1, -1], dtype=torch.int64, device=dev).expand(B, -1).contiguous()
        next_sibling = torch.full((B, n_slots), -1, dtype=torch.int64, device=dev)
        predicts = torch.empty(B * n_slots, dtype=torch.int32, device=dev)
        accept_index = torch.empty((B, n_slots), dtype=torch.int32, device=dev)
        accept_num = torch.empty((B,), dtype=torch.int32, device=dev)
        final = torch.rand(B, device=dev)
        chain_speculative_sampling_triton(predicts, accept_index, accept_num, cand, retrieve_index, next_token,
                                          next_sibling, coins, final, tp, dp, 1.0, 1.0, True)
        tok = predicts.view(B, n_slots)[:, 0].long().cpu()  # the first new token, accepted draft or leftover sample
        emp = torch.bincount(tok, minlength=4).float() / B
        th = law(P, Q, PLAUS, s)
        acc = accept_num.float().mean().item()
        a_th = torch.minimum(Q, torch.where(PLAUS, torch.tensor(math.exp(s) if math.isfinite(s) else math.inf), torch.tensor(1.0)) * P).sum().item()
        print(f"  kernel s={s:<5} accept {acc:.3f} (formula {a_th:.3f})  P " + " ".join(f"{v:.3f}" for v in emp.tolist())
              + "  formula " + " ".join(f"{v:.3f}" for v in th.tolist()))
        assert (emp - th).abs().max() < TOL, (s, emp, th)
        assert abs(acc - a_th) < TOL, (s, acc, a_th)


def law_onehot(p, y, plaus, s):
    """An MTP / EAGLE chain drafts one token y (the head's argmax): accepted with min(1, c_y p_y), else the target
    resamples with y removed."""
    c = (math.exp(s) if math.isfinite(s) else math.inf) if plaus[y] else 1.0
    a = min(1.0, c * p[y].item())
    out = p * (1 - a) / (1 - p[y])
    out[y] = a
    return out


def test_mtp_chain_kernel_gpu():
    """relax_chain_coins through the target-only tree kernel MTP verify uses (zero draft probs, chain layout)."""
    if not torch.cuda.is_available():
        print("  (no GPU: skipped)")
        return
    from sglang.kernels.ops.speculative.sampling import tree_speculative_sampling_target_only
    dev, n = "cuda", 2
    for y in (0, 2):  # a plausible draft ("store") and one outside the plausible set ("moon")
        for s in [0.0, 0.1, 0.25, 0.5, math.inf]:
            torch.manual_seed(20)
            cand = torch.tensor([0, y], dtype=torch.int64, device=dev).expand(B, n).contiguous()
            tp = P.to(dev).expand(B, n, 4).contiguous()
            coins = torch.rand(B, n, device=dev)
            R.relax_chain_coins(rules(s), cand, tp, coins)
            predicts = torch.zeros(B * n, dtype=torch.int32, device=dev)
            accept_index = torch.full((B, n), -1, dtype=torch.int32, device=dev)
            accept_num = torch.empty((B,), dtype=torch.int32, device=dev)
            tree_speculative_sampling_target_only(
                predicts, accept_index, accept_num, cand,
                torch.arange(B * n, dtype=torch.int64, device=dev).view(B, n),
                torch.tensor([1, -1], dtype=torch.int64, device=dev).expand(B, -1).contiguous(),
                torch.full((B, n), -1, dtype=torch.int64, device=dev),
                coins, torch.rand(B, device=dev), tp, torch.zeros_like(tp), 1.0, 1.0, True)
            emp = torch.bincount(predicts.view(B, n)[:, 0].long().cpu(), minlength=4).float() / B
            th = law_onehot(P, y, PLAUS, s)
            acc = accept_num.float().mean().item()
            print(f"  mtp kernel draft {y} s={s:<5} accept {acc:.3f} (formula {th[y].item():.3f})  P "
                  + " ".join(f"{v:.3f}" for v in emp.tolist()) + "  formula " + " ".join(f"{v:.3f}" for v in th.tolist()))
            assert (emp - th).abs().max() < TOL, (y, s, emp, th)
            assert abs(acc - th[y].item()) < TOL, (y, s, acc, th)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
