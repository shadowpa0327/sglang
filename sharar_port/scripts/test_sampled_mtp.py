"""Tests for sampled MTP / EAGLE drafts with the relaxed rule (--speculative-use-rejection-sampling + accept_soft):
the draft proposal shape (spec_utils.draft_proposal_probs: SGLANG_DRAFT_TEMP_SCALE, SGLANG_DRAFT_TOPK) and the output
law of one verify slot through force_plausible_coins and the chain kernel, for a q truncated to the head's top tokens
(target mass outside q's support comes back through the residual (p - q)+).
usage: PYTHONPATH=<checkout> python sharar_port/scripts/test_sampled_mtp.py   (the kernel test skips without a GPU)"""
import math

import torch

import sglang.srt.speculative.dflash_relaxed as R
import sglang.srt.speculative.spec_utils as U

B = 400_000
TOL = 0.005
P = torch.tensor([0.45, 0.30, 0.15, 0.06, 0.04])  # target law; plausible set 2:1 = tokens 0, 1
PLAUS = torch.tensor([True, True, False, False, False])
HEAD = torch.tensor([3.0, 1.5, 1.0, 2.0, -1.0])  # MTP head logits: sharper than P, wrong about token 3


def law(p, q, plaus, s):
    c = torch.where(plaus, torch.tensor(math.exp(s) if math.isfinite(s) else math.inf), torch.tensor(1.0))
    a = torch.minimum(q, c * p)
    left = (p - q).clamp(min=0)
    return a + (1 - a.sum()).clamp(min=0) * left / left.sum(), a.sum().item()


def test_draft_proposal_probs():
    logits = HEAD.repeat(3, 1)
    temps = torch.tensor([[1.0], [0.5], [2.0]])
    U._DRAFT_TEMP_SCALE, U._DRAFT_TOPK = 1.0, 0
    assert torch.allclose(U.draft_proposal_probs(logits, temps), torch.softmax(logits / temps, -1))
    U._DRAFT_TEMP_SCALE = 1.5
    assert torch.allclose(U.draft_proposal_probs(logits, temps), torch.softmax(logits / (1.5 * temps), -1))
    U._DRAFT_TEMP_SCALE, U._DRAFT_TOPK = 1.0, 3
    q = U.draft_proposal_probs(logits, temps, torch.tensor([1 << 30, 2, 1], dtype=torch.int32))
    assert ((q > 0).sum(-1) == torch.tensor([3, 2, 1])).all(), q  # top min(N, top_k)
    assert torch.allclose(q[0], torch.softmax(torch.where(q[0] > 0, HEAD, torch.tensor(-math.inf)), -1))
    U._DRAFT_TOPK = 0


def test_sampled_chain_kernel_gpu():
    if not torch.cuda.is_available():
        print("  (no GPU: skipped)")
        return
    from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton
    dev = "cuda"
    for topn in (0, 2):
        U._DRAFT_TOPK = topn
        q = U.draft_proposal_probs(HEAD.view(1, -1).to(dev), torch.ones(1, 1, device=dev))[0].cpu()
        for s in [0.0, 0.1, 0.25, math.inf]:
            torch.manual_seed(30)
            y = torch.multinomial(q.to(dev), B, replacement=True)
            cand = torch.stack([torch.zeros(B, dtype=torch.long, device=dev), y], 1)
            tp = P.to(dev).expand(B, 2, -1).contiguous()
            dp = q.to(dev).expand(B, 1, -1).contiguous()
            coins = torch.rand(B, 1, device=dev)
            out = {"rules": ([2] * B, [1.0] * B, [True] * B, [0] * B, [True] * B, [s] * B), "gate": None, "reqs": None}
            R.force_plausible_coins(out, cand, tp, dp, coins, 1)
            pred = torch.empty(B * 2, dtype=torch.int32, device=dev)
            chain_speculative_sampling_triton(
                pred, torch.empty((B, 2), dtype=torch.int32, device=dev), an := torch.empty(B, dtype=torch.int32, device=dev),
                cand, torch.arange(B * 2, dtype=torch.int64, device=dev).view(B, 2),
                torch.tensor([1, -1], dtype=torch.int64, device=dev).expand(B, -1).contiguous(),
                torch.full((B, 2), -1, dtype=torch.int64, device=dev), coins, torch.rand(B, device=dev), tp, dp, 1.0, 1.0, True)
            emp = torch.bincount(pred.view(B, 2)[:, 0].long().cpu(), minlength=5).float() / B
            th, a_th = law(P, q, PLAUS, s)
            acc = an.float().mean().item()
            print(f"  top{topn or 'all'} s={s:<5} accept {acc:.3f} (formula {a_th:.3f})  P "
                  + " ".join(f"{v:.3f}" for v in emp.tolist()) + "  formula " + " ".join(f"{v:.3f}" for v in th.tolist()))
            assert (emp - th).abs().max() < TOL and abs(acc - a_th) < TOL, (topn, s, emp, th, acc, a_th)
            if s == 0.0:
                assert (emp - P).abs().max() < TOL, (topn, emp)  # exact, even with q truncated
    U._DRAFT_TOPK = 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
