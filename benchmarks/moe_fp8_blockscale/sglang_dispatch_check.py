"""sglang AiterRunnerCore.run with SGLANG_USE_PYHIP_FP8_MOE: which calls go to PyHIP, and their output vs aiter.

usage: PYTHONPATH=<sglang>/python python benchmarks/moe_fp8_blockscale/sglang_dispatch_check.py
"""

import os

os.environ.update(SGLANG_USE_AITER="1", SGLANG_USE_PYHIP_FP8_MOE="1", SGLANG_PYHIP_FP8_MOE_MIN_TOKENS="4096",
                  SGLANG_PYHIP_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.moe.moe_runner import aiter as runner  # noqa: E402
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # noqa: E402

FP8 = torch.float8_e4m3fnuz

calls = []
_original = runner._load_pyhip_fp8_moe()


def _counting(*args, **kwargs):
    calls.append(args[0].shape[0])
    return _original(*args, **kwargs)


runner._load_pyhip_fp8_moe = lambda: _counting
torch.set_grad_enabled(False)


def case(name, tokens, inter, local, mask_rank=None, expect=True, fp8_input=False):
    g = torch.Generator(device="cuda").manual_seed(1)
    hidden, experts = 4096, 256
    x = torch.randn(tokens, hidden, device="cuda", generator=g).to(torch.bfloat16)
    w13 = shuffle_weight((torch.randn(local, 2 * inter, hidden, device="cuda", generator=g) * 0.5).to(FP8),
                         (16, 16))
    w2 = shuffle_weight((torch.randn(local, hidden, inter, device="cuda", generator=g) * 0.5).to(FP8), (16, 16))
    w13.is_shuffled = w2.is_shuffled = True
    s13 = torch.rand(local, 2 * inter // 128, hidden // 128, device="cuda", generator=g) * 0.02 + 0.005
    s2 = torch.rand(local, hidden // 128, inter // 128, device="cuda", generator=g) * 0.02 + 0.005
    wts, ids = torch.topk(torch.sigmoid(torch.randn(tokens, experts, device="cuda", generator=g)), 8, -1)
    wts = (wts / wts.sum(-1, keepdim=True)).float()
    mask = None
    if mask_rank is not None:
        mask = torch.zeros(experts, dtype=torch.int32, device="cuda")
        mask[mask_rank * local:(mask_rank + 1) * local] = 1
    a1_scale, transposed, xin = None, False, x
    if fp8_input:
        from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import quantize_input

        xin, xs_t = quantize_input(x)
        a1_scale = torch.as_strided(xs_t, (tokens, hidden // 128), (hidden // 128, 1))  # bytes [H/128, T]
        transposed = True
    info = runner.AiterMoeQuantInfo(w13_weight=w13, w2_weight=w2, quant_type=runner.AiterQuantType.PER_128X128,
                                    w13_scale=s13, w2_scale=s2, expert_mask=mask)
    inp = runner.AiterRunnerInput(hidden_states=xin, topk_ids=ids.to(torch.int32), topk_weights=wts,
                                  quant_type=runner.AiterQuantType.PER_128X128, a1_scale=a1_scale,
                                  a1_scale_is_transposed=transposed,
                                  output_dtype=torch.bfloat16 if fp8_input else None)
    core = runner.AiterRunnerCore(MoeRunnerConfig(num_experts=experts, num_local_experts=local, top_k=8,
                                                  hidden_size=hidden, intermediate_size_per_partition=inter))
    got = core.run(inp, info, {}).hidden_states
    used = bool(calls)
    calls.clear()
    with envs.SGLANG_USE_PYHIP_FP8_MOE.override(False):
        want = core.run(inp, info, {}).hidden_states
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    ok = used == expect and rel < 1e-2 and bool(torch.isfinite(got.float()).all())
    print(f"{'OK ' if ok else 'BAD'} {name:44s} pyhip={used} rel_l2 vs aiter={rel:.2e}", flush=True)
    return ok


good = all([
    case("TP4 I512 T8192 bf16 input", 8192, 512, 256),
    case("TP4 I512 T8192 fp8 input (transposed scale)", 8192, 512, 256, fp8_input=True),
    case("EP8 rank3 I2048 T8192 (expert_mask)", 8192, 2048, 32, mask_rank=3),
    case("TP8 I256 T8192 -> aiter", 8192, 256, 256, expect=False),
    case("TP4 I512 T2048 < min tokens -> aiter", 2048, 512, 256, expect=False),
])
print("ALL OK" if good else "FAILURES")
