"""MonoKernel MoE stages (router logits, top-k + plan, W8A16 gate/up and down) against the goldens.

Uses the full 512 + 1 expert geometry with random FP8 banks in SGLang's served format (16x16
shuffle, intermediate rows padded to 64): TP2 (320) and TP4 (160 padded to 192).
"""

import pytest
import torch

from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT, shard
from pyhip.ops.monokernel.qwen38.reference import bf16_linear, golden_moe
from pyhip.ops.monokernel.qwen38.weights import KERNEL_FP8, moe_inter_padded, shuffle_16x16

if not torch.cuda.is_available():
    pytest.skip("MonoKernel stages run on a GPU", allow_module_level=True)

from pyhip.ops.monokernel.qwen38.flydsl.linear import linear_bf16  # noqa: E402
from pyhip.ops.monokernel.qwen38.flydsl.moe import MoEWorkspace, moe_down, moe_up, topk_plan  # noqa: E402

from .test_qwen38_stages import assert_bf16_ulp  # noqa: E402

C = QWEN38_FLASH_NEXT
DEVICE = torch.device("cuda")
EXPERTS = C.n_experts + 1


def _fp8(shape, gen):
    # Moderate magnitudes: exponent fields 4..10 with random sign and mantissa.
    bits = torch.randint(0x20, 0x58, shape, generator=gen, device=DEVICE, dtype=torch.int16)
    sign = torch.randint(0, 2, shape, generator=gen, device=DEVICE, dtype=torch.int16) * 0x80
    return (bits | sign).to(torch.uint8).view(KERNEL_FP8)


@pytest.fixture(scope="module", params=(2, 4))
def bank(request):
    tp = request.param
    s = shard(tp, 0, C)
    inter, padded = C.inter // tp, moe_inter_padded(s)
    gen = torch.Generator(device=DEVICE).manual_seed(tp)
    up = _fp8((EXPERTS, 2 * inter, C.hidden), gen)
    down = _fp8((EXPERTS, C.hidden, inter), gen)
    up_scale = 1e-3 * (0.5 + torch.rand(EXPERTS, 2 * inter, generator=gen, device=DEVICE))
    down_scale = 1e-3 * (0.5 + torch.rand(EXPERTS, C.hidden, generator=gen, device=DEVICE))
    router = (torch.randn(C.n_experts, C.hidden, generator=gen, device=DEVICE) * C.hidden**-0.5).to(torch.bfloat16)
    shared_gate = (torch.randn(C.hidden, generator=gen, device=DEVICE) * C.hidden**-0.5).to(torch.bfloat16)
    pad = padded - inter
    halves = [torch.nn.functional.pad(h.view(torch.uint8), (0, 0, 0, pad)) for h in up.split(inter, dim=1)]
    served = {
        "w13": shuffle_16x16(torch.cat(halves, 1)).view(KERNEL_FP8),
        "w13_scale": torch.cat([torch.nn.functional.pad(h, (0, pad)) for h in up_scale.split(inter, 1)], 1)[..., None],
        "w2": shuffle_16x16(torch.nn.functional.pad(down.view(torch.uint8), (0, pad))).view(KERNEL_FP8),
        "w2_scale": down_scale[..., None],
        "router": shuffle_16x16(router),
        "shared_gate": shuffle_16x16(torch.nn.functional.pad(shared_gate[None], (0, 0, 0, 15))),
    }
    t = {"moe_up": up, "moe_up_scale": up_scale, "moe_down": down, "moe_down_scale": down_scale, "router": router,
         "shared_gate": shared_gate}
    return s, padded, t, served


@pytest.mark.parametrize("rows", (1, 5, 16, 33, 64))
def test_moe(rows: int, bank) -> None:
    s, padded, t, served = bank
    gen = torch.Generator(device=DEVICE).manual_seed(500 + rows)
    x = torch.randn(rows, C.hidden, generator=gen, device=DEVICE).to(torch.bfloat16)
    work = MoEWorkspace(rows, C.n_experts, C.top_k, padded, DEVICE)
    linear_bf16(x, served["router"], work.logits)
    linear_bf16(x, served["shared_gate"], work.shared)
    assert_bf16_ulp(work.logits, bf16_linear(x, t["router"]), exact=0.99)
    torch.testing.assert_close(work.shared[:, 0], x.float() @ t["shared_gate"].float(), rtol=1e-5, atol=1e-5)

    topk_plan(work, rows)
    probs, ids = torch.softmax(work.logits.float(), -1).sort(dim=-1, descending=True, stable=True)
    probs, ids = probs[:, : C.top_k], ids[:, : C.top_k]
    assert torch.equal(work.ids[:, : C.top_k].long(), ids)
    assert bool((work.ids[:, C.top_k] == C.n_experts).all())
    torch.testing.assert_close(work.weights[:, : C.top_k], probs / probs.sum(-1, keepdim=True), rtol=2e-6, atol=0)
    torch.testing.assert_close(work.weights[:, C.top_k], torch.sigmoid(work.shared[:, 0]), rtol=2e-6, atol=0)

    plan = work.plan.cpu()
    active = int(plan[0])
    expected_experts = torch.unique(work.ids.cpu())
    assert torch.equal(plan[1 : 1 + active], expected_experts.int())
    offsets, counts = plan[1 + EXPERTS : 1 + EXPERTS + active], plan[1 + 2 * EXPERTS : 1 + 2 * EXPERTS + active]
    route_row = work.route_row.cpu()
    for expert, offset, count in zip(expected_experts.tolist(), offsets.tolist(), counts.tolist()):
        expected_rows = (work.ids.cpu() == expert).any(-1).nonzero().flatten()
        assert torch.equal(route_row[offset : offset + count], expected_rows.int())

    moe_up(x, served["w13"], served["w13_scale"], work)
    out = torch.empty(rows, C.hidden, dtype=torch.bfloat16, device=DEVICE)
    moe_down(served["w2"], served["w2_scale"], work, out)
    golden = golden_moe(x, t, s, ids=work.ids.long(), weights=work.weights)
    inter = C.inter // s.tp
    got_mid, expected_mid = [], []
    for expert, offset, count in zip(expected_experts.tolist(), offsets.tolist(), counts.tolist()):
        for position in range(count):
            row = int(route_row[offset + position])
            slot = int((work.ids[row] == expert).nonzero()[0])
            got_mid.append(work.mid[offset + position])
            expected_mid.append(golden["mid"][row, slot])
    got_mid = torch.stack(got_mid)
    assert not got_mid[:, inter:].any()
    # silu(g) * u multiplies two FP32 sums: near-zero products can sit a few ulps from the golden.
    assert_bf16_ulp(got_mid[:, :inter], torch.stack(expected_mid), exact=0.98, ulp1=0.995)
    assert_bf16_ulp(out, golden["partial"], exact=0.97, ulp1=0.995)
