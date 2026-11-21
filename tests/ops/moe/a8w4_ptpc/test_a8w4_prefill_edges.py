"""A8W4（mxfp4 权重）MoE 的 M2 复核补充：``test_a8w4_prefill.py`` 没覆盖到的 prefill 分支与形状边界。

参考与容差同 ``test_a8w4_prefill.py``（``moe_a8w4_mxfp4.reference_moe(act_quant=True)`` + M0 的 ``check_output``，
不放宽），并同样确认输出离 BF16 激活的参考远 10 倍以上。

- CPU：``prepare_weights`` 按专家分块、最后一块不满（E=3、chunk=2）时，prefill 另存的 ``w1_pdelta`` /
  ``w2_pdelta`` 与不分块相同。
- GPU，``config`` 指定 prefill 档用于小 T（交界随调优移动时会用到）：T=0 不编译、不启动，直接返回；
  T·topk ≤ E（Qwen3.8 TP2 为 T ≤ 51）时 Gate/Up 的任务数改为 T·topk 的分支（T=1、51），以及交界另一侧
  T=52（不传 ``output``，新建输出）。
- 形状边界（小形状 E=4、topk=2、T=200，默认配置即 prefill 档）：H 取 prefill 档的最小值 512（最后一步
  ``sorted_sum`` 要求 H % 512 == 0），Gate/Up 的 K=512 在 BK128 下共 4 个 K tile（主循环 1 次）、BK256 下
  共 2 个（只有序言与收尾两拍，主循环 0 次）；Down 的 K（补零后的 I）取 LDS 允许的最大值 832；补零后
  I=896 超出 Down 的 LDS，``fused_moe`` 在启动 Gate/Up 之前报 ``AssertionError``、不写输出。
- ``compile_prefill_gemm`` 构建时的形状断言：Gate/Up 的 K % 256、K 方向 tile 数为奇数、N % BN，
  Down 的 N % 256 与 LDS 上限（K=832 可以、896 拒绝）。
"""

import pytest
import torch

from pyhip.ops.moe.flydsl import a8w4_ptpc as A
from pyhip.ops.moe.flydsl.a8w4_ptpc import moe as entry
from pyhip.testing import moe_a8w4_mxfp4 as X
from pyhip.testing import moe_a8w4_ptpc as M

PREFILL = A.select_config(A.decode_max_tokens() + 1)
SMALL_TOKENS = [1, M.E // M.TOPK, M.E // M.TOPK + 1]  # 1、51（T·topk ≤ E 的上界）、52


@pytest.fixture
def gpu(monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("A8W4 prefill kernels target MI308X (gfx942)")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # 测当前源码，不复用旧磁盘编译产物。
    try:
        yield
    finally:
        torch.set_default_device(previous)


@pytest.fixture(scope="module")
def shard():
    """``shard(tp) -> (raw, prepared)``，每个 TP 只生成一次。"""
    cache = {}

    def get(tp):
        if tp not in cache:
            cache[tp] = X.make_mxfp4_weights(tp, seed=40 + tp, device="cuda")
        return cache[tp]

    yield get
    cache.clear()


def _record_compiles(monkeypatch):
    """记录 fused_moe 请求编译的 kernel：prefill 为 (stage, N, K)，decode 为 ("decode", stage)。"""
    compiled = []
    prefill, decode = entry.compile_prefill_gemm, entry.compile_decode_gemm

    def compile_prefill_gemm(**params):
        compiled.append((params["stage"], params["N"], params["K"]))
        return prefill(**params)

    def compile_decode_gemm(**params):
        compiled.append(("decode", params["stage"]))
        return decode(**params)

    monkeypatch.setattr(entry, "compile_prefill_gemm", compile_prefill_gemm)
    monkeypatch.setattr(entry, "compile_decode_gemm", compile_decode_gemm)
    return compiled


def _check(hidden, raw, weight, ids, output, msg):
    _, logits_diff, _ = M.check_output(X.reference_moe(hidden, raw, weight, ids, act_quant=True), output, msg)
    _, bf16_diff, _ = M.output_errors(X.reference_moe(hidden, raw, weight, ids, act_quant=False), output)
    assert bf16_diff > 10 * logits_diff, (msg, logits_diff, bf16_diff)


def _small_raw(experts, hidden, inter, *, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.randint(0, 256, (experts, 2 * inter, hidden // 2), generator=gen, device=device).to(torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, inter // 2), generator=gen, device=device).to(torch.uint8)
    s1 = X.random_codes((experts, 2 * inter, hidden // 32), gen=gen, device=device)
    s2 = X.random_codes((experts, hidden, inter // 32), gen=gen, device=device)
    return w1, s1, w2, s2


def _small_inputs(tokens, experts, hidden, topk, *, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((tokens, hidden), generator=gen, device="cuda").to(torch.bfloat16)
    ids = torch.rand((tokens, experts), generator=gen, device="cuda").topk(topk, dim=1).indices
    weight = torch.randn((tokens, topk), generator=gen, device="cuda").softmax(dim=-1)
    return x, weight.contiguous(), ids.to(torch.int32).contiguous()


# ---------------------------------------------------------------- prepare_weights（CPU）

def test_prefill_delta_partial_last_chunk():
    raw = _small_raw(3, 512, 96, seed=11, device="cpu")
    base = A.prepare_weights(*raw)
    got = A.prepare_weights(*raw, chunk=2)  # 最后一块只有 1 个专家
    assert torch.equal(got.w1_pdelta, base.w1_pdelta)
    assert torch.equal(got.w2_pdelta, base.w2_pdelta)


# ---------------------------------------------------------------- config 指定 prefill 档用于小 T（GPU）

def test_prefill_override_empty_input(gpu, shard, monkeypatch):
    _, prep = shard(2)
    hidden, weight, ids = M.make_inputs(0, seed=700)
    compiled = _record_compiles(monkeypatch)
    out = A.fused_moe(hidden, prep, weight, ids, config=PREFILL)
    assert out.shape == (0, M.H) and out.dtype == torch.bfloat16
    assert compiled == []


@pytest.mark.parametrize("tokens", SMALL_TOKENS, ids=[f"tp2-t{t}" for t in SMALL_TOKENS])
def test_prefill_override_small_tokens(gpu, shard, monkeypatch, tokens):
    raw, prep = shard(2)
    hidden, weight, ids = M.make_inputs(tokens, seed=710 + tokens)
    compiled = _record_compiles(monkeypatch)
    if tokens * M.TOPK > M.E:
        output = A.fused_moe(hidden, prep, weight, ids, config=PREFILL)  # 不传 output
    else:
        output = torch.full_like(hidden, float("nan"))
        assert A.fused_moe(hidden, prep, weight, ids, output=output, config=PREFILL) is output
    assert [c[0] for c in compiled] == ["gateup", "down"], compiled
    _check(hidden, raw, weight, ids, output, f"TP2 T={tokens} {PREFILL}")


# ---------------------------------------------------------------- 形状边界（GPU）

def test_prefill_min_hidden_max_inter(gpu, monkeypatch):
    experts, hidden_size, topk, tokens = 4, 512, 2, 200
    assert A.select_config(tokens) == PREFILL
    x, weight, ids = _small_inputs(tokens, experts, hidden_size, topk, seed=720)
    compiled = _record_compiles(monkeypatch)
    # 补零后 I=832：Down 的 LDS = 64·832 + 1 KiB + 8 KiB = 62464 B ≤ 64 KiB。
    raw = _small_raw(experts, hidden_size, 832, seed=721, device="cuda")
    prep = A.prepare_weights(*raw)
    for tile_k in (128, 256):  # K=512：BK128 主循环 1 次（配置表这一档），BK256 主循环 0 次
        config = dict(PREFILL, tile_k_gate=tile_k)
        output = torch.full_like(x, float("nan"))
        compiled.clear()
        assert A.fused_moe(x, prep, weight, ids, output=output, config=config) is output
        assert compiled == [("gateup", 2 * 832, hidden_size), ("down", hidden_size, 832)], compiled
        _check(x, raw, weight, ids, output, f"E={experts} H={hidden_size} I=832 {config}")
    # 补零后 I=896 超出 Down 的 LDS：构建 Down 时报错，Gate/Up 不启动，不写输出。
    raw = _small_raw(experts, hidden_size, 896, seed=722, device="cuda")
    output = torch.full_like(x, float("nan"))
    compiled.clear()
    with pytest.raises(AssertionError, match="LDS"):
        A.fused_moe(x, A.prepare_weights(*raw), weight, ids, output=output)
    assert [c[0] for c in compiled] == ["gateup", "down"], compiled
    assert bool(output.isnan().all())


def test_compile_prefill_gemm_shape_asserts(gpu):
    from pyhip.ops.moe.flydsl.a8w4_ptpc import prefill as P

    gateup = dict(stage="gateup", N=2 * 192, K=M.H, TOPK=M.TOPK, BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=128,
                  tile_k=128, E=M.E)
    down = dict(stage="down", N=M.H, K=192, TOPK=M.TOPK, BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=256,
                padding=128, E=M.E)
    P.compile_prefill_gemm(**dict(down, K=832))  # LDS 上限以内，可以构建
    for bad, match in (
        (dict(gateup, K=384), "multiple of 256"),                 # w1_pdelta 按 256-K 分块
        (dict(gateup, K=768, tile_k=256), "even K-tile count"),   # 两级流水要偶数个 K tile
        (dict(gateup, BLOCK_TILE_SIZE_N=256), None),              # TP4 的 N=384 不能用 BN256
        (dict(down, N=384), None),                                # Down 的 N % 256
        (dict(down, K=896), "LDS"),                               # 64 行激活放不进 LDS
    ):
        with pytest.raises(AssertionError, match=match):
            P.compile_prefill_gemm(**bad)
