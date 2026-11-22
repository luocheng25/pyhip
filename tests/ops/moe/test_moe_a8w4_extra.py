"""A8W4 MoE 的其他覆盖测试。

只放基本功能测试（test_moe_a8w4.py）与 bench（benchmarks/moe/bench_moe_a8w4.py）覆盖不了的情况，
每个测试的 docstring 写明原因。端到端判据与基本测试相同（``pyhip.testing.moe_a8w4.check``，
参考的激活精度按实际启动的 kernel 选择）；逐位用例要求 0 误差。空 expert、部分填充的尾块、
多块 expert 与 I=160→192 补零已由基本测试覆盖，E=513 的各个 M 档由 bench 覆盖，这里不重复。
"""

import pytest
import torch

from pyhip.ops.moe.flydsl import moe_a8w4
from pyhip.testing import moe_a8w4 as oracle

HIDDEN, TOPK = 2560, 11


@pytest.fixture(autouse=True)
def _gpu_environment(monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("the A8W4 kernels target gfx942")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # 测当前源码，不复用旧磁盘编译产物。
    try:
        yield
    finally:
        torch.set_default_device(previous)


def _prepare(w):
    return moe_a8w4.prepare_a8w4_weights(w["w1"], w["w1_scale"], w["w2"], w["w2_scale"])


def _run(w, prepared, x, topk_weight, topk_ids, **options):
    out = torch.full_like(x, float("nan"))
    moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out, **options)
    mode = oracle.path_activation(moe_a8w4.last_path)
    ref = oracle.reference(x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight, topk_ids,
                           activation=mode)
    return out, ref


def test_grid_shrink_boundary_e513():
    """E=513 满规模下 Gate/Up、Down 的 grid 收缩（M·topk <= E 时 task 数取 M·topk）交界两侧：M=46（506）、M=47（517）。

    基本测试用 E=33；bench 的 M 档（1、16、128…）离这个交界都很远。
    """
    w = oracle.make_weights(513, HIDDEN, 320, seed=31)
    prepared = _prepare(w)
    for tokens in (46, 47):
        x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 513, TOPK, seed=32 + tokens)
        oracle.assert_close(*_run(w, prepared, x, topk_weight, topk_ids))


@pytest.mark.parametrize("inter, options", [
    (320, dict(tile_k=128)),
    (256, dict(block_m=128, block_n=256)),
], ids=["tk128", "bm128_bn256"])
def test_tile_variants(inter, options):
    """非默认 tile：TILE_K=128（delta 字节按 tile 号动态选取）、BLOCK_N=256（每个 wave 两个 16 行块）、BLOCK_M=128。

    基本测试与 bench 只走默认 tile（BLOCK_M=64、BLOCK_N=128、TILE_K=256）。
    """
    w = oracle.make_weights(33, HIDDEN, inter, seed=36)
    x, topk_weight, topk_ids = oracle.make_inputs(300, HIDDEN, 33, TOPK, seed=37)
    oracle.assert_close(*_run(w, _prepare(w), x, topk_weight, topk_ids, **options))


def _pack(nibbles):
    return (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).view(torch.float4_e2m1fn_x2)


def test_delta_flush_boundary():
    """每行一个 32-K 组比该行最大 E8M0 小 11（仍精确）或 12（按约定当作 0），输入只落在这个组的 K 上。

    随机权重同一行的 E8M0 只差 0–3，基本测试与 bench 碰不到 delta=11/12 的边界；本例的 E2M1 码覆盖全部
    16 个（含 -0）。期望与“delta>11 的组清零”的参考一致，并明显偏离精确参考（说明清零确实发生）。
    """
    E, H, I, topk = 9, 1024, 64, 3
    gen = torch.Generator(device="cuda").manual_seed(41)
    w = oracle.make_weights(E, H, I, seed=41)
    rows, groups, top, probe = 2 * I, H // 32, 3, 17
    delta = torch.where(torch.arange(rows) % 2 == 0, 11, 12)
    codes = torch.full((E, rows, groups), 125 - 20, dtype=torch.uint8)
    codes[:, :, top] = 125
    codes[:, :, probe] = (125 - delta).to(torch.uint8)
    nibbles = torch.zeros((E, rows, H), dtype=torch.uint8)
    for g in (top, probe):
        nibbles[:, :, g * 32:(g + 1) * 32] = torch.randint(0, 16, (E, rows, 32), dtype=torch.uint8,
                                                          generator=gen)
    flushed = nibbles.clone()
    flushed[:, delta > 11, probe * 32:(probe + 1) * 32] = 0
    codes = codes.view(torch.float8_e8m0fnu)
    x, topk_weight, topk_ids = oracle.make_inputs(64, H, E, topk, seed=42)
    probe_x = torch.zeros_like(x)
    probe_x[:, probe * 32:(probe + 1) * 32] = x[:, probe * 32:(probe + 1) * 32]
    exact = dict(w, w1=_pack(nibbles), w1_scale=codes)
    out, ref_exact = _run(exact, _prepare(exact), probe_x, topk_weight, topk_ids)
    mode = oracle.path_activation(moe_a8w4.last_path)
    ref_flushed = oracle.reference(probe_x, _pack(flushed), codes, w["w2"], w["w2_scale"], topk_weight,
                                   topk_ids, activation=mode)
    oracle.assert_close(out, ref_flushed)
    assert not oracle.check(out, ref_exact)["ok"]


def _decode(q, d, s, N, K):
    """按 a8w4_common 文档的布局把 (q, d, s) 解码成 float64 [E, N, K]。"""
    E, KB = s.shape[0], K // 64
    KQ = (KB + 3) // 4
    b = q.view(E, N // 16, KB, 4, 16, 2, 4).permute(0, 1, 4, 2, 3, 5, 6).reshape(E, N, KB, 4, 2, 4)
    nib = torch.cat([b & 0xF, b >> 4], dim=-1).reshape(E, N, K).long()
    table = torch.tensor(oracle._E2M1, dtype=torch.float64)
    dl = d.view(E, N // 16, KQ, 2, 16, 4).permute(0, 1, 4, 2, 5, 3).reshape(E, N, KQ * 4, 2)[:, :, :KB]
    dl = dl.reshape(E, N, KB * 2).long()
    assert bool((dl % 8 == 0).all())
    factor = torch.exp2(5 - (dl // 8).double()).repeat_interleave(32, dim=-1)
    return table[nib] * factor * s.view(E, N, 1).double(), dl // 8


def test_prepare_layout_bitexact():
    """按 a8w4_common 文档的字节布局解码 prepare 结果，与 MXFP4 精确值逐元素相等（delta 0–15、-0、I=160→192 补零、通道因子）。

    基本测试只看端到端误差：布局与文档不符但 kernel 恰好按同样方式读取时照样通过；以后的 kernel 依赖这份布局约定。
    """
    E, H, I = 2, 512, 160
    gen = torch.Generator(device="cuda").manual_seed(71)

    def weight(n, k):
        codes = (120 - torch.randint(0, 16, (E, n, k // 32), generator=gen)).to(torch.uint8)
        codes[:, :, 0] = 120
        nibbles = torch.randint(0, 16, (E, n, k), dtype=torch.uint8, generator=gen)
        return _pack(nibbles), codes.view(torch.float8_e8m0fnu)

    w1, s1 = weight(2 * I, H)
    w2, s2 = weight(H, I)
    p = moe_a8w4.prepare_a8w4_weights(w1, s1, w2, s2)
    assert (p.inter, p.inter_padded) == (I, 192)

    def expected(q, s, n, k):
        codes = s.view(torch.uint8).long()
        value = oracle.dequant_mxfp4(q, s).double()
        diff = codes.amax(-1, keepdim=True) - codes
        value = value.masked_fill((diff > 11).repeat_interleave(32, dim=-1), 0.0)
        scale = torch.exp2(codes.amax(-1).double() - 132).float()
        return value, diff.clamp(max=11), scale

    v1, d1, c1 = expected(w1, s1, 2 * I, H)
    z1 = torch.zeros((E, 192 - I, H), dtype=torch.float64)
    got, delta = _decode(p.w1, p.w1_delta, p.w1_scale, 2 * 192, H)
    assert torch.equal(got, torch.cat([v1[:, :I], z1, v1[:, I:], z1], dim=1))
    assert torch.equal(delta[:, :I], d1[:, :I]) and torch.equal(delta[:, 192:192 + I], d1[:, I:])
    assert torch.equal(p.w1_scale.view(E, -1)[:, :I], c1[:, :I])

    v2, d2, c2 = expected(w2, s2, H, I)
    got, delta = _decode(p.w2, p.w2_delta, p.w2_scale, H, 192)
    assert torch.equal(got, torch.nn.functional.pad(v2, (0, 192 - I)))
    assert torch.equal(delta[:, :, :I // 32], d2)
    assert torch.equal(p.w2_scale.view(E, -1), c2)


def test_graph_replay():
    """整个调用可被 CUDA graph 捕获并重放（无 host 同步），重放读到的是新写入的输入。

    基本测试与 bench 都是 eager 调用，不检查 graph 捕获。
    """
    w = oracle.make_weights(33, HIDDEN, 320, seed=51)
    prepared = _prepare(w)
    x, topk_weight, topk_ids = oracle.make_inputs(64, HIDDEN, 33, TOPK, seed=52)
    out = torch.empty_like(x)
    moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out)  # 在捕获外编译
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out)
    mode = oracle.path_activation(moe_a8w4.last_path)
    x2, topk_weight2, topk_ids2 = oracle.make_inputs(64, HIDDEN, 33, TOPK, seed=53)
    x.copy_(x2)
    topk_weight.copy_(topk_weight2)
    topk_ids.copy_(topk_ids2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    ref = oracle.reference(x2, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight2, topk_ids2,
                           activation=mode)
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("dtype, cols", [("bf16", 2560), ("f32", 320)])
def test_row_quant_bitexact(dtype, cols):
    """激活的行量化（scale = amax/240，q = RNE(x/scale)）与参考逐位一致，含全零行与 BF16 常见的精确中点。

    基本测试只看端到端误差：量化算术的微小差异（例如乘倒数）只让少量 FP8 值翻转，小 M 时判据未必发现，
    却会吃掉大 M 下 2% 最大误差的余量。
    """
    gen = torch.Generator(device="cuda").manual_seed(81)
    x = torch.randn((997, cols), generator=gen) * 3
    x[5].zero_()
    x = x.to(torch.bfloat16) if dtype == "bf16" else x
    q, scale = moe_a8w4._quantize(x, x.shape[0], cols, dtype)
    ref = oracle.quant_rows_fp8(x)
    amax = x.float().abs().amax(-1, keepdim=True)
    ref_scale = torch.where(amax == 0, torch.ones_like(amax), amax / torch.full_like(amax, 240.0))
    assert torch.equal(scale.view(-1, 1), ref_scale)
    assert torch.equal(q.float() * scale.view(-1, 1), ref)
    assert int((q.view(torch.uint8) == 0x80).sum()) == 0


def test_rejects_bad_inputs():
    """不支持的输入直接报错：权重 / scale dtype、形状、E8M0 的 NaN 码、H 或 I 不对齐、非 BF16 激活、hidden 不匹配，
    以及 Down 的 BLOCK_M·I_pad 不是 4096 的倍数（BM32 配 I320 / 192，构建 kernel 时报错而不是静默算错）。

    基本测试与 bench 只给合法输入，且只用默认 block_m=64。
    """
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.a8w4_gemm2 import compile_a8w4_gemm2

    w = oracle.make_weights(3, 512, 64, seed=61)
    args = [w["w1"], w["w1_scale"], w["w2"], w["w2_scale"]]

    def prepare_with(index, value):
        bad = list(args)
        bad[index] = value
        return moe_a8w4.prepare_a8w4_weights(*bad)

    with pytest.raises(TypeError):
        prepare_with(0, w["w1"].view(torch.uint8).view(torch.int8))
    with pytest.raises(TypeError):
        prepare_with(1, w["w1_scale"].view(torch.uint8).float())
    with pytest.raises(ValueError):
        prepare_with(2, w["w2"][:, :, :16])
    with pytest.raises(ValueError):
        prepare_with(0, w["w1"][:, :96])  # I=48 不是 32 的倍数
    nan_scale = w["w2_scale"].view(torch.uint8).clone()
    nan_scale[0, 0, 0] = 255
    with pytest.raises(ValueError):
        prepare_with(3, nan_scale)
    small = oracle.make_weights(3, 256, 64, seed=62)
    with pytest.raises(ValueError):
        moe_a8w4.prepare_a8w4_weights(small["w1"], small["w1_scale"], small["w2"], small["w2_scale"])
    prepared = moe_a8w4.prepare_a8w4_weights(*args)
    x, topk_weight, topk_ids = oracle.make_inputs(4, 512, 3, 2, seed=63)
    with pytest.raises(TypeError):
        moe_a8w4.fused_moe_a8w4(x.half(), prepared, topk_weight, topk_ids)
    with pytest.raises(ValueError):
        moe_a8w4.fused_moe_a8w4(x[:, :256].contiguous(), prepared, topk_weight, topk_ids)
    for inter in (320, 192):
        with pytest.raises(AssertionError, match="4096"):
            compile_a8w4_gemm2(N=HIDDEN, K=inter, TOPK=TOPK, BLOCK_TILE_SIZE_M=32)
