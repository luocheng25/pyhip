"""A8W4 MoE 的其他覆盖测试。

只放基本功能测试（test_moe_a8w4.py）与 bench（benchmarks/moe/bench_moe_a8w4.py）覆盖不了的情况，
每个测试的 docstring 写明原因。端到端判据与基本测试相同（``pyhip.testing.moe_a8w4.check``，
参考的激活精度按实际启动的 kernel 选择）；逐位用例要求 0 误差。部分填充的尾块、多块 expert 与 I=160→192
补零已由基本测试覆盖，E=513 的各个 M 档由 bench 覆盖，这里不重复。
M≤32 的 decode 路径：核内量化（M=3）由基本测试、M=1 / 16 的三种分派由 bench 覆盖，这里只测分派交界、
topk 与路由输入的变体、CUDA graph、decode 行量化逐位与 w2≥4 GiB 回退（用例名带 ``decode``）。
33≤M≤2048 的流式路径：排序块 64 由基本测试、16 / 32 由 bench 覆盖，这里只测与 prefill 的交界、两处量化逐位
（用例名带 ``stream``）以及 grid 收缩与 graph；非目标形状的回退见 ``test_shape_fallbacks``。
"""

import dataclasses
import functools

import pytest
import torch

from pyhip.ops.moe.flydsl import moe_a8w4
from pyhip.testing import moe_a8w4 as oracle

HIDDEN, TOPK = 2560, 11
# decode 三种分派（x、mid 是否先用独立 kernel 预量化）实际启动的 kernel 序列。
_DECODE_PATHS = {
    "nn": ["a8w4_gateup_decode", "a8w4_down_decode"],
    "np": ["a8w4_gateup_decode", "a8w4_row_quant_decode", "a8w4_down_decode_pq"],
    "pp": ["a8w4_row_quant_decode", "a8w4_gateup_decode_pq", "a8w4_row_quant_decode", "a8w4_down_decode_pq"],
}
# prefill（默认 tile）实际启动的 kernel 序列。
_PREFILL_PATH = ["moe_sorting_opus", "a8w4_row_quant_fp8", "a8w4_gateup_prefill_bm64_bn128_tk256", "a8w4_row_quant_fp8",
                 "a8w4_down_prefill_bm64", "sorted_sum"]


def _stream_path(block_m, fused):
    """流式路径实际启动的 kernel 序列：``fused`` 时 Gate/Up 在尾部量化中间激活，否则另起行量化。"""
    gateup = [f"a8w4_gateup_quant_stream_bm{block_m}"] if fused else [f"a8w4_gateup_stream_bm{block_m}",
                                                                      "a8w4_row_quant_fp8"]
    return ["moe_sorting_opus", "a8w4_row_quant_bf16", *gateup, f"a8w4_down_stream_bm{block_m}", "a8w4_route_sum"]


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


@functools.cache
def _decode_weights(inter):
    """decode 与流式用例共用的 E=33 权重（按 I 缓存：生成一份约 3 秒）。"""
    w = oracle.make_weights(33, HIDDEN, inter, seed=inter)
    return w, _prepare(w)


def test_grid_shrink_boundary_e513():
    """E=513 满规模下流式 Gate/Up、Down 的 grid 收缩（M·topk <= E 时 task 数取 M·topk）交界两侧：M=46（506）、M=47（517）。

    基本测试用 E=33；bench 的 M 档（1、16、128…）离这个交界都很远。这两个 M 的路由数少于 4E/3，排序块取 32
    （aiter 对 block 16 会换用另一种排序），也覆盖了这条分支与大量空 expert。
    """
    w = oracle.make_weights(513, HIDDEN, 320, seed=31)
    prepared = _prepare(w)
    for tokens in (46, 47):
        x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 513, TOPK, seed=32 + tokens)
        oracle.assert_close(*_run(w, prepared, x, topk_weight, topk_ids))
        assert moe_a8w4.last_path == _stream_path(32, True)


@pytest.mark.parametrize("inter, options", [
    (320, dict(tile_k=128)),
    (256, dict(block_m=128, block_n=256)),
], ids=["tk128", "bm128_bn256"])
def test_tile_variants(inter, options):
    """给了 tile 参数时走 prefill（M=300 本在流式范围内）：TILE_K=128（delta 字节按 tile 号动态选取）、
    BLOCK_N=256（每个 wave 两个 16 行块）、BLOCK_M=128。

    基本测试与 bench 只走默认 tile（BLOCK_M=64、BLOCK_N=128、TILE_K=256）。
    """
    w = oracle.make_weights(33, HIDDEN, inter, seed=36)
    x, topk_weight, topk_ids = oracle.make_inputs(300, HIDDEN, 33, TOPK, seed=37)
    oracle.assert_close(*_run(w, _prepare(w), x, topk_weight, topk_ids, **options))
    assert any(name.startswith("a8w4_gateup_prefill_") for name in moe_a8w4.last_path), moe_a8w4.last_path


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


@pytest.mark.parametrize("options, path", [({}, _stream_path(32, True)), (dict(block_m=64), _PREFILL_PATH)],
                         ids=["stream", "prefill"])
def test_graph_replay(options, path):
    """整个调用可被 CUDA graph 捕获并重放（无 host 同步），重放读到的是新写入的输入：M=64 默认走流式路径
    （排序块 32、Gate/Up 尾部量化），给 tile 参数时走 prefill。

    基本测试与 bench 都是 eager 调用，不检查 graph 捕获。
    """
    w = oracle.make_weights(33, HIDDEN, 320, seed=51)
    prepared = _prepare(w)
    x, topk_weight, topk_ids = oracle.make_inputs(64, HIDDEN, 33, TOPK, seed=52)
    out = torch.empty_like(x)
    moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out, **options)  # 在捕获外编译
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out, **options)
    assert moe_a8w4.last_path == path
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


@pytest.mark.parametrize("inter, tokens, path", [
    (320, 4, "np"), (320, 11, "np"), (320, 12, "pp"), (320, 32, "pp"), (320, 33, "stream"),
    (160, 19, "np"), (160, 20, "pp"),
], ids=["tp2_m4", "tp2_m11", "tp2_m12", "tp2_m32", "tp2_m33", "tp4_m19", "tp4_m20"])
def test_decode_dispatch_boundaries(inter, tokens, path):
    """decode 分派各交界的端到端结果与实际 kernel 序列：M=4（Down 改用预量化槽对 kernel；M=3 一侧是基本测试）、
    TP2 M=11/12 与 TP4 M=19/20（Gate/Up 的 WG 数 M·topk·I/16 跨过 2560，改读预量化的 x）、M=32/33（改走排序的
    流式路径；E=33 时每个 expert 平均 11 行，排序块取 16）。

    基本测试的 decode 只有 M=3，bench 只有 M=1、16：TP2 的 Down 预量化、TP4 的全预量化与这些交界都碰不到。
    """
    w, prepared = _decode_weights(inter)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 33, TOPK, seed=90 + tokens)
    out, ref = _run(w, prepared, x, topk_weight, topk_ids)
    if path == "stream":
        assert moe_a8w4.last_path == _stream_path(16, False)
    else:
        assert moe_a8w4.last_path == _DECODE_PATHS[path]
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("topk, tokens, path", [(8, 16, "pp"), (17, 2, "nn")], ids=["topk8_m16", "topk17_m2"])
def test_decode_topk_variants(topk, tokens, path):
    """topk≠11：topk=8 是偶数，槽对 Down 3 个 wave 两轮、第二轮两个 wave 的槽对整对无效，且 M=16 时 Gate/Up 恰为
    2560 个 WG（按 >= 走预量化）；topk=17 大于 16，核内量化的 Down 改为 4 个 wave 轮流取槽，末轮 3 个槽无效。

    基本测试与 bench 都是 topk=11：槽对只有末对缺高半，核内量化的 Down 每个槽一个 wave。
    """
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 33, topk, seed=94 + topk)
    out, ref = _run(w, prepared, x, topk_weight, topk_ids)
    assert moe_a8w4.last_path == _DECODE_PATHS[path]
    oracle.assert_close(out, ref)


def test_decode_duplicate_and_int64_ids():
    """同一 token 的几个槽选同一 expert（含同一槽对的两半），topk_ids 为 int64 且不连续、topk_weight 不连续（M=5）：
    decode 路径在 host 侧转成连续的 int32 / f32 再 launch。

    输入生成的路由互不重复，且是连续的 int32 / f32：基本测试与 bench 都碰不到。
    """
    w, prepared = _decode_weights(320)
    tokens = 5
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 33, TOPK, seed=96)
    topk_ids[:, 1] = topk_ids[:, 0]
    topk_ids[:, 4] = topk_ids[:, 0]
    ids = torch.empty((tokens, 2 * TOPK), dtype=torch.int64)[:, ::2]
    ids.copy_(topk_ids)
    weights = torch.empty((tokens, 2 * TOPK))[:, 1::2]
    weights.copy_(topk_weight)
    assert not ids.is_contiguous() and not weights.is_contiguous()
    out, ref = _run(w, prepared, x, weights, ids)
    assert moe_a8w4.last_path == _DECODE_PATHS["np"]
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("tokens, path", [(1, "nn"), (5, "np"), (16, "pp")], ids=["m1_nn", "m5_np", "m16_pp"])
def test_decode_graph_replay(tokens, path):
    """decode 路径的三种分派都可被 CUDA graph 捕获并重放：重放读到新写入的输入（路径内分配的 xq / mid / mq 由 graph 持有）。

    基本测试与 bench 都是 eager 调用；test_graph_replay 只测流式与 prefill。
    """
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 33, TOPK, seed=97 + tokens)
    out = torch.empty_like(x)
    moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out)  # 在捕获外编译
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out)
    assert moe_a8w4.last_path == _DECODE_PATHS[path]
    x2, topk_weight2, topk_ids2 = oracle.make_inputs(tokens, HIDDEN, 33, TOPK, seed=98 + tokens)
    x.copy_(x2)
    topk_weight.copy_(topk_weight2)
    topk_ids.copy_(topk_ids2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    ref = oracle.reference(x2, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight2, topk_ids2,
                           activation="fp8")
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("dtype, cols", [("f32", 192), ("f32", 320), ("f32", 640),
                                         ("bf16", 1024), ("bf16", 2560), ("bf16", 4096)])
def test_decode_row_quant_bitexact(dtype, cols):
    """decode 路径的行量化 kernel 与 a8w4_quant、参考逐位一致（q 字节与 scale 位），且不写 rows 以外的行。

    特殊行：全零、amax/240 下溢成 0（scale 取 1，只有 f32 能表示）、次正规 scale、-0 与舍入成 -0 的负数（存 +0）；
    f32 192 / 320 / 640 列每行 48 / 80 / 160 个 16 B 片（末轮不满 64 片），bf16 1024 / 2560 / 4096 列；
    37 行（最后一个 WG 只有 1 行，其后 3 行不得被写）。端到端判据看不出少量 fp8 翻转，基本测试与 bench
    只有 192 / 320 / 2560 列的随机数据。
    """
    rows, guard = 37, 3
    gen = torch.Generator(device="cuda").manual_seed(83)
    x = torch.randn((rows, cols), generator=gen) * 3
    x[1].zero_()
    x[2].zero_()
    x[2, 7] = 1e-45
    x[3] *= 1e-38
    x[4].zero_()
    x[4, 0] = 200.0
    x[4, 1:9] = -1e-6
    x[4, 9] = -0.0
    x = x.to(torch.bfloat16) if dtype == "bf16" else x
    q = torch.full((rows + guard, cols), 0x5A, dtype=torch.uint8)
    scale = torch.full((rows + guard,), -1.0)
    moe_a8w4._launch("a8w4_row_quant_decode", moe_a8w4._row_quant_decode(cols, dtype == "bf16"), x,
                     q.view(torch.float8_e4m3fnuz), scale, rows)
    assert bool((q[rows:] == 0x5A).all()) and bool((scale[rows:] == -1.0).all())
    q0, scale0 = moe_a8w4._quantize(x, rows, cols, dtype)
    assert torch.equal(q[:rows], q0.view(torch.uint8))
    assert torch.equal(scale[:rows].view(torch.int32), scale0.view(torch.int32))
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    ref_scale = amax / torch.full_like(amax, 240.0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)
    assert torch.equal(scale[:rows].view(-1, 1), ref_scale)
    assert torch.equal(q[:rows], (xf / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8))


def test_decode_down_fallback_w2_4gib():
    """w2 总字节数 >= 4 GiB 时，Down 不用预量化槽对 kernel（它用 32 位字节偏移寻址全部 expert），改走单槽核内量化 Down。

    真实 4 GiB 权重太大：用 stride 0 的视图让 w2 报告 2^32 字节（数据指针不变，kernel 只按指针与 expert 偏移读），
    验证 M=16 时的分派与单槽 Down 的结果；基本测试、bench 与其余用例的 w2 都远小于 4 GiB。
    """
    w, prepared = _decode_weights(320)
    big = dataclasses.replace(prepared, w2=prepared.w2.view(-1)[:1].expand(1 << 32))
    x, topk_weight, topk_ids = oracle.make_inputs(16, HIDDEN, 33, TOPK, seed=99)
    out, ref = _run(w, big, x, topk_weight, topk_ids)
    assert moe_a8w4.last_path == ["a8w4_row_quant_decode", "a8w4_gateup_decode_pq", "a8w4_down_decode"]
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("tokens, path", [(2048, _stream_path(64, False)), (2049, _PREFILL_PATH)],
                         ids=["m2048", "m2049"])
def test_stream_dispatch_boundary(tokens, path):
    """流式路径的上界：M=2048 仍走流式（E=33 时每个 expert 平均约 680 行，排序块 64；I=320 时 4 个 wave 的累加器
    放不下整段 I，中间激活另起行量化），M=2049 改走 prefill。

    基本测试的流式 M=1031、prefill M=2053，bench 是 1024 与 4096，都不在交界上。
    """
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 33, TOPK, seed=tokens)
    out, ref = _run(w, prepared, x, topk_weight, topk_ids)
    assert moe_a8w4.last_path == path
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("cols", [512, 2560])
def test_stream_row_quant_bitexact(cols):
    """流式路径的输入量化 kernel（a8w4_row_quant_bf16：每 wave 一行，16 B 读、8 B 写）与 a8w4_quant、参考逐位一致
    （q 字节与 scale 位），且不写 rows 以外的行。

    特殊行：全零、次正规 scale（amax/240 < 2^-126）、-0 与舍入成 -0 的负数（存 +0）、商恰为 FP8 舍入中点（RNE 取偶）；
    512 列每 lane 一片、2560 列五片；37 行（最后一个 WG 只有 1 行）。端到端判据看不出少量 fp8 翻转，
    基本测试与 bench 只有随机数据。
    """
    rows, guard = 37, 3
    gen = torch.Generator(device="cuda").manual_seed(85)
    x = torch.randn((rows, cols), generator=gen) * 3
    x[1].zero_()
    x[2] *= 1e-37
    x[3].zero_()
    x[3, 0] = 200.0
    x[3, 1:9] = -1e-6
    x[3, 9] = -0.0
    # 第 4 行 amax = 240·2^-2，scale 恰为 2^-2；其余元素的商 (k + 0.5)·2^e 恰在两个相邻 FP8 值的正中。
    exponent = torch.randint(-10, 1, (cols,), generator=gen).float()
    tie = (torch.randint(8, 16, (cols,), generator=gen).float() + 0.5) * torch.exp2(exponent)
    x[4] = torch.where(torch.rand((cols,), generator=gen) < 0.5, -tie, tie) * 0.25
    x[4, 0] = 60.0
    x = x.to(torch.bfloat16)
    q = torch.full((rows + guard, cols), 0x5A, dtype=torch.uint8)
    scale = torch.full((rows + guard,), -1.0)
    moe_a8w4._launch("a8w4_row_quant_bf16", moe_a8w4._stream_quant_kernel(cols), x, q.view(torch.float8_e4m3fnuz),
                     scale, rows)
    assert bool((q[rows:] == 0x5A).all()) and bool((scale[rows:] == -1.0).all())
    q0, scale0 = moe_a8w4._quantize(x, rows, cols, "bf16")
    assert torch.equal(q[:rows], q0.view(torch.uint8))
    assert torch.equal(scale[:rows].view(torch.int32), scale0.view(torch.int32))
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    ref_scale = amax / torch.full_like(amax, 240.0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)
    assert float(ref_scale[2]) < 2.0 ** -126 and float(ref_scale[4]) == 0.25
    assert torch.equal(scale[:rows].view(-1, 1), ref_scale)
    assert torch.equal(q[:rows], (xf / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8))


@pytest.mark.parametrize("inter, block_m", [(320, 32), (160, 64)], ids=["tp2_bm32", "tp4_bm64"])
def test_stream_fused_quant_bitexact(inter, block_m):
    """流式 Gate/Up 尾部量化的中间激活（倒数乘法 + 1∓2^-20 夹逼判定；有元素靠近舍入边界、或 scale 不是有限正规数时
    整个 wave 回退 IEEE 除法）与同一排序下写 f32 再用 a8w4_quant 量化的结果逐位一致（q 字节与 scale 位）。
    配置取分派实际用的：TP2 排序块 32（每 wave 5 个 gate 块、不共享激活）、TP4 排序块 64（3 个 gate 块、LDS 共享激活）。

    M=1031 的随机行里有商落在舍入边界 2^-20 以内的元素（下面断言存在，回退分支确实跑到）；另把 6 个 token 的 x 缩小
    1e-17…1e-23 倍或置零，使中间行的 scale 为很小的正规数、次正规数与 0（取 1）。端到端判据看不出少量 fp8 翻转。
    """
    from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting

    from pyhip.ops.moe.flydsl.moe_gemm_2stage.a8w4_stream import compile_a8w4_gateup_stream

    w, prepared = _decode_weights(inter)
    E, I, M = prepared.experts, prepared.inter_padded, 1031
    assert moe_a8w4._stream_fused_quant(I, block_m)
    x, topk_weight, topk_ids = oracle.make_inputs(M, HIDDEN, E, TOPK, seed=120)
    for row, factor in enumerate((1e-17, 1e-18, 3e-19, 1e-20, 1e-23, 0.0)):
        x[row] = (x[row].float() * factor).to(torch.bfloat16)
    sorted_ids, _, sorted_experts, num_valid, _, _, _ = moe_sorting(
        topk_ids, topk_weight, E, HIDDEN, x.dtype, block_m, accumulate=False, output_aux=AUX_SORT_OPUS)
    routing, task_num = (sorted_ids, sorted_experts, num_valid), sorted_experts.numel()
    xq, xs = moe_a8w4._quantize(x, M, HIDDEN, "bf16")
    mq = torch.empty((M * TOPK, I), dtype=torch.float8_e4m3fnuz)
    ms = torch.empty((M * TOPK,))
    moe_a8w4._launch("fused", moe_a8w4._stream_gateup_kernel(2 * I, HIDDEN, TOPK, block_m, E), xq, prepared.w1,
                     prepared.w1_delta, mq, *routing, prepared.w1_scale, xs, ms, M, task_num)
    plain = compile_a8w4_gateup_stream(N=2 * I, K=HIDDEN, TOPK=TOPK, BLOCK_M=block_m, E=E, G=4, W=1, PREFETCH=(6, 4))
    mid = torch.empty((M, TOPK, I))
    moe_a8w4._launch("plain", plain, xq, prepared.w1, prepared.w1_delta, mid, *routing, prepared.w1_scale, xs, xs,
                     M, task_num)
    q0, s0 = moe_a8w4._quantize(mid, M * TOPK, I, "f32")
    assert torch.equal(mq.view(torch.uint8), q0.view(torch.uint8))
    assert torch.equal(ms.view(torch.int32), s0.view(torch.int32))
    assert bool(((s0 > 0) & (s0 < 2.0 ** -126)).any())
    quotient = mid.view(M * TOPK, I) / s0.view(-1, 1)
    ends = [(quotient * (1.0 + d)).clamp(-240.0, 240.0).to(torch.float8_e4m3fnuz).view(torch.uint8)
            for d in (-2.0 ** -20, 2.0 ** -20)]
    assert bool((ends[0] != ends[1]).any())


@functools.cache
def _small_weights(hidden, inter):
    """形状回退用例共用的 E=5 权重（按 H、I 缓存）。"""
    w = oracle.make_weights(5, hidden, inter, seed=hidden + inter)
    return w, _prepare(w)


@pytest.mark.parametrize("hidden, inter, tokens, path, shared", [
    (4096, 64, 8, _DECODE_PATHS["np"], None),
    (4608, 64, 8, _PREFILL_PATH, None),
    (512, 1024, 4, _DECODE_PATHS["np"], None),
    (512, 1088, 4, _DECODE_PATHS["nn"], None),
    (512, 1024, 64, _stream_path(32, False), True),
    (512, 1024, 300, _stream_path(64, False), False),
], ids=["h4096_decode", "h4608_prefill", "i1024_pair_down", "i1088_single_down", "i1024_bm32_shared",
        "i1024_bm64_unshared"])
def test_shape_fallbacks(hidden, inter, tokens, path, shared):
    """非目标形状按 kernel 上限回退，结果仍对：H<=4096 才走 decode（Gate/Up decode 每线程一片 16 B，至多 256 片），
    H=4608 的 M<=32 改走 prefill；槽对 Down 要把每行 I/16 片放进 64·WAVES 个 lane（topk=2 时 1 个 wave），I=1088
    改用单槽 Down；流式 Down 只在共享激活行放得进 64 KiB LDS 时让 4 个 wave 共享（I=1024：排序块 32 放得下，64 放不下，
    后者若误选共享 tile 会在构建 kernel 时报错）。

    前两类只有目标形状（H=2560，I=320 / 192），碰不到这些上限；E=5、topk=2 控制耗时。
    """
    from pyhip.ops.moe.flydsl.moe_gemm_2stage import a8w4_common, a8w4_stream

    w, prepared = _small_weights(hidden, inter)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, hidden, 5, 2, seed=tokens)
    out, ref = _run(w, prepared, x, topk_weight, topk_ids)
    assert moe_a8w4.last_path == path
    oracle.assert_close(out, ref)
    if shared is not None:
        block_m = moe_a8w4._stream_block_m(tokens, 2, 5)
        assert (a8w4_stream.down_stream_lds(inter, block_m, W=4) <= a8w4_common.LDS_BYTES) == shared
