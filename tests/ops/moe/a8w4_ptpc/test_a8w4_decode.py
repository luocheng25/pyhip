"""A8W4（mxfp4 权重）MoE 的 M1：权重准备的逐位格式与 W4A16 decode 档。

形状、输入与容差见 ``pyhip.testing.moe_a8w4_ptpc``，mxfp4 权重与参考见 ``pyhip.testing.moe_a8w4_mxfp4``：
TP2 每卡 I=320，TP4 每卡 I=160（``prepare_weights`` 补零到 192）。参考权重是 mxfp4 精确反量化并施加
kernel 的 flush 规则（比该行最大 code 小 12 以上的组为 0），激活 BF16，FP32 计算；容差用 M0 的
``check_output`` 原样。

- prepare（CPU，小形状）：解回的权重与“反量化 + flush + 补零”逐位相等；fd1 nibble 顺序、-0 改 +0、
  flush 置零、delta 字节、通道因子（含次正规数）、补零行列、E8M0=255 与形状报错。
- decode：TP2 / TP4 × decode 档内的 T（1、8、16、32 和配置表的 decode 上界，当前 128），断言实际编译的
  两级 kernel 就是配置表这一档（batch1 gn32 / batch1 gn64 / split-K）。
- CUDA graph：三种 decode 配置各捕获一次，replay 时换入新的输入与路由。
- prefill 档（decode 上界 + 1 起，M2 实现）见 ``test_a8w4_prefill.py``。
"""

import pytest
import torch

from pyhip.ops.moe.flydsl import a8w4_ptpc as A
from pyhip.testing import moe_a8w4_mxfp4 as X
from pyhip.testing import moe_a8w4_ptpc as M

DECODE_MAX = A.decode_max_tokens()
DECODE_TOKENS = sorted({t for t in (1, 8, 16, 32) if t < DECODE_MAX} | {DECODE_MAX})
MATRIX = [(tp, tokens) for tp in (2, 4) for tokens in DECODE_TOKENS]
GRAPH_CASES = [(2, 1), (2, 32), (4, 8), (4, DECODE_MAX)]


@pytest.fixture
def gpu(monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("A8W4 decode kernels target MI308X (gfx942)")
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
            cache[tp] = X.make_mxfp4_weights(tp, seed=10 + tp, device="cuda")
        return cache[tp]

    yield get
    cache.clear()


# ---------------------------------------------------------------- prepare_weights（CPU）

E_SMALL, H_SMALL, I_SMALL = 2, 256, 96  # I 补零到 128


def _small_raw(seed=0):
    gen = torch.Generator().manual_seed(seed)
    w1 = torch.randint(0, 256, (E_SMALL, 2 * I_SMALL, H_SMALL // 2), generator=gen).to(torch.uint8)
    w2 = torch.randint(0, 256, (E_SMALL, H_SMALL, I_SMALL // 2), generator=gen).to(torch.uint8)
    s1 = X.random_codes((E_SMALL, 2 * I_SMALL, H_SMALL // 32), gen=gen, device="cpu")
    s2 = X.random_codes((E_SMALL, H_SMALL, I_SMALL // 32), gen=gen, device="cpu")
    # 第 0 行：nibble k = k % 16（16 个编码各出现，含 -0），组 code 依次为 c_max - 0..7，第 1 行全是 -0。
    w1[0, 0] = torch.tensor([(2 * j % 16) | ((2 * j + 1) % 16) << 4 for j in range(H_SMALL // 2)])
    s1[0, 0] = torch.tensor([200 - g for g in range(H_SMALL // 32)])
    w1[0, 1] = 0x88
    # 极端 code：c_max = 0、5（因子为次正规数）、6、254；一行里有 delta 11 / 12 的边界。
    for row, codes in ((2, [0] * 8), (3, [5, 0, 1, 2, 3, 4, 5, 5]), (4, [6] * 8), (5, [254, 243, 242, 254] * 2)):
        s1[1, row] = torch.tensor(codes)
    return w1, s1, w2, s2


def _expected_gemm(w, codes, rows, cols):
    """反量化 + flush，再按补零后的行 / 列位置放置。"""
    ref = X.dequant_mxfp4(w, codes)
    out = torch.zeros((ref.shape[0], len(rows), cols), dtype=torch.float32)
    for dst, src in enumerate(rows):
        if src is not None:
            out[:, dst, :ref.shape[-1]] = ref[:, src]
    return out


def test_prepare_matches_dequant_bitwise():
    raw = _small_raw()
    prep = A.prepare_weights(*raw)
    pad = prep.inter_padded
    assert (prep.inter, pad, prep.experts, prep.hidden) == (I_SMALL, 128, E_SMALL, H_SMALL)
    assert prep.w1.shape == (E_SMALL, 2 * pad // 16, H_SMALL // 16, 16, 8)
    assert prep.w1_delta.shape == (E_SMALL, 2 * pad // 64, H_SMALL // 32, 16, 4)
    assert prep.w2.shape == (E_SMALL, H_SMALL // 16, pad // 16, 16, 8)
    assert prep.w2_delta.shape == (E_SMALL, H_SMALL // 64, pad // 32, 16, 4)
    rows1 = list(range(I_SMALL)) + [None] * (pad - I_SMALL) + list(range(I_SMALL, 2 * I_SMALL)) + [None] * (pad - I_SMALL)
    got1 = X.decode_prepared(prep.w1, prep.w1_delta, prep.w1_factor)
    assert torch.equal(got1, _expected_gemm(raw[0], raw[1], rows1, H_SMALL))
    got2 = X.decode_prepared(prep.w2, prep.w2_delta, prep.w2_factor)
    assert torch.equal(got2, _expected_gemm(raw[2], raw[3], list(range(H_SMALL)), pad))
    # 补零的通道：nibble 全 0、因子为 1；w2 补的 K 列 nibble 全 0。
    pad_rows = [r for r, src in enumerate(rows1) if src is None]
    assert not got1[:, pad_rows].any() and bool((prep.w1_factor[:, pad_rows] == 1).all())
    assert not got2[..., I_SMALL:].any()


def test_prepare_byte_layout():
    raw = _small_raw()
    prep = A.prepare_weights(*raw)
    # 第 0 行前 16 个 K 的 8 字节：每 dword 字节 j = k_j | k_{j+4} << 4，-0（8）改为 0。
    nib = [k % 16 for k in range(16)]
    nib = [0 if v == 8 else v for v in nib]
    expect = [nib[8 * d + j] | nib[8 * d + j + 4] << 4 for d in range(2) for j in range(4)]
    assert prep.w1[0, 0, 0, 0].tolist() == expect
    # 第 1 行全是 -0：全部解成 +0。
    assert not prep.w1[0, 0, :, 1].any()
    # 第 0 行组 g 的 code 是 200 - g：delta 字节 = min(g, 11) * 8；行 0 在 64 行块 0 的字节 0。
    assert prep.w1_delta[0, 0, :, 0, 0].tolist() == [min(g, 11) * 8 for g in range(H_SMALL // 32)]
    # 行 5（专家 1）的组 code 254/243/242：delta 11 保留，delta 12 的组整组写 0（flush），字节记 88。
    row5 = prep.w1_delta[1, 0, :, 5, 0].tolist()
    assert row5 == [0, 88, 88, 0] * 2
    flushed = prep.w1[1, 0, :, 5].reshape(H_SMALL // 32, 16)
    assert not flushed[2::4].any() and not flushed[6::4].any()
    # 通道因子 2^(c_max - 132)：c_max 0 / 5 为次正规数，6 为最小正规数，254 为 2^122。
    factors = prep.w1_factor[1, 2:6].tolist()
    assert factors == [2.0 ** -132, 2.0 ** -127, 2.0 ** -126, 2.0 ** 122]
    assert prep.w1_factor[0, 0].item() == 2.0 ** (200 - 132)


def test_prepare_rejects_invalid():
    w1, s1, w2, s2 = _small_raw()
    bad = s2.clone()
    bad[1, 7, 2] = 255
    with pytest.raises(ValueError, match="255"):
        A.prepare_weights(w1, s1, w2, bad)
    with pytest.raises(ValueError, match="inconsistent"):
        A.prepare_weights(w1, s1[:, :, :-1], w2, s2)
    with pytest.raises(ValueError, match="H % 256"):
        A.prepare_weights(w1[..., :64], s1[..., :4], w2[:, :128], s2[:, :128])
    with pytest.raises(TypeError):
        A.prepare_weights(w1.to(torch.int16), s1, w2, s2)
    # torch 的 fp4x2 / e8m0 dtype 按字节读，结果相同。
    same = A.prepare_weights(w1.view(torch.float4_e2m1fn_x2), s1.view(torch.float8_e8m0fnu),
                             w2.view(torch.float4_e2m1fn_x2), s2.view(torch.float8_e8m0fnu))
    base = A.prepare_weights(w1, s1, w2, s2)
    for field in ("w1", "w1_delta", "w1_factor", "w2", "w2_delta", "w2_factor"):
        assert torch.equal(getattr(same, field), getattr(base, field)), field


# ---------------------------------------------------------------- decode（GPU）

def _record_compiles(monkeypatch):
    from pyhip.ops.moe.flydsl.a8w4_ptpc import moe as entry

    compiled = []
    original = entry.compile_decode_gemm

    def compile_decode_gemm(**params):
        compiled.append((params["stage"], params["alg"], params["BLOCK_TILE_SIZE_N"]))
        return original(**params)

    monkeypatch.setattr(entry, "compile_decode_gemm", compile_decode_gemm)
    return compiled


@pytest.mark.parametrize("tp, tokens", MATRIX, ids=[f"tp{tp}-t{t}" for tp, t in MATRIX])
def test_decode_matches_reference(gpu, shard, monkeypatch, tp, tokens):
    raw, prep = shard(tp)
    config = A.select_config(tokens)
    assert config["path"] == "decode", config
    hidden, weight, ids = M.make_inputs(tokens, seed=tokens)
    output = torch.full_like(hidden, float("nan"))
    compiled = _record_compiles(monkeypatch)
    assert A.fused_moe(hidden, prep, weight, ids, output=output) is output
    assert compiled == [("gateup", config["alg"], config["tile_n_gate"]),
                        ("down", config["alg"], config["tile_n_down"])], compiled
    reference = X.reference_moe(hidden, raw, weight, ids)
    M.check_output(reference, output, f"TP{tp} T={tokens} {config}")


@pytest.mark.parametrize("tp, tokens", GRAPH_CASES, ids=[f"tp{tp}-t{t}" for tp, t in GRAPH_CASES])
def test_decode_cuda_graph(gpu, shard, tp, tokens):
    raw, prep = shard(tp)
    assert A.select_config(tokens)["path"] == "decode"
    hidden, weight, ids = M.make_inputs(tokens, seed=300)
    output = torch.empty_like(hidden)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        A.fused_moe(hidden, prep, weight, ids, output=output)  # 捕获前先编译。
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            A.fused_moe(hidden, prep, weight, ids, output=output)
    stream.synchronize()
    previous = None
    for seed in (310, 320):
        new_hidden, new_weight, new_ids = M.make_inputs(tokens, seed=seed)
        hidden.copy_(new_hidden)
        weight.copy_(new_weight)
        ids.copy_(new_ids)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        reference = X.reference_moe(hidden, raw, weight, ids)
        M.check_output(reference, output, f"graph TP{tp} T={tokens} seed={seed}")
        if previous is not None:
            assert not torch.equal(previous, output), "replay did not follow the new inputs"
        previous = output.clone()

