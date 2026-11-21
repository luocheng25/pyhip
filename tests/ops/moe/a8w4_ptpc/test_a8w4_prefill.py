"""A8W4（mxfp4 权重）MoE 的 M2：prefill 档（配置表 decode 上界以上，当前 T>=129）。

形状、输入与容差见 ``pyhip.testing.moe_a8w4_ptpc``，mxfp4 权重与参考见 ``pyhip.testing.moe_a8w4_mxfp4``：
TP2 每卡 I=320，TP4 每卡 I=160（``prepare_weights`` 补零到 192）。参考的数据流与 M0 的 FP8 prefill 相同：
输入和 Gate/Up 结果（先舍入成 BF16）都按 token 量化成 FP8（``reference_moe(act_quant=True)``）；权重为 mxfp4
精确反量化并施加 kernel 的 flush 规则（比该行最大 code 小 12 以上的组为 0），FP32 计算；容差用 M0 的
``check_output`` 原样。

- prepare（CPU，小形状）：prefill 另存的 delta 排列（``w1_pdelta`` / ``w2_pdelta``）逐字节等于按模块说明的
  下标公式从 E8M0 code 算出的 ``min(c_max - code, 11) * 8``（含 I 补零的行 / 组），按专家分块准备结果相同。
- prefill：TP2 / TP4 × T（decode 上界 + 1 即交界，当前 129；1024；8192；65536 只比较每 16 个 token 取一个），
  断言实际编译的两级 kernel 就是配置表这一档、不经过 decode kernel；输出离“BF16 激活”的参考远 10 倍以上
  （确认激活与中间结果确实按 token 量化）。
- CUDA graph（T=1024）：捕获一次，replay 时换入新的输入与路由。
- 配置表以外的 tile：Gate/Up BK256（TP2 T=1024），以及小形状（E=8、H=1024、I=256）上 Gate/Up BN256、
  BK256（每个 wave 两个 M repeat，Qwen3.8 的 N=2I 不是 256 的倍数）。
- ``config`` 的 path 不是 decode / prefill 时，在启动任何 kernel 之前报错。
- ``compile_prefill_gemm`` 按设备缓存，Gate/Up BK 与 Down tile 的取值检查。
"""

import pytest
import torch

from pyhip.ops.moe.flydsl import a8w4_ptpc as A
from pyhip.ops.moe.flydsl.a8w4_ptpc import moe as entry
from pyhip.testing import moe_a8w4_mxfp4 as X
from pyhip.testing import moe_a8w4_ptpc as M

BOUNDARY = A.decode_max_tokens() + 1
TOKENS = sorted({BOUNDARY, 1024, 8192, 65536})
MATRIX = [(tp, tokens) for tp in (2, 4) for tokens in TOKENS]
GRAPH_CASES = [(2, 1024), (4, 1024)]
CHECK_STEP = {65536: 16}


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
            cache[tp] = X.make_mxfp4_weights(tp, seed=30 + tp, device="cuda")
        return cache[tp]

    yield get
    cache.clear()


def _record_compiles(monkeypatch):
    """记录 fused_moe 编译的 kernel：prefill 为 (stage, M, N, BK 或 None, 行距补的字节)，decode 为 ("decode", stage)。"""
    compiled = []
    prefill, decode = entry.compile_prefill_gemm, entry.compile_decode_gemm

    def compile_prefill_gemm(**params):
        compiled.append((params["stage"], params["BLOCK_TILE_SIZE_M"], params["BLOCK_TILE_SIZE_N"],
                         params.get("tile_k"), params.get("padding", 0)))
        return prefill(**params)

    def compile_decode_gemm(**params):
        compiled.append(("decode", params["stage"]))
        return decode(**params)

    monkeypatch.setattr(entry, "compile_prefill_gemm", compile_prefill_gemm)
    monkeypatch.setattr(entry, "compile_decode_gemm", compile_decode_gemm)
    return compiled


def _expected_compiles(config):
    return [("gateup", config["tile_m"], config["tile_n_gate"], config["tile_k_gate"], 0),
            ("down", config["tile_m"], config["tile_n_down"], None, config["padding"])]


def _check(hidden, raw, weight, ids, output, msg):
    """对 prefill 参考按 M0 容差检查（大 T 等间隔取 token），并确认离 BF16 激活的参考远 10 倍以上。"""
    tokens = hidden.shape[0]
    rows = torch.arange(0, tokens, CHECK_STEP.get(tokens, 1))
    inputs = (hidden[rows], raw, weight[rows], ids[rows])
    _, logits_diff, _ = M.check_output(X.reference_moe(*inputs, act_quant=True), output[rows], msg)
    _, bf16_diff, _ = M.output_errors(X.reference_moe(*inputs, act_quant=False), output[rows])
    assert bf16_diff > 10 * logits_diff, (msg, logits_diff, bf16_diff)


# ---------------------------------------------------------------- prepare_weights（CPU）

def _delta_bytes(codes):
    code = codes.to(torch.int32)
    return ((code.amax(dim=-1, keepdim=True) - code).clamp(max=11) * 8).to(torch.uint8)


def test_prefill_delta_layout_bitwise():
    experts, hidden, inter, padded = 2, 512, 96, 128
    gen = torch.Generator(device="cpu").manual_seed(5)
    w1 = torch.randint(0, 256, (experts, 2 * inter, hidden // 2), generator=gen, device="cpu").to(torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, inter // 2), generator=gen, device="cpu").to(torch.uint8)
    s1 = X.random_codes((experts, 2 * inter, hidden // 32), gen=gen, device="cpu")
    s2 = X.random_codes((experts, hidden, inter // 32), gen=gen, device="cpu")
    prep = A.prepare_weights(w1, s1, w2, s2)
    assert prep.inter_padded == padded
    assert prep.w1_pdelta.shape == (experts, 2 * padded // 16, hidden // 256, 2, 16, 4)
    assert prep.w2_pdelta.shape == (experts, hidden // 64, padded // 32, 16, 4)
    # 补零：w1 的补行（gate、up 各 32 行）code 为 0；w2 的补列（K 的最后一组）code 为 0。
    codes1 = torch.zeros((experts, 2 * padded, hidden // 32), dtype=torch.uint8)
    codes1[:, :inter], codes1[:, padded:padded + inter] = s1[:, :inter], s1[:, inter:]
    codes2 = torch.zeros((experts, hidden, padded // 32), dtype=torch.uint8)
    codes2[..., :inter // 32] = s2
    delta1, delta2 = _delta_bytes(codes1), _delta_bytes(codes2)
    assert bool((delta2[..., inter // 32:] == 88).all())  # 补的组 delta 封顶 11（nibble 为 0，解码为 0）
    # w1_pdelta[e, n16, kb, h, l, i] 属于第 16*n16 + l 行的第 8*kb + 2*i + h 组。
    e, n16, kb, h, l, i = torch.meshgrid(*(torch.arange(s) for s in prep.w1_pdelta.shape), indexing="ij")
    assert torch.equal(prep.w1_pdelta, delta1[e, 16 * n16 + l, 8 * kb + 2 * i + h])
    # w2_pdelta[e, n64, g, r, i] 属于第 64*n64 + 16*(r//4) + r%4 + 4*i 行的第 g 组。
    e, n64, g, r, i = torch.meshgrid(*(torch.arange(s) for s in prep.w2_pdelta.shape), indexing="ij")
    assert torch.equal(prep.w2_pdelta, delta2[e, 64 * n64 + 16 * (r // 4) + r % 4 + 4 * i, g])
    # 按专家分块准备（chunk=1）结果相同。
    chunked = A.prepare_weights(w1, s1, w2, s2, chunk=1)
    assert torch.equal(chunked.w1_pdelta, prep.w1_pdelta) and torch.equal(chunked.w2_pdelta, prep.w2_pdelta)


# ---------------------------------------------------------------- prefill（GPU）

@pytest.mark.parametrize("tp, tokens", MATRIX, ids=[f"tp{tp}-t{t}" for tp, t in MATRIX])
def test_prefill_matches_reference(gpu, shard, monkeypatch, tp, tokens):
    raw, prep = shard(tp)
    config = A.select_config(tokens)
    assert config["path"] == "prefill", config
    hidden, weight, ids = M.make_inputs(tokens, seed=tokens + tp)
    output = torch.full_like(hidden, float("nan"))
    compiled = _record_compiles(monkeypatch)
    assert A.fused_moe(hidden, prep, weight, ids, output=output) is output
    assert compiled == _expected_compiles(config), compiled
    _check(hidden, raw, weight, ids, output, f"TP{tp} T={tokens} {config}")


@pytest.mark.parametrize("tp, tokens", GRAPH_CASES, ids=[f"tp{tp}-t{t}" for tp, t in GRAPH_CASES])
def test_prefill_cuda_graph(gpu, shard, tp, tokens):
    raw, prep = shard(tp)
    assert A.select_config(tokens)["path"] == "prefill"
    hidden, weight, ids = M.make_inputs(tokens, seed=400)
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
    for seed in (410, 420):
        new_hidden, new_weight, new_ids = M.make_inputs(tokens, seed=seed)
        hidden.copy_(new_hidden)
        weight.copy_(new_weight)
        ids.copy_(new_ids)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check(hidden, raw, weight, ids, output, f"graph TP{tp} T={tokens} seed={seed}")
        if previous is not None:
            assert not torch.equal(previous, output), "replay did not follow the new inputs"
        previous = output.clone()


def _small_case(experts=8, hidden=1024, inter=256, tokens=300, topk=4, seed=7):
    """小形状的自然布局 mxfp4 与输入：H、I 是 256 的倍数，Gate/Up 可用 BN256。"""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w1 = torch.randint(0, 256, (experts, 2 * inter, hidden // 2), generator=gen, device="cuda").to(torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, inter // 2), generator=gen, device="cuda").to(torch.uint8)
    s1 = X.random_codes((experts, 2 * inter, hidden // 32), gen=gen, device="cuda")
    s2 = X.random_codes((experts, hidden, inter // 32), gen=gen, device="cuda")
    x = torch.randn((tokens, hidden), generator=gen, device="cuda").to(torch.bfloat16)
    ids = torch.rand((tokens, experts), generator=gen, device="cuda").topk(topk, dim=1).indices.to(torch.int32)
    weight = torch.randn((tokens, topk), generator=gen, device="cuda").softmax(dim=-1)
    return (w1, s1, w2, s2), x, weight.contiguous(), ids.contiguous()


def test_prefill_other_tiles(gpu, shard, monkeypatch):
    base = A.select_config(BOUNDARY)
    # Gate/Up BK256：每个 delta dword 一个 K tile（BK128 时两个 K tile 共用一个 dword）。
    raw, prep = shard(2)
    config = dict(base, tile_k_gate=256)
    hidden, weight, ids = M.make_inputs(1024, seed=500)
    output = torch.full_like(hidden, float("nan"))
    compiled = _record_compiles(monkeypatch)
    assert A.fused_moe(hidden, prep, weight, ids, output=output, config=config) is output
    assert compiled == _expected_compiles(config), compiled
    _check(hidden, raw, weight, ids, output, f"TP2 T=1024 {config}")
    # 小形状上的 Gate/Up BN256（每个 wave 两个 M repeat）+ BK256。
    raw, x, weight, ids = _small_case()
    config = dict(base, tile_n_gate=256, tile_k_gate=256)
    output = torch.full_like(x, float("nan"))
    compiled.clear()
    assert A.fused_moe(x, A.prepare_weights(*raw), weight, ids, output=output, config=config) is output
    assert compiled == _expected_compiles(config), compiled
    _check(x, raw, weight, ids, output, f"E=8 H=1024 I=256 {config}")


def test_unknown_path_rejected_before_launch(gpu, shard, monkeypatch):
    _, prep = shard(4)
    hidden, weight, ids = M.make_inputs(BOUNDARY, seed=600)
    output = torch.full_like(hidden, float("nan"))
    compiled = _record_compiles(monkeypatch)
    with pytest.raises(ValueError, match="unknown A8W4 path"):
        A.fused_moe(hidden, prep, weight, ids, output=output, config=dict(A.select_config(BOUNDARY), path="x"))
    assert compiled == [] and bool(output.isnan().all())


def test_compile_prefill_gemm_cached_per_device(gpu, monkeypatch):
    from pyhip.ops.moe.flydsl.a8w4_ptpc import prefill as P

    gateup = dict(stage="gateup", N=2 * 192, K=M.H, TOPK=M.TOPK, BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=128,
                  tile_k=128, E=M.E)
    first = P.compile_prefill_gemm(**gateup)
    assert P.compile_prefill_gemm(**gateup) is first
    down = dict(stage="down", N=M.H, K=192, TOPK=M.TOPK, BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=256,
                padding=128, E=M.E)
    assert P.compile_prefill_gemm(**down) is P.compile_prefill_gemm(**down, tile_k=256)  # Down 不用 BK
    monkeypatch.setattr(P, "get_device_cache_key", lambda: ("another device",))
    assert P.compile_prefill_gemm(**gateup) is not first
    for bad in (dict(gateup, tile_k=64), dict(gateup, tile_k=None), dict(down, BLOCK_TILE_SIZE_N=128),
                dict(down, BLOCK_TILE_SIZE_M=32)):
        with pytest.raises(AssertionError):
            P.compile_prefill_gemm(**bad)
