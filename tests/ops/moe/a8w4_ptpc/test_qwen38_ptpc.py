"""Qwen3.8-Flash-Next 单卡分片的 tuned_moe FlyDSL ptpc 路径对照 FP32 参考。

形状、输入、参考和配置表见 ``pyhip.testing.moe_a8w4_ptpc``：TP2 每卡 I=320，TP4 每卡
I=160（补零到 192）。每个 T 经 ``make_moe_runner`` 固定使用 ``FLY_CONFIGS`` 选出的
tuned_moe FlyDSL 候选，不调优、不回退 Aiter；容差沿用上游 aiter
``op_tests/test_flydsl_moe_gfx942.py``；并断言两次 GEMM 的算法与 Down path 就是配置表选的那一档，
且在 gfx942 上都由 ``gemm.compile_gemm`` 分流到上游 gfx9_fp8 实现，而不是 ``*_legacy``。

T 覆盖配置表的每一档及各档上界（1、16、128、TP2 的 1024、TP4 的 8192、65536）：decode（1/8/16/32/128）
和 prefill（129、256、257、1024、不是 64 整倍数的 2000、8192、最大的 65536）。decode 与 prefill 的
交界由配置表决定，当前是 128/129（T<=128 走 decode，T>=129 走 prefill 1x4）；256/257 不是交界。
TP4 再用 8200 覆盖配置表 8192 以上那一行（上游 compact Down，bench 的 T=65536 用它）。
T=65536 时 Down 的路由中间结果超过 2 GiB（字节偏移超出 int32）。
decode 的三种配置（batch1 gn32 / gn64、split-K）各在 CUDA graph 中捕获一次，replay 时换入
新的输入和路由。

另外两项：BF16 舍入切到 RTE（``AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=0``）时四类上游 kernel
仍满足同一参考与容差；``compile_gemm(has_num_local_tokens=True)`` 的上游 batch1 原生 launcher
在 CUDA graph 中按设备上的 token 数跳过补齐行。
"""

import pytest
import torch

from pyhip.testing import moe_a8w4_ptpc as M

TOKENS = (1, 8, 16, 32, 128, 129, 256, 257, 1024, 2000, 8192, 65536)
MATRIX = [(tp, tokens) for tp in (2, 4) for tokens in TOKENS] + [(4, 8200)]
GRAPH_CASES = [(2, 1), (2, 32), (4, 8), (4, 128)]
# batch1、split-K、prefill 1x4（BK128）与 compact（BK256）各一个配置。
RTE_CASES = [(4, 1), (2, 32), (2, 257), (4, 8200)]


@pytest.fixture(autouse=True)
def _gpu_environment(monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("Qwen3.8 FlyDSL ptpc 配置按 MI308X（gfx942）选定")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # 测当前源码，不复用旧磁盘编译产物。
    try:
        yield
    finally:
        torch.set_default_device(previous)


@pytest.fixture(scope="module")
def shard():
    """``shard(tp) -> (raw, kernel)``，每个 TP 只生成一次。"""
    cache = {}

    def get(tp):
        if tp not in cache:
            cache[tp] = M.make_weights(tp, seed=tp)
        return cache[tp]

    yield get
    cache.clear()


def _record_compile_gemm(monkeypatch):
    """记录 tuned_moe 经 moe_gemm_splitk.compile_gemm 编译的每个配置。"""
    from pyhip.ops.moe.flydsl import moe_gemm_splitk as fly

    compiled = []
    original = fly.compile_gemm

    def compile_gemm(**params):
        compiled.append(params)
        return original(**params)

    monkeypatch.setattr(fly, "compile_gemm", compile_gemm)
    return compiled


@pytest.mark.parametrize("tp, tokens", MATRIX, ids=[f"tp{tp}-t{t}" for tp, t in MATRIX])
def test_matches_reference(shard, monkeypatch, tp, tokens):
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.gemm import gemm_implementation

    raw, kernel = shard(tp)
    config = M.fly_config(tp, tokens)
    hidden, weight, ids = M.make_inputs(tokens, seed=tokens)
    output = torch.full_like(hidden, float("nan"))
    call = M.moe_call(kernel, hidden, weight, ids, output)
    # 固定配置必须是 tuned_moe 对这组输入实际会生成的 FlyDSL 候选。
    assert config in M.fly_candidates(call), config
    compiled = _record_compile_gemm(monkeypatch)
    assert M.fly_runner(config)(**call) is output
    stages = [params["stage"] for params in compiled]
    assert stages == ["gateup", "down"], stages
    # 实际编译的就是配置表这一档的路径：batch1 / split-K / prefill 1x4（Down 为 1x4 或 compact）。
    algorithm = "prefill_1x4" if config["_impl"] == "fly_prefill" else config.get("decode_alg", "splitk")
    assert [params["alg"] for params in compiled] == [algorithm, algorithm], compiled
    assert compiled[1].get("down_path", "default") == config.get("down_path", "default"), compiled
    assert all(gemm_implementation(**params) == "upstream" for params in compiled), compiled
    reference = M.reference_moe(hidden, raw, weight, ids, act_quant=M.act_quant(config))
    M.check_output(reference, output, f"TP{tp} T={tokens} {M.describe(config)}")


@pytest.mark.parametrize("tp, tokens", GRAPH_CASES, ids=[f"tp{tp}-t{t}" for tp, t in GRAPH_CASES])
def test_decode_cuda_graph(shard, tp, tokens):
    raw, kernel = shard(tp)
    config = M.fly_config(tp, tokens)
    assert config["_impl"] == "fly_decode", config
    hidden, weight, ids = M.make_inputs(tokens, seed=300)
    output = torch.empty_like(hidden)
    call = M.moe_call(kernel, hidden, weight, ids, output)
    runner = M.fly_runner(config)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner(**call)  # 捕获前先编译。
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner(**call)
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
        reference = M.reference_moe(hidden, raw, weight, ids, act_quant=False)
        M.check_output(reference, output, f"graph TP{tp} T={tokens} seed={seed}")
        if previous is not None:
            assert not torch.equal(previous, output), "replay did not follow the new inputs"
        previous = output.clone()


@pytest.mark.parametrize("tp, tokens", RTE_CASES, ids=[f"tp{tp}-t{t}" for tp, t in RTE_CASES])
def test_rte_rounding_matches_reference(shard, monkeypatch, tp, tokens):
    from functools import cache

    from pyhip.ops.moe.flydsl.moe_gemm_2stage import common, gemm

    raw, kernel = shard(tp)
    config = M.fly_config(tp, tokens)
    hidden, weight, ids = M.make_inputs(tokens, seed=tokens + 1)
    reference = M.reference_moe(hidden, raw, weight, ids, act_quant=M.act_quant(config))
    outputs = {}
    for mode, rta in (("rta", True), ("rte", False)):
        # 开关在 kernel 追踪时读取：每种模式换一份空的编译缓存，按该模式重新生成两级 kernel，
        # 与进程的默认设置无关，也不影响其它用例复用的版本。
        monkeypatch.setattr(common, "_SIMPLIFIED_BF16_RTA", rta)
        monkeypatch.setattr(gemm, "_compile_gemm_cached", cache(gemm._compile_gemm_cached.__wrapped__))
        output = torch.full_like(hidden, float("nan"))
        assert M.fly_runner(config)(**M.moe_call(kernel, hidden, weight, ids, output)) is output
        assert gemm._compile_gemm_cached.cache_info().currsize == 2
        M.check_output(reference, output, f"{mode} TP{tp} T={tokens} {M.describe(config)}")
        outputs[mode] = output
    if tokens >= 8192:
        # 约 2 亿次 BF16 舍入里有数千个中点，两种舍入的结果必然不同。
        assert not torch.equal(outputs["rta"], outputs["rte"])


def test_num_local_tokens_skips_padding_rows(shard):
    """``has_num_local_tokens=True``：上游 batch1 原生签名（多一个 token 数指针，取路由数 T*TOPK）；
    计数之后的行（CUDA graph 的补齐行）不计算，Gate/Up 清零后保持 0；计数在 replay 之间可变。"""
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.common import _run_compiled, torch_tensor_to_pointer as ptr
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.gemm import compile_gemm

    tp, tokens = 4, 8
    raw, kernel = shard(tp)
    w1, w2, w1_scale, w2_scale = kernel
    inter = w2.shape[2]
    options = dict(weight_dtype="fp8", weight_quant_type="ptpc", act_quant_type="ptpc", TOPK=M.TOPK, E=M.E,
                   alg="batch1", BLOCK_TILE_SIZE_M=16, BLOCK_TILE_SIZE_N=64, has_num_local_tokens=True)
    gateup = compile_gemm(N=2 * inter, K=M.H, stage="gateup", fused_down_clear=True, **options)
    down = compile_gemm(N=M.H, K=inter, stage="down", **options)
    hidden, weight, ids = M.make_inputs(tokens, seed=500)
    mid = torch.empty((tokens, M.TOPK, inter), dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    count = torch.full((1,), tokens, dtype=torch.int32)

    def run(stream):
        routes = tokens * M.TOPK
        _run_compiled(gateup, ptr(hidden), ptr(w1), ptr(mid), ptr(ids), ptr(output), ptr(w1_scale),
                      ptr(count), routes, stream)
        _run_compiled(down, ptr(mid), ptr(w2), ptr(output), ptr(ids), ptr(weight), ptr(w2_scale),
                      ptr(count), routes, stream)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(stream)  # 捕获前先编译。
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run(stream)
    stream.synchronize()
    for valid in (tokens, 5, 1, 0):
        count.fill_(valid)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert not output[valid:].any(), f"padding rows after {valid} were written"
        if valid:
            reference = M.reference_moe(hidden[:valid], raw, weight[:valid], ids[:valid], act_quant=False)
            M.check_output(reference, output[:valid], f"num_local_tokens={valid}")

