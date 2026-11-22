"""FlyDSL 两阶段 MoE 底座（Aiter gfx9_fp8）的其他覆盖测试。

只放基本功能测试（test_fly_2stage_basic.py）与 bench 覆盖不了的情况，每个测试的
docstring 写明原因：精度 / 激活变体、kernel 分支与 inter_dim 交界、尾块与 padding、
graph replay、编译期报错、逐位格式，以及 tuned_moe 的候选层门槛与固定配置报错。任务 bench
只计时 A8W4 新算子（对照是 Aiter 自身的 fused_moe），不经过这里的 BF16 / FP8 底座路径，
也不检查数值细节。容差与基本功能测试相同（calc_diff <= 0.02，逐位用例为 0 误差）。
"""

import inspect

import aiter
import pytest
import torch
from aiter.jit.utils.chip_info import get_cu_num

from pyhip.ops.moe import tuned_moe as tm
from pyhip.ops.moe.flydsl.moe_gemm_2stage import common, gemm2_8x1_compact
from pyhip.testing.moe import check_output, make_moe_runner, prepare_moe, torch_reference


TP2 = dict(HIDDEN_SIZE=2560, INTER_SIZE=320, TP=1, E=64, TOPK=8)
SMALL = dict(HIDDEN_SIZE=1024, INTER_SIZE=128, TP=1, E=8, TOPK=4)
# 每个 expert 257 行：在 80 CU 的 MI308X 上 compact 同时执行 M256 full 与 M64 tail 任务。
PREFILL_TOKENS = 2056
SPLITK = dict(_impl="fly_decode", tile_m_gate=16, tile_m_down=16, tile_n_gate=64, tile_n_down=64)


@pytest.fixture(autouse=True)
def _gpu_environment(monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("the Aiter gfx9_fp8 kernels target gfx942")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # 测当前源码，不复用旧磁盘编译产物。
    try:
        yield
    finally:
        torch.set_default_device(previous)


@pytest.fixture
def kernels(monkeypatch):
    """按启动顺序记录 (stage, alg, down_path)，用来断言实际走了哪条路径。"""
    seen = []
    original = tm._fly_kernel

    def record(**params):
        seen.append((params["stage"], params["alg"], params.get("down_path", "default")))
        return original(**params)

    monkeypatch.setattr(tm, "_fly_kernel", record)
    return seen


@pytest.fixture
def task_counts(monkeypatch):
    """记录 compact 任务表的 [full 行数, tail 行数]（由 GPU 每次 launch 写入）。"""
    counts = []
    allocate = gemm2_8x1_compact.allocate_task_buffers

    def record(*args):
        buffers = allocate(*args)
        counts.append(buffers[2])
        return buffers

    monkeypatch.setattr(gemm2_8x1_compact, "allocate_task_buffers", record)
    return counts


def _prepare(model, tokens, precision, **options):
    bf16 = precision == "bf16"
    call, _ = prepare_moe(model, tokens, dtype="bf16" if bf16 else "fp8",
                          quant="model" if bf16 else precision, seed=8, **options)
    call["hidden_states"].mul_(50)  # 进入激活函数非线性区，并让中间激活量化有意义。
    return call


def _check(call, config):
    runner = make_moe_runner(config)
    reference = torch_reference(call)
    assert torch.isfinite(reference).all()
    call["output"].fill_(float("nan"))
    check = check_output(runner(**call), call["output"], reference)
    assert check["status"] == "PASS", f"{config}: {check}"
    return runner


def _replay_graph(call, runner, storage=None):
    """在旁路 stream 上捕获连续两次调用；改输入、路由和路由权重后 replay 并重新校验。"""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner(**call)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner(**call)
            result = runner(**call)
    stream.synchronize()
    x = call["hidden_states"].clone()
    experts = call["w1"].shape[0]
    for factor in (-.75, 0.0, 1.0):
        call["hidden_states"].copy_(x * factor)
        call["topk_ids"].copy_((call["topk_ids"] + 1) % experts)
        call["topk_weight"].mul_(-.5)
        reference = torch_reference(call)
        call["output"].fill_(float("nan"))
        graph.replay()
        check = check_output(result, call["output"], reference)
        assert check["status"] == "PASS", check
        if storage is not None:
            assert (storage[:32] == -123).all() and (storage[-32:] == -123).all()


def _prefill(inter, down_path, padding):
    # I640 用 Aiter 调优表在该 I 上选的 Gate/Up tile（N256/K128），其余同基本用例（N128/K256）。
    gate = dict(tile_n_gate=256, tile_k_gate=128) if inter == 640 else dict(tile_n_gate=128, tile_k_gate=256)
    return dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=64,
                tile_n_down=256 if down_path == "1x4_64x256" else 128,
                down_path=down_path, padding=padding) | gate


def _round_bf16_reference(value):
    """定点测试遵循当前舍入模式（默认 RTE）；不改变完整 MoE 的独立参考和容差。"""
    if not (common._SIMPLIFIED_BF16_RTA or common._SIMPLIFIED_BF16_RTE):
        return value.bfloat16()
    bits = value.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bias = 0x8000 if common._SIMPLIFIED_BF16_RTA else 0x7FFF + ((bits >> 16) & 1)
    return ((bits + bias) >> 16).to(torch.int16).view(torch.bfloat16)


@pytest.mark.parametrize("precision, tokens", [("per_tensor", 4), ("bf16", 16)])
def test_splitk_variants(kernels, precision, tokens):
    """基本用例只测 PTPC、N64 tile 且 M·topk > E。这里补 per-tensor / BF16 权重的 scale 分支、
    N128 tile 的宽拷贝，以及 M·topk <= E（4×8 <= 64）时 launcher 改为每条路由一个 task。"""
    _check(_prepare(TP2, tokens, precision), SPLITK | dict(tile_n_gate=128, tile_n_down=128))
    assert kernels == [("gateup", "splitk", "default"), ("down", "splitk", "default")]


@pytest.mark.parametrize("tokens, hidden, inter, precision, activation, gate_n", [
    (3, 512, 128, "ptpc", "silu", 32),
    (8, 512, 128, "per_tensor", "swiglu", 64),
    (7, 32768, 64, "bf16", "silu", 64),
    (5, 32768, 64, "ptpc", "silu", 32),
], ids=["ptpc-m3", "per_tensor-swiglu-m8", "bf16-h32768-m7", "ptpc-h32768-m5"])
def test_batch1_clear_graph(tokens, hidden, inter, precision, activation, gate_n):
    """基本用例只跑一次 eager 的 M1 PTPC。这里检查 Gate/Up 融合清零写满输出且不越过两侧
    红区（H32768、I64 需要多轮清零）、2–8 token、per-tensor / BF16 / SwiGLU 的 batch1
    epilogue，以及 graph replay 时输入与路由变化后结果仍正确。"""
    model = dict(HIDDEN_SIZE=hidden, INTER_SIZE=inter, TP=1, E=4, TOPK=2)
    call = _prepare(model, tokens, precision, activation=activation)
    storage = torch.full((tokens * hidden + 64,), -123.0, dtype=torch.bfloat16)
    call["output"] = storage[32:-32].view(tokens, hidden)
    runner = _check(call, dict(_impl="fly_decode", decode_alg="batch1",
                               tile_n_gate=gate_n, tile_n_down=64))
    assert (storage[:32] == -123).all() and (storage[-32:] == -123).all()
    _replay_graph(call, runner, storage)


@pytest.mark.parametrize("tokens, config", [
    pytest.param(2, SPLITK | dict(tile_m_gate=64, tile_m_down=64, tile_n_gate=128,
                                  tile_n_down=128), id="splitk-m2"),
    pytest.param(64, dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=64, tile_n_gate=128,
                          tile_n_down=128, tile_k_gate=64), id="prefill-m64"),
])
def test_swiglu(tokens, config):
    """基本用例只测 SiLU。SwiGLU 的 clamp 与 (up + 1) 在 split-K 和 prefill 的 Gate/Up
    epilogue 各有一份（batch1 的在 test_batch1_clear_graph）；limit=0.02 让 clamp 生效。"""
    _check(_prepare(SMALL, tokens, "bf16", activation="swiglu", swiglu_limit=.02), config)


@pytest.mark.parametrize("precision, hidden, inter, down_path, padding, tokens", [
    # per-tensor：FlyDSL absmax 量化、Gate/Up 输出预清零和单一激活 scale，基本用例只测 PTPC。
    pytest.param("per_tensor", 2560, 320, "default", None, PREFILL_TOKENS, id="per_tensor-default"),
    pytest.param("per_tensor", 2560, 320, "1x4_64x256", 128, PREFILL_TOKENS, id="per_tensor-1x4"),
    pytest.param("per_tensor", 2560, 192, "8x1_compact", 128, PREFILL_TOKENS, id="per_tensor-compact"),
    # 8x1 full kernel 按 K 分三类调度：K192（基本用例）、K128n 不携带 B 地址（K256）与携带（K640）。
    pytest.param("ptpc", 2560, 256, "8x1_compact", 128, PREFILL_TOKENS, id="compact-i256"),
    pytest.param("ptpc", 2560, 640, "8x1_compact", 128, PREFILL_TOKENS, id="compact-i640"),
    # 8x1 的 N 调度下界：N512 只有 4 个 N128 tile。
    pytest.param("ptpc", 512, 192, "8x1_compact", 128, PREFILL_TOKENS, id="compact-h512"),
    # 1x4 在 K640 时 LDS 接近 64 KiB；同时覆盖专用 Down 的 padding=0（调优候选之一）。
    pytest.param("ptpc", 2560, 640, "1x4_64x256", 0, PREFILL_TOKENS, id="1x4-i640-pad0"),
    # M·topk <= E（4×8 <= 64）时 prefill launcher 只发 M·topk 个 task，基本用例的 M 远大于 E。
    pytest.param("ptpc", 2560, 320, "default", None, 4, id="default-m4"),
])
def test_prefill_variants(kernels, task_counts, precision, hidden, inter, down_path, padding, tokens):
    """基本用例只测 PTPC、H2560、I320/I192、padding 128 与大 M。每行注释写明补的分支或交界。"""
    model = TP2 | dict(HIDDEN_SIZE=hidden, INTER_SIZE=inter)
    _check(_prepare(model, tokens, precision), _prefill(inter, down_path, padding))
    assert kernels == [("gateup", "prefill_1x4", "default"), ("down", "prefill_1x4", down_path)]
    if down_path == "8x1_compact" and get_cu_num() == 80:
        assert len(task_counts) == 1 and bool((task_counts[0] > 0).all()), task_counts


@pytest.mark.parametrize("inter, tokens, config", [
    pytest.param(320, 16, SPLITK, id="splitk"),
    pytest.param(192, PREFILL_TOKENS, _prefill(192, "8x1_compact", 128), id="prefill-8x1_compact"),
])
def test_graph_replay(inter, tokens, config):
    """基本用例只跑 eager。graph replay 时排序、loc_ids 与 compact 任务表都必须在设备端按
    新路由重建（不能有 host 同步或沿用捕获时的值）；batch1 的 graph 见 test_batch1_clear_graph。"""
    call = _prepare(TP2 | dict(INTER_SIZE=inter), tokens, "ptpc")
    _replay_graph(call, _check(call, config))


def _prefill_candidates(call):
    """tuned_moe 为该调用枚举的 FlyDSL prefill 候选。"""
    bound = inspect.signature(tm.fused_moe).bind(**call)
    bound.apply_defaults()
    return [c.all_kwargs() for c in tm._configs(**tm._make_tune_args(bound.arguments))
            if c.all_kwargs()["_impl"] == "fly_prefill"]


@pytest.mark.parametrize("precision, inter, gate_ns, default_ms", [
    ("ptpc", 192, {128}, {32, 64, 128}),  # 2I=384 不是 N256 的倍数。
    ("ptpc", 320, {128}, {32, 64, 128}),  # 2I=640 同上。
    ("ptpc", 640, {128, 256}, {32, 64}),  # 默认 Down 的 M128×I640 FP8 激活 80 KiB。
    ("bf16", 256, {128, 256}, {32, 64, 128}),  # M128×I256 BF16 恰为 64 KiB，保留。
    ("bf16", 320, {128}, {32, 64}),  # M128×I320 BF16 80 KiB。
], ids=["ptpc-i192", "ptpc-i320", "ptpc-i640", "bf16-i256", "bf16-i320"])
def test_candidate_gates(precision, inter, gate_ns, default_ms):
    """tuned_moe 候选层的两条门槛：prefill Gate/Up 的不完整 N tile、默认 Down 超过 64 KiB LDS 的
    M×I 激活。两者编译都不报错、launch 才失败，不能交给编译失败剪枝；基本用例与 bench 用固定
    配置，不经过候选枚举。专用 Down 固定 M64，不受 LDS 门槛影响。"""
    configs = _prefill_candidates(_prepare(TP2 | dict(INTER_SIZE=inter), 65, precision))
    assert {c["tile_n_gate"] for c in configs} == gate_ns
    assert {c["tile_m_down"] for c in configs if "down_path" not in c} == default_ms
    assert {c["down_path"] for c in configs if "down_path" in c} == (
        set() if precision == "bf16" else {"1x4_64x256", "8x1_compact"})


def test_lds_64k_boundary(kernels):
    """边界：默认 Down 的 M128×I256 BF16 激活恰为 64 KiB，候选层保留它，launch 与结果都必须正确；
    上一个测试只检查候选枚举，基本用例的 BF16 prefill 只用 M64。"""
    _check(_prepare(TP2 | dict(INTER_SIZE=256), PREFILL_TOKENS, "bf16"),
           dict(_impl="fly_prefill", tile_m_gate=128, tile_m_down=128, tile_n_gate=128,
                tile_n_down=128, tile_k_gate=64))
    assert kernels == [("gateup", "prefill_1x4", "default"), ("down", "prefill_1x4", "default")]


def test_split_m_tile_rejected():
    """报错：Gate/Up 与 Down 共用同一份按 M tile 排序的结果，固定配置给出不同 M 时必须在启动
    kernel 前拒绝。候选层只枚举相同的 M，基本用例与 bench 走不到这个分支。"""
    runner = make_moe_runner(dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=128,
                                  tile_n_gate=128, tile_n_down=128, tile_k_gate=128))
    with pytest.raises(ValueError, match="share the sorting M tile"):
        runner(**_prepare(TP2, 65, "ptpc"))


@pytest.mark.parametrize("alg", ["batch1", "splitk"])
@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_gateup_rejects_incomplete_k(alg, kind):
    """报错：K 不是 256 的倍数时 batch1 / split-K Gate/Up 必须在编译期拒绝。tuned_moe 不在
    候选层过滤 K，靠编译失败剪枝；基本用例与 bench 只走合法形状。"""
    from pyhip.ops.moe.flydsl.moe_gemm_2stage import compile_moe_gemm1

    with pytest.raises(AssertionError, match="divisible by 256"):
        compile_moe_gemm1(N=256, K=384, weight_dtype=kind,
                          weight_quant_type="no" if kind == "bf16" else "ptpc",
                          TOPK=2, BLOCK_TILE_SIZE_M=16, BLOCK_TILE_SIZE_N=64, alg=alg, E=4)


def test_bf16_rounding_bits():
    """逐位格式：其余用例按 calc_diff 判定，看不出 FP32→BF16 的舍入方式（halfway、奇偶 LSB、
    subnormal、溢出、带符号零与 NaN）；这里逐位比较向量与标量两种转换。"""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    upper = [0, 1, 0x7f, 0x80, 0x3f80, 0x3f81, 0x3fff, 0x4000, 0x7f7f,
             0x8000, 0x8001, 0x807f, 0x8080, 0xbf80, 0xbf81, 0xff7f]
    bits = [(hi << 16) | lo for hi in upper for lo in (0, 1, 0x7fff, 0x8000, 0x8001, 0xffff)]
    bits += [0x7f800000, 0xff800000, 0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff]
    bits += [0] * (512 - len(bits))
    x_cpu = torch.tensor(bits, dtype=torch.int64, device="cpu").to(torch.int32).view(torch.float32)
    expected = _round_bf16_reference(x_cpu)
    x = x_cpu.cuda()
    vector_out = torch.empty(x.shape, dtype=torch.bfloat16)
    scalar_out = torch.empty_like(vector_out)

    @flyc.kernel
    def convert(A: fx.Pointer, B: fx.Pointer, C: fx.Pointer):
        base = fx.thread_idx.x * 8
        values = fx.make_view(A + base, fx.make_layout(8, 1)).load()
        fx.make_view(B + base, fx.make_layout(8, 1)).store(common._f32_to_bf16(values))
        for i in fx.range_constexpr(8):
            C[base + i] = common._f32_to_bf16(A[base + i])

    @flyc.jit
    def launch(A: fx.Pointer, B: fx.Pointer, C: fx.Pointer, stream: fx.Stream):
        convert(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    _run_compiled(launch, common._ptr(x), common._ptr(vector_out), common._ptr(scalar_out),
                  torch.cuda.current_stream())
    nan = expected.isnan()
    for result in (vector_out.cpu(), scalar_out.cpu()):
        assert torch.equal(result.isnan(), nan)
        assert torch.equal(result[~nan].view(torch.int16), expected[~nan].view(torch.int16))


@pytest.mark.parametrize("path", ["default", "1x4_64x256", "8x1_compact"])
def test_down_rounds_before_packing(path):
    """逐位格式：点积恰为 1 时输出只由 FP32 的 weight scale × 路由权重决定，检查 Down
    epilogue 只舍入一次、打包前不丢精度。compact 传 cu_count=1 使 M256 full 任务不被拆分，
    同时覆盖 full 与 tail 两种 epilogue。"""
    from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from aiter.ops.shuffle import shuffle_weight
    from pyhip.ops.moe.flydsl.moe_gemm_2stage import compile_gemm, sorted_sum

    ptr = common._ptr
    b, h, k = 257, 512, 192
    qdtype = aiter.dtypes.fp8
    x = torch.zeros((b, 1, k), dtype=torch.bfloat16)
    w = torch.zeros((1, h, k), dtype=torch.bfloat16)
    x[..., 0] = 1
    w[..., 0] = 1
    x, w = x.to(qdtype), shuffle_weight(w.to(qdtype))
    scales = torch.tensor([1 + 1 / 256, 1 + 3 / 256, -(1 + 3 / 256),
                           1 + 3 / 256 - 2**-20, 1 + 3 / 256 + 2**-20, -0.0, .5, 2.0],
                          dtype=torch.float32).repeat(h // 8)
    a_scale = torch.ones((b, 1), dtype=torch.float32)
    ids = torch.zeros((b, 1), dtype=torch.int32)
    weights = torch.tensor([1.0, -.5, 2.0], dtype=torch.float32).repeat((b + 2) // 3)[:b, None].contiguous()
    padding = 0 if path == "default" else 128
    si, sw, se, valid, _, _, loc = moe_sorting(ids, weights, 1, h, torch.bfloat16, 64,
                                               accumulate=False, output_aux=AUX_SORT_OPUS)
    routes = torch.full((se.numel() * 64, h + padding // 2), float("nan"), dtype=torch.bfloat16)
    launcher = compile_gemm(N=h, K=k, weight_dtype="fp8", weight_quant_type="ptpc",
                            act_quant_type="ptpc", TOPK=1, E=1, BLOCK_TILE_SIZE_M=64,
                            BLOCK_TILE_SIZE_N=256 if path == "1x4_64x256" else 128,
                            stage="down", alg="prefill_1x4", down_path=path,
                            down_output_padding_bytes=padding)
    extra = ()
    if path == "8x1_compact":
        full, tail, counts = gemm2_8x1_compact.allocate_task_buffers(se, 1, 1)
        extra = (ptr(full), ptr(tail), ptr(counts), full.shape[0], tail.shape[0], 1)
    _run_compiled(launcher, *(ptr(t) for t in (x, w, routes, si, sw, se, valid, scales, a_scale)),
                  b, se.numel(), *extra, torch.cuda.current_stream())
    if path == "8x1_compact":
        assert counts.tolist() == [256, 64]
    out = torch.empty((b, h), dtype=torch.bfloat16)
    sorted_sum(1, h, padding)(loc, routes, out, b)
    torch.testing.assert_close(out, _round_bf16_reference(weights * scales), atol=0, rtol=0)
