"""完整 tuned MoE 和 benchmark 调优前置流程的回归。"""

import csv
import inspect
from types import SimpleNamespace

import pytest
import torch
import aiter

from benchmarks.moe import bench_tuned_moe as bench
from pyhip import calc_diff
from pyhip.ops.moe import tuned_moe as tm
from pyhip.testing.moe import make_moe_runner, measure_moe, prepare_moe, torch_reference


@pytest.fixture(autouse=True)
def _cuda_default(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires ROCm GPU")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # 不影响 autotune 配置缓存测试。
    yield
    torch.set_default_device(previous)


def _block_call(tokens=17, hidden=512, inter=256, shuffled=(True, True)):
    model = dict(HIDDEN_SIZE=hidden, INTER_SIZE=inter, TP=1, E=4, TOPK=2)
    call, _ = prepare_moe(model, tokens, dtype="fp8", quant="block", seed=8, preshuffle="off")
    # 不只覆盖 benchmark 的小输入；扩大幅度以检验 SiLU 和中间激活量化。
    call["hidden_states"].mul_(50)
    from aiter.ops.shuffle import shuffle_weight

    for name, enabled in zip(("w1", "w2"), shuffled):
        if enabled:
            call[name] = shuffle_weight(call[name])
    return call


def _gelu_call(tokens=17, hidden=512, inter=256):
    model = dict(HIDDEN_SIZE=hidden, INTER_SIZE=inter, TP=1, E=4, TOPK=2)
    call, _ = prepare_moe(model, tokens, dtype="bf16", activation="gelu", seed=8, preshuffle="off")
    call["hidden_states"].mul_(100)  # 覆盖 GELU 的非线性区间，而不只测接近零的输入。
    return call


def _check(result, call, reference):
    assert result is call["output"]
    assert result.shape == reference.shape and result.dtype == reference.dtype
    assert torch.isfinite(result).all()
    assert calc_diff(reference, result) <= .02


def _tune_arguments(call):
    bound = inspect.signature(tm.fused_moe).bind(**call)
    bound.apply_defaults()
    return tm._make_tune_args(bound.arguments)


def _native_configs(call):
    return [config for config in tm._configs(**_tune_arguments(call))
            if config.all_kwargs()["_impl"] == "jit_blockscale"]


def _fly_call(tokens=4, hidden=512, inter=128, kind="bf16", activation="silu", gate_mode="separated"):
    model = dict(HIDDEN_SIZE=hidden, INTER_SIZE=inter, TP=1, E=4, TOPK=2)
    call, _ = prepare_moe(model, tokens,
        dtype="mxfp4" if kind == "fp4" else "bf16" if kind == "bf16" else "fp8",
        quant=kind if kind in ("ptpc", "per_tensor") else "model", seed=8,
        gate_mode=gate_mode, preshuffle="on", routing="balanced", activation=activation,
        beta=.5 if activation == "situv2" else None,
        linear_beta=2.0 if activation == "situv2" else None, swiglu_limit=None,
    )
    call["hidden_states"].mul_(50)
    return call


@pytest.mark.parametrize("hidden", [128, 256, 384, 512, 768])
@pytest.mark.parametrize("kind", ["bf16", "ptpc"])
def test_fly_candidates_defer_shape_checks(hidden, kind):
    call = _fly_call(tokens=65, hidden=hidden, kind=kind)
    configs = [c.all_kwargs() for c in tm._configs(**_tune_arguments(call))]
    assert any(c["_impl"].startswith("jit_") for c in configs)  # 不限制其它后端。
    assert any(c["_impl"] == "fly_decode" for c in configs)
    assert any(c["_impl"] == "fly_prefill" for c in configs)
    # 候选层只排除 launch 时才出错的不完整 N tile / 默认 Down LDS 超限，K 等限制交给编译检查。
    prefill = [c for c in configs if c["_impl"] == "fly_prefill" and "down_path" not in c]
    assert {(c["tile_m_gate"], c["tile_n_gate"], c["tile_k_gate"]) for c in prefill} == {
        (m, n, k) for m in (32, 64, 128) for n in (128, 256)
        for k in ((64, 128) if kind == "bf16" else (128, 256))
    }
    limited = tm._configs(**_tune_arguments(call | {"block_size_M": 64}))
    assert all(c.all_kwargs()["tile_m_down"] == 64 for c in limited if c.all_kwargs()["_impl"] != "aiter")


@pytest.mark.parametrize("kind, path, inter, gate_n", [
    ("bf16", "default", 128, 128),
    ("ptpc", "default", 256, 128),
    ("ptpc", "default", 192, 128),
    ("ptpc", "default", 320, 128),
    ("per_tensor", "default", 192, 128),
    ("per_tensor", "default", 320, 128),
    ("ptpc", "1x4_64x256", 192, 128),
    ("ptpc", "8x1_compact", 192, 128),
    ("per_tensor", "1x4_64x256", 256, 128),
    ("per_tensor", "8x1_compact", 256, 128),
    ("ptpc", "1x4_64x256", 256, 256),
    ("ptpc", "8x1_compact", 256, 256),
    ("per_tensor", "1x4_64x256", 384, 256),
    ("per_tensor", "8x1_compact", 384, 256),
])
def test_fly_prefill_k_tiles(monkeypatch, kind, path, inter, gate_n):
    call = _fly_call(tokens=257, inter=inter, kind=kind)
    tuning = _tune_arguments(call)
    configs = [c.all_kwargs() for c in tm._configs(**tuning)
               if c.all_kwargs()["_impl"] == "fly_prefill"
               and c.all_kwargs().get("down_path", "default") == path
               and c.all_kwargs()["tile_m_gate"] == 64
               and c.all_kwargs()["tile_m_down"] == 64
               and c.all_kwargs()["tile_n_gate"] == gate_n
               and c.all_kwargs().get("padding") == (None if path == "default" else 128)]
    expected_ks = {64, 128} if kind == "bf16" else {128, 256}
    assert {c["tile_k_gate"] for c in configs} == expected_ks and len(configs) == 2
    reference = tm._torch_reference(call)
    compiled = []
    original_kernel = tm._fly_kernel

    def fly_kernel(**params):
        compiled.append(params)
        return original_kernel(**params)

    monkeypatch.setattr(tm, "_fly_kernel", fly_kernel)
    monkeypatch.setattr(tm, "record_dispatch", True)
    monkeypatch.setattr(tm, "last_dispatch", None)
    for config in configs:
        call["output"].fill_(float("nan"))
        _check(tm._fmoe_wrapper(**tuning, **config), call, reference)
        assert tm.last_dispatch["tile_k_gate"] == config["tile_k_gate"]
        assert compiled[-2]["tile_k"] == config["tile_k_gate"]
        assert "tile_k" not in compiled[-1]  # Gate/Up 的 BK 不改变 Down 的 K 分块。


@pytest.mark.parametrize("kind, inter, bad_options, reason", [
    ("ptpc", 192, dict(tile_m_gate=32, tile_m_down=32), "num_atoms"),
    ("ptpc", 1024, dict(down_path="1x4_64x256", tile_n_down=256, padding=128),
     "exceeds gfx942 LDS capacity"),
    ("ptpc", 320, dict(down_path="8x1_compact", padding=128), "AssertionError"),
])
def test_fly_compile_failure_pruning(capsys, kind, inter, bad_options, reason):
    call = _fly_call(tokens=65, inter=inter, kind=kind)
    tuning = _tune_arguments(call)
    configs = tm._configs(**tuning)
    bad_params = dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=64, tile_n_gate=128,
                      tile_n_down=128, tile_k_gate=64 if kind == "bf16" else 128) | bad_options
    bad = next(c for c in configs if c.all_kwargs() == bad_params)
    good = next(c for c in configs if c.all_kwargs() == dict(
        _impl="fly_decode", tile_m_gate=16, tile_m_down=16, tile_n_gate=64, tile_n_down=64))
    assert tm._prune_invalid_configs([bad, good], tuning) == [good]
    assert reason in capsys.readouterr().out


def test_aiter_signature():
    # output/stage2_scatter 由 PyHIP 补齐，其余参数与当前 Aiter 一致。
    parameters = inspect.signature(tm.fused_moe).parameters
    native = inspect.signature(tm._aiter_fused_moe).parameters
    diff = set(parameters) - set(native)
    compatibility_names = {
         "output", "stage2_scatter", "quant_type_a", "quant_dtype_a", "quant_dtype_a2",
    }
    assert diff == compatibility_names - set(native)
    # 共有参数必须保持原生顺序，避免位置调用绑错参数。
    assert [name for name in parameters if name in native] == list(native)
    for name, parameter in native.items():
        assert parameters[name].name == parameter.name
        assert parameters[name].kind == parameter.kind
        assert parameters[name].default == parameter.default

def test_fixed_runner_without_autotune(monkeypatch):
    call = _fly_call(tokens=1)
    reference = torch_reference(call)
    config = dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=64,
                  tile_n_gate=128, tile_n_down=128, tile_k_gate=64)
    runner = make_moe_runner(config)

    def unexpected(*args, **kwargs):
        pytest.fail("fixed kernel tests must not tune, enumerate candidates or fall back to Aiter")

    monkeypatch.setattr(tm, "_configs", unexpected)
    monkeypatch.setattr(tm, "_autotuned_fmoe", unexpected)
    monkeypatch.setattr(tm, "_aiter_fused_moe", unexpected)
    monkeypatch.setattr(tm, "record_dispatch", True)
    monkeypatch.setattr(tm, "last_dispatch", None)
    config["_impl"] = "aiter"  # runner 保存静态配置副本，不受调用方后续修改影响。
    _check(runner(**call), call, reference)
    assert tm.last_dispatch["_impl"] == "fly_prefill"
    seen = set()

    def measured(**buffers):
        seen.add((buffers["w1"].data_ptr(), buffers["output"].data_ptr()))
        assert buffers["w1"].is_shuffled and buffers["w2"].is_shuffled
        return runner(**buffers)

    stats = measure_moe(measured, call, reference, copies=2, warmup=1, iters=3)
    assert stats["correctness"]["status"] == "PASS"
    assert len(stats["samples_us"]) == 3 and len(seen) == 3
    assert stats["num_copies"] == 2


@pytest.mark.parametrize("dtype, quant, activation, gate_mode", [
    ("bf16", "model", "silu", "separated"), ("bf16", "model", "gelu", "separated"),
    ("fp8", "ptpc", "silu", "separated"), ("fp8", "per_tensor", "silu", "separated"),
    ("fp8", "block", "silu", "separated"), ("mxfp4", "model", "silu", "separated"),
    ("mxfp4", "model", "situv2", "interleave"),
])
def test_shared_preparation(dtype, quant, activation, gate_mode):
    if dtype == "mxfp4" and not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("MXFP4 requires gfx950")
    model = dict(HIDDEN_SIZE=256, INTER_SIZE=256, TP=1, E=4, TOPK=2)
    options = dict(dtype=dtype, quant=quant, activation=activation, gate_mode=gate_mode,
                   seed=8, preshuffle="on", routing="balanced", beta=None, linear_beta=None,
                   swiglu_limit=None)
    call, kind = prepare_moe(model, 3, **options)
    benchmark_call, benchmark_kind = bench.prepare(model, 3, SimpleNamespace(**options))
    assert kind == benchmark_kind and tm._torch_reference is torch_reference
    for name, value in call.items():
        if isinstance(value, torch.Tensor) and name != "output":
            assert torch.equal(value.view(torch.uint8), benchmark_call[name].view(torch.uint8)), name
    assert call["w1"].is_shuffled and call["w2"].is_shuffled


def test_fly_buffer_byte_offsets():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from pyhip.codegen.flydsl.helpers import BufferTensor

    # 32-bit load 曾因 v1i32 lowering 导致 LLVM 崩溃；BF16 同时覆盖字节/元素偏移。
    dtype, elem = torch.bfloat16, fx.BFloat16
    elements, packet_bytes = 2, 4
    x = (torch.arange(64 * elements) % 29 + 1).to(dtype)
    storage = torch.full((66 * elements,), 113, dtype=dtype)
    out = storage[elements:-elements]

    @flyc.kernel
    def copy_packet(X: fx.Pointer, Y: fx.Pointer):
        root = fx.rocdl.make_buffer_tensor(
            fx.make_view(X, fx.make_layout(64 * elements, 1)), num_records_bytes=62 * packet_bytes)
        # 非零 slice 偏移不能再加到显式、根 descriptor 相对的完整字节偏移上。
        packet = BufferTensor(fx.make_view(fx.get_iter(root) + 2 * elements, fx.make_layout(elements, 1)))
        values = packet.load(voffset_bytes=fx.Int32(fx.thread_idx.x * packet_bytes),
                             soffset_bytes=fx.Int32(packet_bytes), aux=2)
        fx.make_view(Y + fx.thread_idx.x * elements, fx.make_layout(elements, 1)).store(values)

    @flyc.jit
    def launch(X: fx.Pointer, Y: fx.Pointer, stream: fx.Stream):
        copy_packet(X, Y).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    _run_compiled(launch, flyc.from_c_void_p(elem, x.data_ptr()),
                  flyc.from_c_void_p(elem, out.data_ptr()), torch.cuda.current_stream())
    expected = torch.zeros_like(out)
    expected[:61 * elements] = x[elements:62 * elements]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert (storage[:elements] == 113).all() and (storage[-elements:] == 113).all()


@pytest.mark.parametrize("tokens, hidden, inter, shuffled", [
    (1, 256, 256, (False, False)),
    (17, 512, 256, (True, False)),
    (33, 256, 512, (False, True)),
    (513, 512, 512, (True, True)),
    (2049, 4096, 256, (True, True)),
])
def test_gelu_candidate(tokens, hidden, inter, shuffled):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("GELU 8-wave M256/N256 needs 130 KiB LDS")
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2
    from aiter.ops.shuffle import shuffle_weight

    call = _gelu_call(tokens, hidden, inter)
    x, w1, w2, weights, ids = (call[name] for name in
                              ("hidden_states", "w1", "w2", "topk_weight", "topk_ids"))
    mid = torch_moe_stage1(x, w1, w2, weights, ids, dtype=x.dtype, activation=aiter.ActivationType.Gelu)
    expected = torch_moe_stage2(mid, w1, w2, weights, ids, dtype=x.dtype)
    for name, enabled in zip(("w1", "w2"), shuffled):
        if enabled:
            call[name] = shuffle_weight(call[name])
    before = {name: call[name].clone() for name in ("w1", "w2")}
    reference = tm._torch_reference(call)
    assert calc_diff(expected, reference) <= 1e-6
    tuning = _tune_arguments(call)
    configs = tm._configs(**tuning)
    assert [c.all_kwargs()["_impl"] for c in configs] == ["aiter", "jit_gelu"]
    call["output"].fill_(float("nan"))
    _check(tm._fmoe_wrapper(**tuning, **configs[1].all_kwargs()), call, expected)
    for name, value in before.items():
        assert torch.equal(call[name], value)
    assert tuple(bool(getattr(call[name], "is_shuffled", False)) for name in ("w1", "w2")) == shuffled


def test_gelu_config_constraints(monkeypatch):
    call = _gelu_call()

    def has_native(call):
        return any(c.all_kwargs()["_impl"] == "jit_gelu" for c in tm._configs(**_tune_arguments(call)))

    if torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        assert has_native(call)
    assert not has_native(call | {"block_size_M": 128})
    assert not has_native(_gelu_call(hidden=384))
    assert not has_native(_gelu_call(inter=384))
    assert not has_native(call | {"w1": torch.cat((call["w1"], call["w1"]), dim=1)})
    assert not has_native(call | {"quant_type": aiter.QuantType.per_Token})
    assert not has_native(call | {"swiglu_limit": 1.0})
    assert not has_native(call | {"expert_mask": torch.ones(4, dtype=torch.int32)})
    # 不分配数 GB tensor：同 bucket 最大 M 的输入也必须满足 buffer 地址范围。
    tuning = _tune_arguments(call) | {"batch_bucket": 1 << 21}
    assert [c.all_kwargs()["_impl"] for c in tm._configs(**tuning)] == ["aiter"]
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(
        name="AMD Instinct MI300X", gcnArchName="gfx942", multi_processor_count=304,
    ))
    assert not has_native(call)


def test_gelu_tuning_cache_graph(monkeypatch, tmp_path):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tm, "record_dispatch", True)
    monkeypatch.setattr(tm, "last_dispatch", None)
    call = _gelu_call(tokens=33)
    # Parameter 不带 is_shuffled 时仍按 raw 处理，不猜测实际存储。
    call["w1"] = torch.nn.Parameter(call["w1"], requires_grad=False)
    reference = tm._torch_reference(call)
    _check(tm.fused_moe(**call), call, reference)
    winner = tm.last_dispatch.copy()
    assert winner["_impl"] in ("aiter", "jit_gelu")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")

    def unexpected_search(*args, **kwargs):
        pytest.fail("GELU cache hit must not retune")

    monkeypatch.setattr(tm._autotuned_fmoe, "configs", unexpected_search)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            tm.fused_moe(**call)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            tm.fused_moe(**call)
            result = tm.fused_moe(**call)
    stream.synchronize()
    assert tm.last_dispatch == winner
    for zero in (False, True, False):
        call["hidden_states"].fill_(0.0 if zero else .1)
        call["topk_ids"].copy_((call["topk_ids"] + 1) % 4)
        call["topk_weight"].mul_(-.5)
        reference = tm._torch_reference(call)
        call["output"].fill_(float("nan"))
        graph.replay()
        _check(result, call, reference)


def test_gelu_benchmark_shape():
    call = _gelu_call()
    assert call["w1"].shape == (4, 256, 512)
    assert call["w2"].shape == (4, 512, 256)
    args = SimpleNamespace(dtype="bf16", quant="model", activation="gelu", gate_mode="separated")
    shape = bench._aiter_shape(bench.MOE_MODELS["qwen35_35B_k256"], 3, args)
    assert shape["act_type"] == aiter.ActivationType.Gelu and not shape["use_g1u1"]
    assert shape["q_type"] == aiter.QuantType.No and shape["token"] == 4
    with pytest.raises(SystemExit) as error:
        bench.main(["--activation", "gelu", "--dtype", "bf16", "--tune-aiter"])
    assert error.value.code == 2


def test_dispatch_recording(monkeypatch):
    assert tm.record_dispatch is False
    call = _block_call()
    tuning = _tune_arguments(call)
    stale = {"_impl": "previous"}
    monkeypatch.setattr(tm, "last_dispatch", stale)
    monkeypatch.setattr(tm, "_run_jit", lambda *args: call["output"])
    monkeypatch.setattr(tm, "_aiter_fused_moe", lambda *args, **kwargs: kwargs["output"])
    tm._fmoe_wrapper(**tuning, _impl="jit_batch")
    assert tm.last_dispatch is stale

    monkeypatch.setattr(tm, "record_dispatch", True)
    tm._fmoe_wrapper(**tuning, _impl="jit_batch", tile_m_gate=32, tile_n_down=128, block_n=1024)
    assert tm.last_dispatch == dict(_impl="jit_batch", tile_m_gate=32, tile_m_down=16,
                                    tile_n_gate=64, tile_n_down=128, block_n=1024,
                                    decode_alg="splitk", down_path="default", padding=None, num_oc_splits=1)
    assert all(value is None or type(value) in (str, int, float, bool)
               for value in tm.last_dispatch.values())
    tm._fmoe_wrapper(**tuning, _impl="aiter")
    assert tm.last_dispatch == {"_impl": "aiter"}
    tm.last_dispatch = stale
    # 不支持本地调优的参数直接转交 Aiter，也必须覆盖旧记录。
    assert tm.fused_moe(**call, hidden_pad=128) is call["output"]
    assert tm.last_dispatch == {"_impl": "aiter"}
    monkeypatch.setattr(tm, "record_dispatch", False)
    tm.last_dispatch = stale
    tm.fused_moe(**call, hidden_pad=128)
    assert tm.last_dispatch is stale


def test_dispatch_winner_cache_and_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "configs"))
    monkeypatch.setattr(tm, "record_dispatch", True)
    monkeypatch.setattr(tm, "last_dispatch", None)
    call = _block_call()
    dispatches = []

    def run(*args):
        dispatches.append(tm.last_dispatch["_impl"])
        return call["output"]

    def do_bench(fn, **kwargs):
        fn()
        return 1.0 if tm.last_dispatch["_impl"] == "fly_prefill" else 2.0

    def tuner(configs):
        return tm.autotune(configs=configs, key=["batch_bucket", "model_key"], do_bench=do_bench,
                           artifact_name="pyhip_dispatch_test")(tm._fmoe_wrapper)

    def unexpected_search(*args, **kwargs):
        pytest.fail("cache/artifact hit must not search candidates")

    monkeypatch.setattr(tm, "_run_jit", run)
    monkeypatch.setattr(tm, "_run_fly", run)
    monkeypatch.setattr(tm, "_autotuned_fmoe", tuner([
        tm.Config(_impl="fly_prefill", tile_k_gate=256), tm.Config(_impl="jit_splitk")]))
    assert tm.fused_moe(**call) is call["output"]
    # 最后一个被计时的候选不是 winner；正式调用必须覆盖它。
    assert dispatches == ["fly_prefill", "jit_splitk", "fly_prefill"]
    winner = tm.last_dispatch.copy()
    assert winner["_impl"] == "fly_prefill" and winner["tile_k_gate"] == 256
    assert bench.json.loads(_tune_arguments(call)["model_key"])["version"] == 7
    assert len(list((tmp_path / "configs").glob("pyhip_dispatch_test-*.json"))) == 1

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setattr(tm._autotuned_fmoe, "configs", unexpected_search)
    dispatches.clear()
    tm.last_dispatch = None
    tm.fused_moe(**call)
    assert dispatches == ["fly_prefill"] and tm.last_dispatch == winner

    # 新 autotuner + 空普通缓存目录，只能从离线 artifact 命中。
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(tm, "_autotuned_fmoe", tuner(unexpected_search))
    dispatches.clear()
    tm.last_dispatch = None
    tm.fused_moe(**call)
    assert dispatches == ["fly_prefill"] and tm.last_dispatch == winner


@pytest.mark.parametrize("failed, previous_record", [(False, False), (False, True), (True, False), (True, True)])
def test_benchmark_dispatch_recording(monkeypatch, failed, previous_record):
    call = _block_call()
    model = dict(HIDDEN_SIZE=512, INTER_SIZE=256, TP=1, E=4, TOPK=2)
    args = SimpleNamespace(dtype="fp8", gate_mode="separated", preshuffle="on", routing="balanced", seed=8,
                           activation="silu", swiglu_limit=None, beta=None, linear_beta=None,
                           retune=False, tune_aiter=None, check_only=False, rounds=1)
    monkeypatch.setattr(tm, "record_dispatch", previous_record)
    monkeypatch.setattr(tm, "last_dispatch", {"_impl": "stale"})
    monkeypatch.setattr(tm, "_autotuned_fmoe", object())  # benchmark 不能查询 autotuner 内部状态。
    monkeypatch.setattr(bench, "prepare", lambda *args: (call, "block"))
    monkeypatch.setattr(bench, "torch_reference", lambda call: torch.zeros_like(call["hidden_states"]))

    def aiter_call(**call):
        assert not tm.record_dispatch
        return call["output"].zero_()

    def tuned_call(**call):
        assert tm.record_dispatch and tm.last_dispatch is None
        tm.last_dispatch = {"_impl": "recorded"}
        if failed:
            raise RuntimeError("intentional dispatch failure")
        return call["output"].zero_()

    def measure(*args, **kwargs):
        assert not failed and not tm.record_dispatch
        tm.last_dispatch["_impl"] = "later"  # row 保存独立快照，不能跟着全局容器变化。
        return dict(samples_us=[1.0], mean_us=1.0, correctness=dict(status="PASS", diff=0.0))

    monkeypatch.setattr(tm, "_aiter_fused_moe", aiter_call)
    monkeypatch.setattr(tm, "fused_moe", tuned_call)
    monkeypatch.setattr(bench, "measure", measure)
    row = bench.run_case("test", model, 17, args)
    assert row["status"] == ("NOT_COMPARABLE" if failed else "PASS")
    assert row["winner"] == (None if failed else {"_impl": "recorded"})
    assert tm.record_dispatch is previous_record


@pytest.mark.parametrize("case, check_only, expected", [
    ("pass", False, "PASS"),
    ("aiter_bad", False, "AITER_INCORRECT"),
    ("aiter_nan", False, "AITER_INCORRECT"),
    ("aiter_timed_bad", False, "AITER_INCORRECT"),
    ("aiter_bad", True, "AITER_INCORRECT"),
    ("pass", True, "CHECK_PASS"),
    ("aiter_error", False, "NOT_COMPARABLE"),
    ("aiter_metadata", False, "NOT_COMPARABLE"),
    ("winner_bad", False, "NOT_COMPARABLE"),
    ("winner_timed_bad", False, "ERROR"),
])
def test_benchmark_accuracy_and_speedup(monkeypatch, capsys, case, check_only, expected):
    import pyhip

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    call = dict(hidden_states=x, output=torch.empty_like(x))
    model = dict(HIDDEN_SIZE=4, INTER_SIZE=4, TP=1, E=2, TOPK=1)
    args = SimpleNamespace(dtype="bf16", gate_mode="separated", preshuffle="on", routing="balanced", seed=0,
                           activation="silu", swiglu_limit=None, beta=None, linear_beta=None,
                           retune=False, tune_aiter=None, check_only=check_only, rounds=2,
                           copies=2, warmup=0, iters=2)
    monkeypatch.setattr(bench, "prepare", lambda *args: (call, "bf16"))
    monkeypatch.setattr(bench, "torch_reference", lambda call: x)
    monkeypatch.setattr(tm, "record_dispatch", False)
    monkeypatch.setattr(tm, "last_dispatch", None)
    calls = {"aiter": 0, "winner": 0}
    timed = []

    def invoke(backend, buffers):
        calls[backend] += 1
        if backend == "winner" and tm.record_dispatch:
            tm.last_dispatch = dict(_impl="test")
        output = buffers["output"].fill_(1)
        if case == f"{backend}_error":
            raise RuntimeError("intentional execution error")
        if case == f"{backend}_metadata":
            return output.clone()
        if case == f"{backend}_nan":
            output.fill_(float("nan"))
        elif case == f"{backend}_bad" or (case == f"{backend}_timed_bad" and calls[backend] == 2):
            output.zero_()
        return output

    def fake_perftest(op, *, num_iters, num_warmup, num_copies, num_stats, **buffers):
        assert not tm.record_dispatch
        timed.append(True)
        # 两个实际输出副本：其中一个坏，下一轮全好，也不能把失败覆盖成 PASS。
        for _ in range(num_copies):
            result = op(**(buffers | {"output": buffers["output"].clone()}))
        num_stats.update(samples_us=[2.0] * num_iters)
        return result, 2.0

    monkeypatch.setattr(tm, "_aiter_fused_moe", lambda **buffers: invoke("aiter", buffers))
    monkeypatch.setattr(tm, "fused_moe", lambda **buffers: invoke("winner", buffers))
    monkeypatch.setattr(pyhip, "run_perftest", fake_perftest)
    row = bench.run_case("test", model, 2, args)
    assert row["status"] == expected
    if not check_only and expected in ("PASS", "AITER_INCORRECT"):
        assert row["speedup"] == 1.0 and len(timed) == 4
        assert row["correctness"]["tuned"]["diff"] == 0
        if case != "pass":
            assert row["correctness"]["aiter"]["status"] == "INCORRECT"
            assert "timings only" in row["reason"]
    else:
        assert "speedup" not in row
        if expected != "ERROR":
            assert not timed
    bench.json.dumps(row, allow_nan=False)
    bench.print_table([row])
    text = capsys.readouterr().out
    assert "| Aiter diff | winner diff |" in text
    if case == "aiter_nan":
        assert "| NaN/Inf | 0 |" in text
    elif case in ("aiter_bad", "aiter_timed_bad"):
        assert "| 1 | 0 |" in text
    elif case in ("aiter_error", "aiter_metadata"):
        assert "| — | 0 |" in text


@pytest.mark.parametrize("dtype, quant, expected", [
    ("bf16", "model", aiter.QuantType.No),
    ("fp8", "ptpc", aiter.QuantType.per_Token),
    ("fp8", "per_tensor", aiter.QuantType.per_Tensor),
    ("fp8", "block", aiter.QuantType.per_1x128),
    ("mxfp4", "model", aiter.QuantType.per_1x32),
])
def test_aiter_tuning_shape(dtype, quant, expected):
    args = SimpleNamespace(dtype=dtype, quant=quant, activation="silu", gate_mode="separated")
    model = bench.MOE_MODELS["qwen35_397B_k256"]
    shape = bench._aiter_shape(model, 3, args)
    assert shape["token"] == 4
    assert shape["model_dim"] == 4096 and shape["inter_dim"] == 256
    assert shape["q_type"] == expected
    # dtype 是输出类型，不是量化权重类型。
    assert shape["dtype"] == torch.bfloat16
    assert shape["q_dtype_a"] == shape["q_dtype_w"] == bench._moe_types(model, args)[1]
    assert shape["use_g1u1"] and not shape["doweight_stage1"]


def test_aiter_tuning_prepass(monkeypatch, tmp_path):
    import aiter.fused_moe as fm
    from aiter.jit.core import AITER_CONFIGS

    args = SimpleNamespace(dtype="fp8", quant="model", activation="silu", gate_mode="separated",
                           tokens=[3, 4, 64, 64], tune_aiter=tmp_path, routing="balanced", seed=8,
                           beta=None, linear_beta=None, swiglu_limit=None)
    models = ["hy3", "qwen35_35B_k256"]
    calls = []

    def fake_tuner(command, *, cwd, env, check):
        assert check and "--all" in command and command[command.index("--mp") + 1] == "1"
        assert env["TUNE_ONLY"] == "" and env["TUNE_MOE_KERNEL_REGEX"] == ""
        assert env["TUNE_MOE_EXPERT_BALANCE"] == "True"
        with (tmp_path / "untuned.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4  # 两个模型 × 两个 Aiter bucket，重复 M 不重复调优。
        assert {row["q_type"] for row in rows} == {"QuantType.per_Tensor", "QuantType.per_Token"}
        assert {row["inter_dim"] for row in rows} == {"192", "256"}  # 不改 Hy3 的实际 I。
        calls.append(rows)
        for row in rows:
            row.update(gfx=fm.get_gfx_runtime(), cu_num=fm.get_cu_num(), block_m=32, ksplit=0,
                       kernelName1=f"moe_ck2stages_test_{len(calls)}_gemm1",
                       kernelName2=f"moe_ck2stages_test_{len(calls)}_gemm2", run_1stage=0, us=1)
        if len(calls) == 3:
            rows.pop()  # 模拟官方 tuner 未为全部 shape 写入结果。
        with (tmp_path / "tuned.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    monkeypatch.setattr(bench, "subprocess", SimpleNamespace(run=fake_tuner))
    monkeypatch.setenv("AITER_CONFIG_FMOE", "previous-config.csv")
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    # 相同路径调优两次仍要刷新 Python caches；不读取上一次结果。
    for run in (1, 2):
        with bench.tune_aiter(models, args) as path:
            assert path == str(tmp_path / "tuned.csv")
            assert AITER_CONFIGS.AITER_CONFIG_FMOE_FILE == path
            assert all(row["kernelName1"] == f"moe_ck2stages_test_{run}_gemm1"
                       for row in fm.cfg_2stages[0].values())
        assert fm.cfg_2stages is None
        assert bench.os.environ["AITER_CONFIG_FMOE"] == "previous-config.csv"
        assert bench.os.environ["AITER_BYPASS_TUNE_CONFIG"] == "1"
    with pytest.raises(RuntimeError, match="no valid tuned config"):
        with bench.tune_aiter(models, args):
            pytest.fail("must not benchmark with a missing tuned shape")
    assert fm.cfg_2stages is None
    assert bench.os.environ["AITER_CONFIG_FMOE"] == "previous-config.csv"


@pytest.mark.parametrize("quant_type", [aiter.QuantType.per_128x128, aiter.QuantType.per_1x128])
def test_blockscale_reference(quant_type):
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

    call = _block_call(shuffled=(False, False))
    call["quant_type"] = quant_type
    x, w1, w2 = (call[name] for name in ("hidden_states", "w1", "w2"))
    weights, ids = call["topk_weight"], call["topk_ids"]
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x128)
    aq, a_scale = quantize(x, quant_dtype=w1.dtype)
    mid = torch_moe_stage1(aq, w1, w2, weights, ids, dtype=x.dtype, quant_type=quant_type,
                           a1_scale=a_scale, w1_scale=call["w1_scale"])
    dq, d_scale = quantize(mid.view(-1, mid.shape[-1]), quant_dtype=w2.dtype)
    reference = torch_moe_stage2(dq, w1, w2, weights, ids, dtype=x.dtype, quant_type=quant_type,
                                a2_scale=d_scale, w2_scale=call["w2_scale"])
    actual = tm._torch_reference(call)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("shuffled", [(False, False), (True, False), (False, True), (True, True)])
def test_8wave_candidates(shuffled):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("these 8-wave tiles require gfx950")
    from aiter.ops.shuffle import shuffle_weight

    call, _ = prepare_moe(dict(HIDDEN_SIZE=1024, INTER_SIZE=256, TP=1, E=4, TOPK=2),
                           17, dtype="bf16", seed=8, preshuffle="off")
    call["hidden_states"].mul_(50)
    for name, enabled in zip(("w1", "w2"), shuffled):
        if enabled:
            call[name] = shuffle_weight(call[name])
    tuning = _tune_arguments(call)
    configs = [c for c in tm._configs(**tuning) if c.all_kwargs()["_impl"] == "jit_8wave"]
    assert len(configs) == (7 if shuffled[1] else 4)
    reference = torch_reference(call)
    before = {name: call[name].clone() for name in ("w1", "w2")}
    for config in configs:
        call["output"].fill_(float("nan"))
        _check(tm._fmoe_wrapper(**tuning, **config.all_kwargs()), call, reference)
    for name, value in before.items():
        assert torch.equal(value, call[name])
    assert tuple(bool(getattr(call[name], "is_shuffled", False)) for name in ("w1", "w2")) == shuffled


def test_migrated_config_constraints(monkeypatch):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")

    def native(call, impl):
        return [c.all_kwargs() for c in tm._configs(**_tune_arguments(call))
                if c.all_kwargs()["_impl"].startswith(impl)]

    bf16 = _fly_call(tokens=33, hidden=1024, inter=256)
    limited = native(bf16 | {"block_size_M": 128}, "jit_8wave")
    assert len(limited) == 2 and all(c["tile_m_down"] == 128 for c in limited)
    fp4 = _fly_call(tokens=33, hidden=1024, inter=256, kind="fp4")
    configs = native(fp4, "jit_mxfp4")
    assert len(configs) == 5
    assert {(c["tile_m_down"], c["tile_n_gate"]) for c in configs
            if c["_impl"] == "jit_mxfp4_4wave"} == {(m, n) for m in (128, 256) for n in (128, 256)}
    assert all(c["tile_m_down"] == 128 for c in native(fp4 | {"block_size_M": 128}, "jit_mxfp4"))
    # final_reduce 的限制不影响 GEMM 候选；运行时可改用 torch.sum。
    for hidden in (256, 512, 1536):
        assert native(_fly_call(hidden=hidden, inter=256, kind="fp4"), "jit_mxfp4") == configs
    assert not native(_fly_call(kind="fp4", gate_mode="interleave"), "jit_mxfp4")
    fp4["w1"].is_shuffled = False
    assert not native(fp4, "jit_mxfp4")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(
        name="AMD Instinct MI300X", gcnArchName="gfx942", multi_processor_count=304,
    ))
    assert not native(bf16, "jit_8wave")


@pytest.mark.parametrize("hidden", [256, 512, 1024, 1536, 2048], ids=lambda h: f"h{h}")
@pytest.mark.parametrize("impl", ["jit_mxfp4", "jit_mxfp4_4wave"])
def test_mxfp4_final_reduce_dispatch(hidden, impl, monkeypatch):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    from pyhip.ops.moe.asm import moe_gemm_mxfp4 as asm

    call = _fly_call(tokens=17, hidden=hidden, inter=256, kind="fp4")
    reference = torch_reference(call, mxfp4_activations=True)
    runner = make_moe_runner(dict(_impl=impl, tile_m_gate=128, tile_m_down=128,
                                  tile_n_gate=128, tile_n_down=128))
    original_reduce = asm.moe_gemm_final_reduce_bf16
    seen = []

    def reduce(*args):
        assert hidden % 1024 == 0, "unsupported H must use torch.sum"
        seen.append(args[3])
        return original_reduce(*args)

    monkeypatch.setattr(asm, "moe_gemm_final_reduce_bf16", reduce)
    call["output"].fill_(float("nan"))
    _check(runner(**call), call, reference)
    assert seen == ([hidden] if hidden % 1024 == 0 else [])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = runner(**call)
    call["output"].fill_(float("nan"))
    graph.replay()
    _check(result, call, reference)


def test_mxfp4_pruning_keeps_public_reference():
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    call, _ = prepare_moe(dict(HIDDEN_SIZE=1024, INTER_SIZE=256, TP=1, E=4, TOPK=2),
                           17, dtype="mxfp4", seed=8, structured_scales=True)
    tuning = _tune_arguments(call)
    configs = tm._configs(**tuning)
    a4w4 = [c for c in configs if c.all_kwargs()["_impl"] in ("jit_mxfp4", "jit_mxfp4_4wave")]
    good = next(c for c in configs if c.all_kwargs() == dict(
        _impl="jit_splitk", tile_m_gate=16, tile_m_down=16, tile_n_gate=64, tile_n_down=64))
    reference = torch_reference(call)
    a4w4_reference = torch_reference(call, mxfp4_activations=True)
    # A4W4 自身正确不等于满足公共 A16W4 精度要求；调优不能换参考来放行。
    assert calc_diff(reference, a4w4_reference) > .02
    for config in a4w4:
        _check(tm._fmoe_wrapper(**tuning, **config.all_kwargs()), call, a4w4_reference)
    assert tm._prune_invalid_configs([*a4w4, good], tuning) == [good]
    with pytest.raises(ValueError, match="requires MXFP4 weights"):
        torch_reference(_fly_call(), mxfp4_activations=True)


@pytest.mark.parametrize("kind, config", [
    ("bf16", dict(_impl="jit_8wave", tile_m_gate=128, tile_m_down=128, tile_n_gate=256, tile_n_down=128)),
    ("bf16", dict(_impl="jit_8wave", tile_m_gate=256, tile_m_down=256, tile_n_gate=256,
                  tile_n_down=64, down_path="persistent", num_oc_splits=4)),
    ("mxfp4", dict(_impl="jit_mxfp4", tile_m_gate=128, tile_m_down=128, tile_n_gate=128, tile_n_down=128)),
    ("mxfp4", dict(_impl="jit_mxfp4_4wave", tile_m_gate=256, tile_m_down=256, tile_n_gate=256, tile_n_down=128)),
], ids=["bf16-default", "bf16-persistent", "mxfp4-generic", "mxfp4-4wave"])
def test_migrated_stream_graph(kind, config):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    call, _ = prepare_moe(dict(HIDDEN_SIZE=1024, INTER_SIZE=256, TP=1, E=4, TOPK=2),
                           33, dtype=kind, seed=8, structured_scales=kind == "mxfp4")
    call["hidden_states"].mul_(50)
    runner = make_moe_runner(config)
    reference = torch_reference(call, mxfp4_activations=kind == "mxfp4")
    stats = measure_moe(runner, call, reference, copies=2, warmup=1, iters=3)
    assert stats["correctness"]["status"] == "PASS" and len(stats["samples_us"]) == 3
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            runner(**call)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner(**call)
            result = runner(**call)
    stream.synchronize()
    original = call["hidden_states"].clone()
    for factor in (0.0, -.75, 1.0):
        call["hidden_states"].copy_(original * factor)
        call["topk_ids"].copy_((call["topk_ids"] + 1) % 4)
        call["topk_weight"].mul_(-.5)
        reference = torch_reference(call, mxfp4_activations=kind == "mxfp4")
        call["output"].fill_(float("nan"))
        graph.replay()
        _check(result, call, reference)


@pytest.mark.parametrize("kind, impl", [("bf16", "jit_8wave"), ("mxfp4", "jit_mxfp4_4wave")])
def test_migrated_tuning_cache(kind, impl, monkeypatch, tmp_path):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    call, _ = prepare_moe(dict(HIDDEN_SIZE=1024, INTER_SIZE=256, TP=1, E=4, TOPK=2),
                           17, dtype=kind, seed=8)
    call["hidden_states"].fill_(.0625)
    if kind == "mxfp4":
        # 精确可表示的输入/权重，使 A4W4 也通过原公共参考门槛。
        for name in ("w1", "w2"):
            call[name].view(torch.uint8).fill_(0x22)
            call[f"{name}_scale"].view(torch.uint8).fill_(119)
    configs = [c for c in tm._configs(**_tune_arguments(call)) if c.all_kwargs()["_impl"] == impl]
    assert configs
    tuner = tm.autotune(configs=configs, key=["batch_bucket", "model_key"],
                        prune_configs_by=tm._prune_invalid_configs,
                        artifact_name=f"migration_{impl}")(tm._fmoe_wrapper)
    monkeypatch.setattr(tm, "_autotuned_fmoe", tuner)
    monkeypatch.setattr(tm, "record_dispatch", True)
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    _check(tm.fused_moe(**call), call, torch_reference(call))
    winner = tm.last_dispatch.copy()
    assert winner["_impl"] == impl

    def unexpected(*args, **kwargs):
        pytest.fail("cache hit must not retune")

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setattr(tuner, "configs", unexpected)
    call["hidden_states"].mul_(-.5)
    call["topk_ids"].copy_((call["topk_ids"] + 1) % 4)
    call["output"].fill_(float("nan"))
    _check(tm.fused_moe(**call), call, torch_reference(call))
    assert tm.last_dispatch == winner


@pytest.mark.parametrize("tokens, hidden, inter, shuffled", [
    (1, 512, 128, (False, False)),
    (17, 512, 128, (True, False)),
    (33, 512, 128, (True, True)),
    (33, 512, 256, (False, True)),
    (513, 512, 256, (True, True)),
    (129, 384, 384, (True, True)),
    (33, 128, 128, (False, False)),
    (17, 512, 512, (True, True)),
])
def test_blockscale_candidates(tokens, hidden, inter, shuffled):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native block-scale 8-wave kernels require gfx950")
    call = _block_call(tokens, hidden, inter, shuffled)
    tuning = _tune_arguments(call)
    configs = _native_configs(call)
    assert configs
    reference = tm._torch_reference(call)
    before = {name: call[name].view(torch.uint8).clone()
              for name in ("w1", "w2", "w1_scale", "w2_scale")}
    for config in configs:
        call["output"].fill_(float("nan"))
        _check(tm._fmoe_wrapper(**tuning, **config.all_kwargs()), call, reference)
    for name, value in before.items():
        assert torch.equal(call[name].view(torch.uint8), value)
    assert tuple(bool(getattr(call[name], "is_shuffled", False)) for name in ("w1", "w2")) == shuffled


def test_blockscale_config_constraints(monkeypatch):
    call = _block_call()
    call["block_size_M"] = 128
    configs = _native_configs(call)
    if torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        assert configs
    assert all(config.all_kwargs()["tile_m_down"] == 128 for config in configs)
    assert all(config.all_kwargs().get("down_path") != "persistent" for config in configs)
    # gfx942 可继续走现有 split-K/Aiter，不能发射 gfx950 的 FP8 指令。
    call["w1"] = call["w1"].float().to(torch.float8_e4m3fnuz)
    call["w2"] = call["w2"].float().to(torch.float8_e4m3fnuz)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(
        name="AMD Instinct MI300X", gcnArchName="gfx942", multi_processor_count=304,
    ))
    assert not _native_configs(call)


def test_blockscale_tuning_cache(monkeypatch, tmp_path):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    monkeypatch.setattr(tm, "record_dispatch", True)
    monkeypatch.setattr(tm, "last_dispatch", None)
    call = _block_call(tokens=33, shuffled=(True, False))
    call["block_size_M"] = 128
    reference = tm._torch_reference(call)
    _check(tm.fused_moe(**call), call, reference)
    winner = tm.last_dispatch.copy()
    assert any(all(winner[name] == value for name, value in config.all_kwargs().items())
               for config in tm._configs(**_tune_arguments(call)))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")

    def unexpected_search(*args, **kwargs):
        pytest.fail("a cached call must not search configs again")

    monkeypatch.setattr(tm._autotuned_fmoe, "configs", unexpected_search)
    # 同 shape 的数据/路由更新不能被缓存吞掉；out 必须重新覆盖。
    call["hidden_states"].mul_(-.75)
    call["topk_ids"].copy_((call["topk_ids"] + 1) % 4)
    call["topk_weight"].mul_(.5)
    call["output"].fill_(float("nan"))
    reference = tm._torch_reference(call)
    tm.last_dispatch = None
    _check(tm.fused_moe(**call), call, reference)
    assert tm.last_dispatch == winner
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = tm.fused_moe(**call)
    call["output"].fill_(float("nan"))
    tm.last_dispatch = None
    graph.replay()
    _check(result, call, reference)
    assert tm.last_dispatch is None  # replay 不执行 Python dispatcher。


@pytest.mark.parametrize("splits", [0, 1, 2, 4])
def test_blockscale_stream_graph(splits):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    call = _block_call(tokens=33, hidden=1024, inter=128)
    config = next(config for config in _native_configs(call)
                  if config.all_kwargs().get("num_oc_splits", 0) == splits)
    tuning = _tune_arguments(call)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            tm._fmoe_wrapper(**tuning, **config.all_kwargs())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            tm._fmoe_wrapper(**tuning, **config.all_kwargs())
            result = tm._fmoe_wrapper(**tuning, **config.all_kwargs())
    stream.synchronize()
    for zero in (False, True, False):
        call["hidden_states"].fill_(0.0 if zero else .05)
        call["topk_ids"].copy_((call["topk_ids"] + 1) % 4)
        reference = tm._torch_reference(call)
        call["output"].fill_(float("nan"))
        graph.replay()
        _check(result, call, reference)