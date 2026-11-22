"""固定配置的 MoE kernel pytest；不运行 autotune，也不依赖其候选列表。

每个 batch/dtype/layout 是独立 node，可用 -k 或 node id 选择。
普通用例只验正确性，jit_blockscale 同时打印耗时和 diff；
-m perf 显式运行大 shape 和其余固定路径的多 buffer 计时。
FlyDSL 两阶段底座的正确性用例在 test_fly_2stage_basic.py / test_fly_2stage_extra.py。
"""

import json
import math
import statistics

import pytest
import torch

from pyhip.testing.moe import check_output, make_moe_runner, measure_moe, prepare_moe, torch_reference
from pyhip.testing.moe_shapes import MOE_MODELS


# 保留旧脚本的完整 batch 边界集合，去掉重复的 6144，不再藏在一次测试的循环中。
SPLITK_TOKENS = sorted(set(
    list(range(2, 64)) + list(range(128, 256)) + [256, 512, 768, 2048, 4096, 6144, 8192]
    + list(range(6144, 6400))))
DECODE_MODEL = dict(HIDDEN_SIZE=4096, INTER_SIZE=1024, TP=8, E=64, TOPK=10)
SMALL_MODEL = dict(HIDDEN_SIZE=1024, INTER_SIZE=1024, TP=8, E=8, TOPK=4)
SPLITK_CONFIG = dict(_impl="jit_splitk", tile_m_gate=16, tile_m_down=16,
                      tile_n_gate=64, tile_n_down=64)


@pytest.fixture(autouse=True)
def _gpu_environment(monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("MoE kernels target gfx942/gfx950")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # 测当前源码，不复用旧磁盘编译产物。
    monkeypatch.delenv("MOE_PREFILL_TILE_K", raising=False)
    try:
        yield
    finally:
        torch.set_default_device(previous)


def _prepare(model, tokens, precision, **options):
    if precision == "mxfp4" and not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("MXFP4 requires gfx950")
    dtype = precision if precision in ("bf16", "mxfp4") else "fp8"
    quant = "model" if dtype != "fp8" else precision
    call, _ = prepare_moe(model, tokens, dtype=dtype, quant=quant, seed=8,
                          structured_scales=precision == "mxfp4", **options)
    return call


def _check(call, config, *, mxfp4_activations=False):
    runner = make_moe_runner(config)
    reference = torch_reference(call, mxfp4_activations=mxfp4_activations)
    assert torch.isfinite(reference).all()
    call["output"].fill_(float("nan"))
    check = check_output(runner(**call), call["output"], reference)
    assert check["status"] == "PASS", f"{config}: {check}"
    return runner, reference


def _measure(call, config, record_property, *, mxfp4_activations=False):
    from pyhip.testing import misc

    assert misc.CUDAPERF is None, "unset CUDAPERF when running perf tests"
    runner, reference = _check(call, config, mxfp4_activations=mxfp4_activations)
    stats = measure_moe(runner, call, reference, copies=2, warmup=2, iters=10)
    samples = stats["samples_us"]
    assert len(samples) == 10 and all(math.isfinite(t) and t > 0 for t in samples)
    us = statistics.median(samples)
    m, h = call["hidden_states"].shape
    topk, inter = call["topk_ids"].shape[1], call["w1"].shape[1] // 2
    tflops = 6 * m * topk * h * inter / (us * 1e6)
    record_property("config", json.dumps(config, sort_keys=True))
    record_property("samples_us", json.dumps(samples))
    record_property("median_us", us)
    record_property("effective_tflops", tflops)
    record_property("diff", stats["correctness"]["diff"])
    print(f"\n{config} M={m} H={h} I={inter} E={call['w1'].shape[0]} topk={topk}: "
          f"{us:.2f} us, {tflops:.3f} TFLOPS, diff={stats['correctness']['diff']:.6g}")


@pytest.mark.parametrize("precision", ["bf16", "ptpc"])
def test_asm_batch1(precision):
    call = _prepare(DECODE_MODEL | dict(E=128, TOPK=8), 1, precision)
    _check(call, dict(_impl="jit_batch1", tile_n_gate=32, tile_n_down=32))


@pytest.mark.parametrize("tokens", range(2, 64), ids=lambda m: f"m{m}")
@pytest.mark.parametrize("precision", ["bf16", "ptpc"])
def test_asm_batch(tokens, precision):
    _check(_prepare(DECODE_MODEL, tokens, precision), dict(_impl="jit_batch", tile_n_gate=32))


@pytest.mark.parametrize("tokens", [2, 4, 7, 8, 17, 32, 63, 64], ids=lambda m: f"m{m}")
def test_asm_batch_loopn(tokens):
    # 显式覆盖 B<8 atomic 与 B>=8 route-output+sum，不再按 CU 数偷偷换 Down。
    _check(_prepare(DECODE_MODEL, tokens, "ptpc"),
           dict(_impl="jit_batch", tile_n_gate=32, block_n=1024))


@pytest.mark.parametrize("tokens", SPLITK_TOKENS, ids=lambda m: f"m{m}")
@pytest.mark.parametrize("precision", ["bf16", "ptpc", "mxfp4"])
def test_asm_splitk(tokens, precision):
    _check(_prepare(DECODE_MODEL, tokens, precision), SPLITK_CONFIG)


@pytest.mark.parametrize("tokens", [1, 17, 513], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("tile_m", [128, 256], ids=lambda m: f"bm{m}")
@pytest.mark.parametrize("tile_n_down", [128, 256], ids=lambda n: f"dn{n}")
@pytest.mark.parametrize("preshuffle", ["off", "on"], ids=["raw", "shuffled"])
def test_jit_8wave(tokens, tile_m, tile_n_down, preshuffle):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("these 8-wave tiles require gfx950")
    model = dict(HIDDEN_SIZE=1024, INTER_SIZE=256, TP=1, E=8, TOPK=4)
    call = _prepare(model, tokens, "bf16", preshuffle=preshuffle)
    call["hidden_states"].mul_(50)
    _check(call, dict(_impl="jit_8wave", tile_m_gate=tile_m, tile_m_down=tile_m,
                      tile_n_gate=256, tile_n_down=tile_n_down))


@pytest.mark.parametrize("tokens", [1, 17, 513], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("inter", [128, 256], ids=lambda i: f"i{i}")
@pytest.mark.parametrize("splits", [1, 2, 4], ids=lambda s: f"split{s}")
def test_jit_8wave_persistent(tokens, inter, splits):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("these 8-wave tiles require gfx950")
    model = dict(HIDDEN_SIZE=1024, INTER_SIZE=inter, TP=1, E=8, TOPK=4)
    call = _prepare(model, tokens, "bf16")
    call["hidden_states"].mul_(50)
    _check(call, dict(_impl="jit_8wave", tile_m_gate=256, tile_m_down=256,
                      tile_n_gate=256, tile_n_down=64, down_path="persistent", num_oc_splits=splits))


@pytest.mark.parametrize("tokens", [1, 17, 513], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("hidden, inter", [
    (256, 256), (512, 256), (1536, 256), (2048, 256), (1024, 256), (1024, 512), (1024, 768),
], ids=["h256-i256", "h512-i256", "h1536-i256", "h2048-i256", "h1024-i256", "h1024-i512", "h1024-i768"])
@pytest.mark.parametrize("impl, tile_m, tile_n_gate", [
    ("jit_mxfp4", 128, 128), ("jit_mxfp4_4wave", 128, 128),
    ("jit_mxfp4_4wave", 128, 256), ("jit_mxfp4_4wave", 256, 128),
    ("jit_mxfp4_4wave", 256, 256),
], ids=["generic", "4wave-m128-n128", "4wave-m128-n256", "4wave-m256-n128", "4wave-m256-n256"])
def test_jit_mxfp4(tokens, hidden, inter, impl, tile_m, tile_n_gate):
    model = dict(HIDDEN_SIZE=hidden, INTER_SIZE=inter, TP=1, E=8, TOPK=4)
    call = _prepare(model, tokens, "mxfp4")
    call["hidden_states"].mul_(50)
    _check(call, dict(_impl=impl, tile_m_gate=tile_m, tile_m_down=tile_m,
                      tile_n_gate=tile_n_gate, tile_n_down=128), mxfp4_activations=True)


def _run_jit_blockscale(model, tokens, tile_m, tile_n_down, down_path, record_property):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native FP8 block-scale kernels require gfx950")
    if not ((down_path == "default" and tile_n_down > 64)
            or (down_path == "persistent" and tile_m == 256 and tile_n_down == 64)):
        pytest.skip("unsupported block-scale Down tile combination")
    call = _prepare(model, tokens, "block")
    call["hidden_states"].mul_(50)  # 覆盖 SiLU 非线性区间及两阶段激活量化。
    _measure(call, dict(_impl="jit_blockscale", tile_m_gate=tile_m, tile_m_down=tile_m,
                        tile_n_gate=256, tile_n_down=tile_n_down, down_path=down_path),
             record_property)


@pytest.mark.parametrize("tokens", [1, 17, 257], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("tile_m", [128, 256], ids=lambda m: f"bm{m}")
@pytest.mark.parametrize("tile_n_down", [64, 128, 256], ids=lambda n: f"dn{n}")
@pytest.mark.parametrize("down_path", ["default", "persistent"], ids=lambda p: f"dp{p}")
def test_jit_blockscale(tokens, tile_m, tile_n_down, down_path, record_property):
    model = dict(HIDDEN_SIZE=1024, INTER_SIZE=256, TP=1, E=8, TOPK=4)
    _run_jit_blockscale(model, tokens, tile_m, tile_n_down, down_path, record_property)


@pytest.mark.parametrize("tokens", [1, 17, 64, 257], ids=lambda m: f"m{m}")
def test_asm_one_stage(tokens):
    _check(_prepare(SMALL_MODEL, tokens, "bf16"), dict(_impl="jit_1stage"))


@pytest.mark.parametrize("tokens", [1, 17, 513], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("preshuffle", ["off", "on"], ids=["raw", "shuffled"])
def test_jit_gelu(tokens, preshuffle):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("GELU 8-wave kernels require gfx950")
    model = dict(HIDDEN_SIZE=512, INTER_SIZE=256, TP=1, E=8, TOPK=4)
    call = _prepare(model, tokens, "bf16", activation="gelu", preshuffle=preshuffle)
    call["hidden_states"].mul_(100)  # 覆盖 GELU 非线性区间。
    _check(call, dict(_impl="jit_gelu", tile_m_gate=256, tile_m_down=256,
                      tile_n_gate=256, tile_n_down=256))


@pytest.mark.perf
@pytest.mark.parametrize("tokens", [8192, 16384], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("model_name", ["qwen35_397B_k256"])
@pytest.mark.parametrize("down_path", ["default", "persistent"])
def test_jit_8wave_perf(model_name, tokens, down_path, record_property):
    call = _prepare(MOE_MODELS[model_name], tokens, "bf16")
    call["hidden_states"].mul_(50)
    _measure(call, dict(_impl="jit_8wave", tile_m_gate=256, tile_m_down=256,
                        tile_n_gate=256, tile_n_down=64 if down_path == "persistent" else 256,
                        down_path=down_path), record_property)


@pytest.mark.perf
@pytest.mark.parametrize("tokens", [1024, 8192], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("model_name", ["qwen35_397B_k256"])
@pytest.mark.parametrize("impl, tile_m, tile_n_gate", [
    ("jit_mxfp4", 128, 128), ("jit_mxfp4_4wave", 256, 256),
], ids=["generic", "4wave"])
def test_jit_mxfp4_perf(model_name, tokens, impl, tile_m, tile_n_gate, record_property):
    call = _prepare(MOE_MODELS[model_name], tokens, "mxfp4")
    call["hidden_states"].mul_(50)
    _measure(call, dict(_impl=impl, tile_m_gate=tile_m, tile_m_down=tile_m,
                        tile_n_gate=tile_n_gate, tile_n_down=128), record_property,
             mxfp4_activations=True)


@pytest.mark.perf
@pytest.mark.parametrize("tokens", [8192, 16384], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("model_name", ["qwen35_397B_k256"])
def test_jit_blockscale_perf(model_name, tokens,  record_property):
    _run_jit_blockscale(MOE_MODELS[model_name], tokens, 256, 64, "persistent",
                       record_property)


@pytest.mark.perf
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 12, 16, 32, 64], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("precision", ["bf16", "ptpc"])
def test_small_batch_perf(tokens, precision, record_property):
    config = (dict(_impl="jit_batch1", tile_n_gate=32, tile_n_down=32) if tokens == 1 else
              dict(_impl="jit_batch", tile_n_gate=32, block_n=1024 if precision == "ptpc" else 0))
    _measure(_prepare(DECODE_MODEL, tokens, precision), config, record_property)


@pytest.mark.perf
@pytest.mark.parametrize("tokens", [1, 2, 4, 8], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("precision", ["bf16", "ptpc"])
def test_fly_batch1_perf(tokens, precision, record_property):
    config = dict(_impl="fly_decode", decode_alg="batch1", tile_n_gate=32, tile_n_down=64)
    _measure(_prepare(DECODE_MODEL, tokens, precision), config, record_property)


@pytest.mark.perf
@pytest.mark.parametrize("tokens", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("precision", ["bf16", "ptpc", "block", "mxfp4"])
def test_splitk_perf(tokens, precision, record_property):
    _measure(_prepare(DECODE_MODEL | dict(E=512), tokens, precision), SPLITK_CONFIG, record_property)


@pytest.mark.perf
@pytest.mark.parametrize("tokens", [64, 1024, 8192], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("model_name", list(MOE_MODELS))
@pytest.mark.parametrize("path, down_m, down_n, padding", [
    ("default", 64, 128, None), ("1x4_64x256", 64, 256, 128), ("8x1_compact", 64, 128, 128),
], ids=["default", "1x4", "compact"])
def test_model_prefill_perf(tokens, model_name, path, down_m, down_n, padding, record_property):
    model = MOE_MODELS[model_name]
    config = dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=down_m,
                  tile_n_gate=128, tile_n_down=down_n, tile_k_gate=128, down_path=path, padding=padding)
    _measure(_prepare(model, tokens, model["quant_type"]), config, record_property)