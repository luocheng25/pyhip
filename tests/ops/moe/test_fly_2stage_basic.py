"""FlyDSL 两阶段 MoE 底座（Aiter gfx9_fp8）的基本功能测试：每条公开路径一例。

每次小改动先跑本文件。固定配置经 ``make_moe_runner`` 执行完整调用（排序、量化、
Gate/Up、Down、规约），与独立 Torch 参考比较（``calc_diff <= 0.02``），并断言实际
启动的 kernel 路径。维度取 Qwen3.8 的 H2560 与 TP2 / TP4 的 I320 / I192，E/topk 缩小
以控制参考耗时。其余覆盖见 test_fly_2stage_extra.py。
"""

import pytest
import torch
from aiter.jit.utils.chip_info import get_cu_num

from pyhip.ops.moe import tuned_moe as tm
from pyhip.ops.moe.flydsl.moe_gemm_2stage import gemm2_8x1_compact
from pyhip.testing.moe import check_output, make_moe_runner, prepare_moe, torch_reference


TP2 = dict(HIDDEN_SIZE=2560, INTER_SIZE=320, TP=1, E=64, TOPK=8)
TP4 = TP2 | dict(INTER_SIZE=192)  # 8x1_compact 不支持 I320。
# 每个 expert 257 行：共 64 个 M256 full 任务和 64 个 M64 tail；在 80 CU 的 MI308X 上 full 不足一轮但超过 3/5，不拆。
PREFILL_TOKENS = 2056


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


def _prepare(model, tokens, precision, **options):
    bf16 = precision == "bf16"
    call, _ = prepare_moe(model, tokens, dtype="bf16" if bf16 else "fp8",
                          quant="model" if bf16 else precision, seed=8, **options)
    call["hidden_states"].mul_(50)  # 进入 SiLU 非线性区，并让中间激活量化有意义。
    return call


def _check(call, config):
    reference = torch_reference(call)
    assert torch.isfinite(reference).all()
    call["output"].fill_(float("nan"))
    check = check_output(make_moe_runner(config)(**call), call["output"], reference)
    assert check["status"] == "PASS", f"{config}: {check}"


def _prefill(down_path, padding):
    # Aiter 调优表在 Qwen3.8 I320/I192 prefill 上选用的 Gate/Up tile：M64、N128、K256。
    return dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=64, tile_n_gate=128,
                tile_n_down=256 if down_path == "1x4_64x256" else 128, tile_k_gate=256,
                down_path=down_path, padding=padding)


def test_batch1(kernels):
    _check(_prepare(TP2, 1, "ptpc"),
           dict(_impl="fly_decode", decode_alg="batch1", tile_n_gate=32, tile_n_down=64))
    assert kernels == [("gateup", "batch1", "default"), ("down", "batch1", "default")]


def test_splitk(kernels):
    _check(_prepare(TP2, 16, "ptpc"), dict(_impl="fly_decode", tile_m_gate=16, tile_m_down=16,
                                           tile_n_gate=64, tile_n_down=64))
    assert kernels == [("gateup", "splitk", "default"), ("down", "splitk", "default")]


@pytest.mark.parametrize("model, down_path, padding", [
    (TP2, "default", None), (TP2, "1x4_64x256", 128), (TP4, "8x1_compact", 128),
], ids=["default", "1x4_64x256", "8x1_compact"])
def test_prefill(kernels, monkeypatch, model, down_path, padding):
    counts = []
    allocate = gemm2_8x1_compact.allocate_task_buffers

    def record(*args):
        buffers = allocate(*args)
        counts.append(buffers[2])
        return buffers

    monkeypatch.setattr(gemm2_8x1_compact, "allocate_task_buffers", record)
    _check(_prepare(model, PREFILL_TOKENS, "ptpc"), _prefill(down_path, padding))
    assert kernels == [("gateup", "prefill_1x4", "default"), ("down", "prefill_1x4", down_path)]
    if down_path == "8x1_compact" and get_cu_num() == 80:
        # GPU 生成的任务表：M256 full 与 M64 tail 的行数都非零，两种 kernel 都执行过。
        assert len(counts) == 1 and bool((counts[0] > 0).all()), counts


def test_prefill_bf16(kernels):
    _check(_prepare(TP2, PREFILL_TOKENS, "bf16"), _prefill("default", None) | dict(tile_k_gate=64))
    assert kernels == [("gateup", "prefill_1x4", "default"), ("down", "prefill_1x4", "default")]
