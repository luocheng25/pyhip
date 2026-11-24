"""MoE A8W4（FP8 激活 × MXFP4 权重）及其 FlyDSL 两阶段底座（Aiter gfx9_fp8）的全部测试，含性能测试。

本任务的测试只放在这一个文件里：新用例加进对应区域，不新建测试文件。文件用注释分成六个区域：
  区域 0  公共：导入、GPU 环境
  区域 1  基本功能 · A8W4：公开入口每个分档一例
  区域 2  基本功能 · 底座：每条公开路径一例
  区域 3  其他覆盖 · A8W4：基本功能与性能测试覆盖不了的情况（布局与量化、decode、流式与 prefill、graph、报错与回退）
  区域 4  其他覆盖 · 底座
  区域 5  性能测试：Qwen3.8 TP2 / TP4（对照 Aiter FP8 ptpc）与 MiMo-V2.6-Flash TP8 / TP4 共 68 个 case（脚本入口，pytest 不收集）
基本功能用例标 ``@pytest.mark.basic``，每次小改动先跑：``pytest tests/ops/moe/test_moe_a8w4.py -m basic``
（myflow 的 commands.correctness）；其他覆盖用 ``-m "not basic"``（commands.verify_extra），每条 docstring
写明基本功能与性能测试为什么覆盖不了；性能测试直接运行本文件：``python tests/ops/moe/test_moe_a8w4.py``
（commands.bench）。
"""

# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# 区域 0｜公共：导入、GPU 环境
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

import argparse
import dataclasses
import functools
import inspect
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import aiter
import pytest
import torch
from aiter.jit.utils.chip_info import get_cu_num

from pyhip.ops.moe import tuned_moe as tm
from pyhip.ops.moe.flydsl import moe_a8w4
from pyhip.ops.moe.flydsl.moe_gemm_2stage import common, gemm2_8x1_compact
from pyhip.testing import moe_a8w4 as oracle
from pyhip.testing.moe import check_output, make_moe_runner, prepare_moe, torch_reference


@pytest.fixture(autouse=True)
def _gpu_environment(monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("the A8W4 and Aiter gfx9_fp8 kernels target gfx942")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # 测当前源码，不复用旧磁盘编译产物。
    try:
        yield
    finally:
        torch.set_default_device(previous)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# 区域 1｜基本功能 · A8W4（@pytest.mark.basic）
# ``prepare_a8w4_weights`` + ``fused_moe_a8w4`` 跑完整调用，与独立 FP32 参考比较（``pyhip.testing.moe_a8w4`` 的
# 判据：calc_diff <= 1e-3 且 max|out-ref| <= 4.5%·max|ref|，无 NaN/Inf），参考的激活精度按算子实际启动的 kernel
# 选择，并断言确实启动了 A8W4 / A16W4 kernel 与预期分档的 kernel。H=2560、topk=11，最后一个 expert 是每个
# token 必选的融合 shared expert；E 缩到 33 以控制参考耗时。三个分档：M≤32 免排序的 decode（TP2 M=3，激活在
# kernel 内量化）；M>32 且每个 expert 平均不超过 22 行（E=33 时 M≤66）的流式路径（TP2 M=64：排序块 32，
# Gate/Up 尾部量化中间激活）；其余走 prefill（TP2 M=1031；TP4 M=257，I=160 由 prepare 补零到 192）。
# prefill 的例子含多块 expert 与部分填充的尾块。MiMo 形状（H=4096、E=256、topk=8，无 shared expert、均匀路由）
# 各取一例：TP8（I=256）decode M=5 与流式 M=300，TP4（I=512）decode M=24 与 prefill M=1600（H=4096 时每个 expert 平均
# 不超过 48 行即 M≤1536 走流式；MiMo 的 decode 按 expert 去重：M=5 两个 kernel、M=24 四个 kernel）。
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

HIDDEN, EXPERTS, TOPK = 2560, 33, 11
MIMO_HIDDEN, MIMO_EXPERTS, MIMO_TOPK = 4096, 256, 8


def _a8w4_prepare(w):
    return moe_a8w4.prepare_a8w4_weights(w["w1"], w["w1_scale"], w["w2"], w["w2_scale"])


def _a8w4_check(inter, tokens, seed, kernel, hidden=HIDDEN, experts=EXPERTS, topk=TOPK, shared=True):
    w = oracle.make_weights(experts, hidden, inter, seed=seed)
    prepared = _a8w4_prepare(w)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, hidden, experts, topk, seed=seed + 1, shared=shared)
    out = torch.full_like(x, float("nan"))
    result = moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out)
    assert result is out
    assert any(kernel in name for name in moe_a8w4.last_path), moe_a8w4.last_path
    mode = oracle.path_activation(moe_a8w4.last_path)
    ref = oracle.reference(x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight, topk_ids,
                           activation=mode)
    oracle.assert_close(out, ref)
    return prepared


# kernel：该分档的 kernel 名里一定出现的片段（decode / 流式 / prefill）。
@pytest.mark.basic
@pytest.mark.parametrize("tokens, kernel", [(3, "_decode"), (64, "_stream_"), (1031, "_prefill_")],
                         ids=["m3", "m64", "m1031"])
def test_tp2(tokens, kernel):
    _a8w4_check(320, tokens, seed=11, kernel=kernel)


@pytest.mark.basic
def test_tp4_padded():
    prepared = _a8w4_check(160, 257, seed=21, kernel="_prefill_")
    assert (prepared.inter, prepared.inter_padded) == (160, 192)


@pytest.mark.basic
@pytest.mark.parametrize("inter, tokens, kernel", [(256, 5, "_decode"), (256, 300, "_stream_"), (512, 24, "_decode"),
                                                   (512, 1600, "_gateup_prefill_")],
                         ids=["tp8_m5", "tp8_m300", "tp4_m24", "tp4_m1600"])
def test_mimo(inter, tokens, kernel):
    _a8w4_check(inter, tokens, seed=41, kernel=kernel, hidden=MIMO_HIDDEN, experts=MIMO_EXPERTS, topk=MIMO_TOPK,
                shared=False)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# 区域 2｜基本功能 · 底座（@pytest.mark.basic）
# FlyDSL 两阶段 MoE 底座（Aiter gfx9_fp8）每条公开路径一例：固定配置经 ``make_moe_runner`` 执行完整调用（排序、
# 量化、Gate/Up、Down、规约），与独立 Torch 参考比较（``calc_diff <= 0.02``），并断言实际启动的 kernel 路径。
# 维度取 Qwen3.8 的 H2560 与 TP2 / TP4 的 I320 / I192，E/topk 缩小以控制参考耗时。
# 下面的常量、fixture 与辅助函数区域 4 也用。
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

FP8_TP2 = dict(HIDDEN_SIZE=2560, INTER_SIZE=320, TP=1, E=64, TOPK=8)
FP8_TP4 = FP8_TP2 | dict(INTER_SIZE=192)  # 8x1_compact 不支持 I320。
# 每个 expert 257 行：共 64 个 M256 full 任务和 64 个 M64 tail；在 80 CU 的 MI308X 上 full 不足一轮但超过 3/5，
# 不拆，compact 同时执行 M256 full 与 M64 tail 任务。
FP8_PREFILL_TOKENS = 2056


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


def _fp8_prepare(model, tokens, precision, **options):
    bf16 = precision == "bf16"
    call, _ = prepare_moe(model, tokens, dtype="bf16" if bf16 else "fp8",
                          quant="model" if bf16 else precision, seed=8, **options)
    call["hidden_states"].mul_(50)  # 进入激活函数非线性区，并让中间激活量化有意义。
    return call


def _fp8_check(call, config):
    runner = make_moe_runner(config)
    reference = torch_reference(call)
    assert torch.isfinite(reference).all()
    call["output"].fill_(float("nan"))
    check = check_output(runner(**call), call["output"], reference)
    assert check["status"] == "PASS", f"{config}: {check}"
    return runner


def _fp8_prefill(inter, down_path, padding):
    # Aiter 调优表在 Qwen3.8 I320/I192 prefill 上选用的 Gate/Up tile：M64、N128、K256；
    # I640 用该表在这个 I 上选的 N256/K128。
    gate = dict(tile_n_gate=256, tile_k_gate=128) if inter == 640 else dict(tile_n_gate=128, tile_k_gate=256)
    return dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=64,
                tile_n_down=256 if down_path == "1x4_64x256" else 128,
                down_path=down_path, padding=padding) | gate


@pytest.mark.basic
def test_fp8_batch1(kernels):
    _fp8_check(_fp8_prepare(FP8_TP2, 1, "ptpc"),
               dict(_impl="fly_decode", decode_alg="batch1", tile_n_gate=32, tile_n_down=64))
    assert kernels == [("gateup", "batch1", "default"), ("down", "batch1", "default")]


@pytest.mark.basic
def test_fp8_splitk(kernels):
    _fp8_check(_fp8_prepare(FP8_TP2, 16, "ptpc"), dict(_impl="fly_decode", tile_m_gate=16, tile_m_down=16,
                                                       tile_n_gate=64, tile_n_down=64))
    assert kernels == [("gateup", "splitk", "default"), ("down", "splitk", "default")]


@pytest.mark.basic
@pytest.mark.parametrize("model, down_path, padding", [
    (FP8_TP2, "default", None), (FP8_TP2, "1x4_64x256", 128), (FP8_TP4, "8x1_compact", 128),
], ids=["default", "1x4_64x256", "8x1_compact"])
def test_fp8_prefill(kernels, task_counts, model, down_path, padding):
    _fp8_check(_fp8_prepare(model, FP8_PREFILL_TOKENS, "ptpc"), _fp8_prefill(model["INTER_SIZE"], down_path, padding))
    assert kernels == [("gateup", "prefill_1x4", "default"), ("down", "prefill_1x4", down_path)]
    if down_path == "8x1_compact" and get_cu_num() == 80:
        # GPU 生成的任务表：M256 full 与 M64 tail 的行数都非零，两种 kernel 都执行过。
        assert len(task_counts) == 1 and bool((task_counts[0] > 0).all()), task_counts


@pytest.mark.basic
def test_fp8_prefill_bf16(kernels):
    _fp8_check(_fp8_prepare(FP8_TP2, FP8_PREFILL_TOKENS, "bf16"),
               _fp8_prefill(320, "default", None) | dict(tile_k_gate=64))
    assert kernels == [("gateup", "prefill_1x4", "default"), ("down", "prefill_1x4", "default")]


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# 区域 3｜其他覆盖 · A8W4
# 只放区域 1 与 bench（区域 5 的性能测试）覆盖不了的情况，每个测试的 docstring 写明原因（docstring 里的“基本测试”指区域 1）。端到端判据
# 与区域 1 相同（参考的激活精度按实际启动的 kernel 选择）；逐位用例要求 0 误差。部分填充的尾块、多块 expert 与
# I=160→192 补零已由区域 1 覆盖，E=513 的各个 M 档由 bench 覆盖，这里不重复。
# M≤32 的 decode 路径：Qwen 的核内量化（M=3）与 MiMo 的按 expert 去重（M=5 / 24）由区域 1、各分派由 bench（M=1、2、
# 4…32，MiMo 另有 48）覆盖，这里只测分派交界、topk 与路由输入的变体、CUDA graph、行量化与去重计划逐位、w2≥4 GiB
# 回退（用例名带 ``decode``）。M>32 且每个 expert 平均不超过 22 行（H≥4096 为 48，H2560 排序块 64 自带量化为 56）的
# 流式路径：排序块 32 由区域 1、16 / 32 / 64 由 bench 覆盖，这里只测与 decode、prefill 的交界、prep 与尾部量化逐位
# （用例名带 ``stream``）、排序块 64 换 prefill Down 及其切分，以及 grid 收缩与 graph。
# prefill 路径：默认 tile 由区域 1 与 bench 覆盖，这里只测半块变体的交界、tile 参数与 graph。非目标形状的回退见
# ``test_shape_fallbacks``；同 token 重复 expert 与原地输出见 3.7。
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

# decode 三种分派（x、mid 是否先用独立 kernel 预量化）各自的 Gate/Up 与 Down kernel。
_DECODE_PATHS = {
    "nn": ["a8w4_gateup_decode", "a8w4_down_decode"],
    "np": ["a8w4_gateup_decode", "a8w4_down_decode_pq"],
    "pp": ["a8w4_gateup_decode_pq", "a8w4_down_decode_pq"],
}


def _prefill_path(tile_k, block_m=64, block_n=128):
    """prefill 路径的 Gate/Up 与 Down kernel（名字带排序块、Gate/Up 的 N tile 与 K tile）。"""
    return [f"a8w4_gateup_prefill_bm{block_m}_bn{block_n}_tk{tile_k}", f"a8w4_down_prefill_bm{block_m}"]


def _stream_path(block_m, fused):
    """流式路径的 Gate/Up 与 Down kernel（名字带排序块）：``fused`` 时 Gate/Up 在尾部量化中间激活。"""
    gateup = f"a8w4_gateup_quant_stream_bm{block_m}" if fused else f"a8w4_gateup_stream_bm{block_m}"
    return [gateup, f"a8w4_down_stream_bm{block_m}"]


def _assert_path(expected):
    """决定分档与变体的关键 kernel（``expected``，按名字精确匹配）按序出现在 ``last_path`` 中，中间可以夹别的 kernel：
    排序、行量化与 route_sum / sorted_sum 不逐个断言，它们的正确性由每个用例对参考的端到端校验覆盖。
    """
    rest = iter(moe_a8w4.last_path)
    assert all(name in rest for name in expected), (expected, moe_a8w4.last_path)


def _a8w4_run(w, prepared, x, topk_weight, topk_ids, **options):
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
    return w, _a8w4_prepare(w)


# ── 3.1 权重布局与 delta ───────────────────────────────────────────────────────────────────────────────

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
    out, ref_exact = _a8w4_run(exact, _a8w4_prepare(exact), probe_x, topk_weight, topk_ids)
    mode = oracle.path_activation(moe_a8w4.last_path)
    ref_flushed = oracle.reference(probe_x, _pack(flushed), codes, w["w2"], w["w2_scale"], topk_weight,
                                   topk_ids, activation=mode)
    oracle.assert_close(out, ref_flushed)
    assert not oracle.check(out, ref_exact)["ok"]


def _decode_layout(q, d, s, N, K):
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
    got, delta = _decode_layout(p.w1, p.w1_delta, p.w1_scale, 2 * 192, H)
    assert torch.equal(got, torch.cat([v1[:, :I], z1, v1[:, I:], z1], dim=1))
    assert torch.equal(delta[:, :I], d1[:, :I]) and torch.equal(delta[:, 192:192 + I], d1[:, I:])
    assert torch.equal(p.w1_scale.view(E, -1)[:, :I], c1[:, :I])

    v2, d2, c2 = expected(w2, s2, H, I)
    got, delta = _decode_layout(p.w2, p.w2_delta, p.w2_scale, H, 192)
    assert torch.equal(got, torch.nn.functional.pad(v2, (0, 192 - I)))
    assert torch.equal(delta[:, :, :I // 32], d2)
    assert torch.equal(p.w2_scale.view(E, -1), c2)


# ── 3.2 行量化逐位 ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype, cols", [("bf16", 2560), ("f32", 320)])
def test_row_quant_bitexact(dtype, cols):
    """激活的行量化（scale = amax/240，q = RNE(x/scale)）与参考逐位一致，含全零行与 BF16 常见的精确中点；
    kernel 用倒数乘法加两次 FMA 修正求商、scale < 2^-64 的行逐元素除法，另加两行直接乘倒数会改变 FP8 值的数据。

    基本测试只看端到端误差：量化算术的微小差异（例如乘倒数）只让少量 FP8 值翻转，小 M 时判据未必发现，
    却会吃掉大 M 下 4.5% 最大误差的余量；随机数据里乘倒数恰好舍错的元素很少，所以另构造这两行。
    """
    gen = torch.Generator(device="cuda").manual_seed(81)
    x = torch.randn((997, cols), generator=gen) * 3
    x[5].zero_()
    # 第 6 行 scale = 52.5/240 = 7/32（不是 2 的幂），其余元素的商恰为 FP8 舍入中点（尾数 23/16 … 31/16）：
    # 乘倒数的结果多出一个 f32 ulp，越过中点后舍入方向改变。第 7 行再乘 2^-70，scale < 2^-64 走逐元素除法。
    tie = torch.arange(23, 32, 2).float().view(-1, 1) / 16 * torch.exp2(torch.arange(-6, 7).float())
    tie = torch.cat([tie.flatten(), -tie.flatten()]) * (7 / 32)
    x[6] = tie.repeat(cols // tie.numel() + 1)[:cols]
    x[6, 0] = 52.5
    x[7] = x[6] * 2.0 ** -70
    x = x.to(torch.bfloat16) if dtype == "bf16" else x
    q, scale = moe_a8w4._quantize(x, x.shape[0], cols, dtype)
    ref = oracle.quant_rows_fp8(x)
    amax = x.float().abs().amax(-1, keepdim=True)
    ref_scale = torch.where(amax == 0, torch.ones_like(amax), amax / torch.full_like(amax, 240.0))
    assert torch.equal(scale.view(-1, 1), ref_scale)
    assert torch.equal(q.float() * scale.view(-1, 1), ref)
    assert int((q.view(torch.uint8) == 0x80).sum()) == 0
    exact = (x.float() / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8)
    naive = (x.float() * (torch.ones_like(ref_scale) / ref_scale)).to(torch.float8_e4m3fnuz).view(torch.uint8)
    assert bool((exact[6] != naive[6]).any()) and bool((exact[7] != naive[7]).any())
    assert float(scale[7]) < 2.0 ** -64 <= float(scale[6])


@pytest.mark.parametrize("dtype, cols, fill", [("bf16", 2560, 11), ("f32", 512, 40)], ids=["bf16_fill11", "f32_fill40"])
def test_row_quant_fill(dtype, cols, fill):
    """prefill 的输入行量化顺带把 loc_ids [rows, fill] 置 -1（之后 Down 只写有效路由的排序行）：q 与 scale 和不带 fill
    时逐位相同，fill 恰好写满 rows 行、不越界；fill=40 多于 f32·512 列时每行的 32 个 lane（每个 lane 写两轮）。

    端到端只在同 token 重复 expert 时看得到这些 -1（test_prefill_duplicate_ids_keep_last_route），看不出越界写。
    """
    gen = torch.Generator(device="cuda").manual_seed(91)
    rows = 997
    x = torch.randn((rows, cols), generator=gen)
    x = x.to(torch.bfloat16) if dtype == "bf16" else x
    q0, s0 = moe_a8w4._quantize(x, rows, cols, dtype)
    aux = torch.full((rows + 1, fill), 7, dtype=torch.int32)
    q, s = moe_a8w4._quantize(x, rows, cols, dtype, fill=aux[:rows])
    assert torch.equal(q.view(torch.uint8), q0.view(torch.uint8)) and torch.equal(s, s0)
    assert bool((aux[:rows] == -1).all()) and bool((aux[rows] == 7).all())


@pytest.mark.parametrize("dtype, cols", [("f32", 192), ("f32", 320), ("f32", 640),
                                         ("bf16", 1024), ("bf16", 2560), ("bf16", 4096)])
def test_decode_row_quant_bitexact(dtype, cols):
    """decode 路径的行量化 kernel 与 a8w4_quant、参考逐位一致（q 字节与 scale 位），且不写 rows 以外的行。

    特殊行：全零、amax/240 下溢成 0（scale 取 1，只有 f32 能表示）、次正规 scale、-0 与舍入成 -0 的负数（存 +0）；
    f32 192 / 320 / 640 列每行 48 / 80 / 160 个 16 B 片（末轮不满 64 片），bf16 1024 / 2560 / 4096 列；
    37 行（最后一个 WG 只有 1 行，其后 3 行不得被写）。端到端判据看不出少量 fp8 翻转，基本测试与 bench
    只有随机数据（f32 192 / 256 / 320 / 512 列，bf16 2560 / 4096 列）。
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


def test_decode_plan_quant_bitexact():
    """按 expert 去重的 decode 的计划 kernel（a8w4_row_quant_plan：每行 4 个 wave 分片量化 x、amax 经 LDS 合并，同一次 launch
    写去重计划并把 out 的行清零）：xq / xs 与 a8w4_quant、参考逐位一致，out 的前 rows 行清零，rows 以外的行不写；计划的条数与
    组员与按定义算的相同（每个 expert 的路由按路由号每 3 条一组，组首记条数与组内各路由，其余条数为 0；条数以外的字不读，不查）。

    特殊行同 test_decode_row_quant_bitexact（全零、次正规 scale、-0 与舍入成 -0 的负数）；H=4096（去重只用于这个 H），
    37 行 × topk 8 = 296 条路由落在 40 个 expert 上，按 M≤64 规划（8 段 64 条路由的 ballot），组员跨段、尾组不满都会出现。
    端到端判据看不出少量 fp8 翻转；计划出错多数会让端到端失败，但跨段与尾组不满的组合随机路由未必碰到。
    """
    rows, guard, cols, group = 37, 3, MIMO_HIDDEN, 3
    gen = torch.Generator(device="cuda").manual_seed(87)
    x = torch.randn((rows, cols), generator=gen) * 3
    x[1].zero_()
    x[3] *= 1e-38
    x[4].zero_()
    x[4, 0] = 200.0
    x[4, 1:9] = -1e-6
    x[4, 9] = -0.0
    x = x.to(torch.bfloat16)
    _, _, topk_ids = oracle.make_inputs(rows, 512, 40, MIMO_TOPK, seed=88, shared=False)
    q = torch.full((rows + guard, cols), 0x5A, dtype=torch.uint8)
    scale = torch.full((rows + guard,), -1.0)
    plan = torch.full((rows * MIMO_TOPK, 4), -1, dtype=torch.int32)
    out = torch.full((rows + guard, cols), float("nan"), dtype=torch.bfloat16)
    moe_a8w4._launch("a8w4_row_quant_plan_decode", moe_a8w4._row_quant_plan(cols, MIMO_TOPK, group, 64), x,
                     q.view(torch.float8_e4m3fnuz), scale, topk_ids, plan, out, rows)
    assert bool((q[rows:] == 0x5A).all()) and bool((scale[rows:] == -1.0).all())
    assert bool((out[:rows] == 0).all()) and bool(out[rows:].isnan().all())
    q0, scale0 = moe_a8w4._quantize(x, rows, cols, "bf16")
    assert torch.equal(q[:rows], q0.view(torch.uint8))
    assert torch.equal(scale[:rows].view(torch.int32), scale0.view(torch.int32))
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    ref_scale = amax / torch.full_like(amax, 240.0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)
    assert float(ref_scale[3]) < 2.0 ** -126
    assert torch.equal(q[:rows], (xf / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8))
    ids = topk_ids.flatten().tolist()
    full = partial = 0
    for r, e in enumerate(ids):
        same = [s for s, f in enumerate(ids) if f == e]
        rank = same.index(r)
        count = min(group, len(same) - rank) if rank % group == 0 else 0
        entry = plan[r].tolist()
        assert entry[:2] == [count, r] and entry[1:1 + count] == same[rank:rank + count], (r, entry)
        full += count == group
        partial += 0 < count < group
    assert full and partial


@pytest.mark.parametrize("cols, qw", [(512, 4), (2560, 4), (2560, 1), (4096, 4)],
                         ids=["h512_qw4", "h2560_qw4", "h2560_qw1", "h4096_qw4"])
def test_stream_row_quant_bitexact(cols, qw):
    """流式路径的 prep kernel（a8w4_stream_prep：一次 launch 排序、量化输入并清零 out）：量化与 a8w4_quant、参考逐位一致
    （q 字节与 scale 位），out 的前 rows 行清零，rows 以外的行一律不写；排序合 aiter moe_sorting 的格式（每条路由的 loc_ids
    指向自己的排序行，expert 升序、各补到整块，填充行 token=M、权重 0，num_valid=[补齐后的行数, M]）。

    量化的两种分工都测：QW=4 每 WG 一行、4 个 wave 分 512 列块（H512 只有 wave 0 有数据，H2560 的 5 块不均分），amax 经 LDS
    交换，倒数乘法 + 1∓2^-20 夹逼、靠近舍入边界或 scale 非正规时回退 IEEE 除法；QW=1 每 wave 一行、逐元素 IEEE 除法（分派对
    H2560 且 M>256 用）。特殊行：全零、次正规 scale、-0 与舍入成 -0 的负数（存 +0）、商恰为舍入中点（scale 2^-2）、scale 7/32
    下直接乘倒数会舍错的中点；37 行（QW=1 的最后一个 WG 只有 1 行）。E=513、topk=11：2 个排序 WG，每线程管 3 个 expert。
    端到端判据看不出少量 fp8 翻转；基本测试与 bench 只有随机数据，也不查排序格式。
    """
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.a8w4_stream import stream_prep_sizes

    rows, guard, experts, block_m = 37, 3, 513, 16
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
    # 第 5 行 scale = 52.5/240 = 7/32，其余元素的商恰为舍入中点：直接乘倒数会舍错（同 test_row_quant_bitexact）。
    tie7 = torch.arange(23, 32, 2).float().view(-1, 1) / 16 * torch.exp2(torch.arange(-6, 7).float())
    tie7 = torch.cat([tie7.flatten(), -tie7.flatten()]) * (7 / 32)
    x[5] = tie7.repeat(cols // tie7.numel() + 1)[:cols]
    x[5, 0] = 52.5
    x = x.to(torch.bfloat16)
    _, topk_weight, topk_ids = oracle.make_inputs(rows, 512, experts, TOPK, seed=86)
    n_sort, capacity, blocks = stream_prep_sizes(rows, TOPK, experts, block_m)
    sorted_ids = torch.full((capacity,), -1, dtype=torch.int32)
    sorted_weights = torch.full((capacity,), -1.0)
    sorted_experts = torch.full((blocks,), -1, dtype=torch.int32)
    num_valid = torch.full((2,), -1, dtype=torch.int32)
    loc_ids = torch.full((rows, TOPK), -2, dtype=torch.int32)
    q = torch.full((rows + guard, cols), 0x5A, dtype=torch.uint8)
    scale = torch.full((rows + guard,), -1.0)
    out = torch.full((rows + guard, cols), float("nan"), dtype=torch.bfloat16)
    moe_a8w4._launch("a8w4_stream_prep", moe_a8w4._stream_prep_kernel(cols, TOPK, experts, block_m, True, qw),
                     topk_ids, topk_weight, x, sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids,
                     q.view(torch.float8_e4m3fnuz), scale, out, rows, n_sort, capacity)
    assert bool((q[rows:] == 0x5A).all()) and bool((scale[rows:] == -1.0).all())
    assert bool((out[:rows] == 0).all()) and bool(out[rows:].isnan().all())
    q0, scale0 = moe_a8w4._quantize(x, rows, cols, "bf16")
    assert torch.equal(q[:rows], q0.view(torch.uint8))
    assert torch.equal(scale[:rows].view(torch.int32), scale0.view(torch.int32))
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    ref_scale = amax / torch.full_like(amax, 240.0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)
    assert float(ref_scale[2]) < 2.0 ** -126 and float(ref_scale[4]) == 0.25 and float(ref_scale[5]) == 7 / 32
    assert torch.equal(scale[:rows].view(-1, 1), ref_scale)
    exact = (xf / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8)
    naive = (xf * (torch.ones_like(ref_scale) / ref_scale)).to(torch.float8_e4m3fnuz).view(torch.uint8)
    assert torch.equal(q[:rows], exact) and bool((exact[5] != naive[5]).any())
    counts = torch.bincount(topk_ids.flatten().long(), minlength=experts)
    padded = (counts + block_m - 1) // block_m * block_m
    total = int(padded.sum())
    assert num_valid.tolist() == [total, rows]
    loc = loc_ids.flatten().long()
    route = torch.arange(rows * TOPK, dtype=torch.int32)
    assert torch.equal(sorted_ids[loc], (route // TOPK) | ((route % TOPK) << 24))
    assert torch.equal(sorted_weights[loc], topk_weight.flatten())
    assert torch.equal(sorted_experts[loc // block_m], topk_ids.flatten())
    assert torch.equal(sorted_experts[:total // block_m],
                       torch.repeat_interleave(torch.arange(experts, dtype=torch.int32), padded // block_m))
    filler = torch.ones(total, dtype=torch.bool)
    filler[loc] = False
    assert bool((sorted_ids[:total][filler] == rows).all()) and bool((sorted_weights[:total][filler] == 0).all())


@pytest.mark.parametrize("inter, block_m", [(320, 32), (160, 64)], ids=["tp2_bm32", "tp4_bm64"])
def test_stream_fused_quant_bitexact(inter, block_m):
    """流式 Gate/Up 尾部量化的中间激活（倒数乘法 + 1∓2^-20 夹逼判定；有元素靠近舍入边界、或 scale 不是有限正规数时
    整个 wave 回退 IEEE 除法）与同一排序下写 f32 再用 a8w4_quant 量化的结果逐位一致（q 字节与 scale 位）。
    TP2 排序块 32（分派实际用的：每 wave 5 个 gate 块、不共享激活）；TP4 排序块 64（3 个 gate 块、LDS 共享激活，
    子块数可到 4；分派对 TP4 用排序块 32，同一份代码，排序块 64 只在调大 STREAM_MAX_ROWS 时用）。

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


# ── 3.3 decode 分派与输入变体 ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("inter, tokens, path", [
    (320, 4, "np"), (320, 11, "np"), (320, 12, "pp"), (320, 32, "pp"), (320, 33, "stream"),
    (160, 19, "np"), (160, 20, "pp"),
], ids=["tp2_m4", "tp2_m11", "tp2_m12", "tp2_m32", "tp2_m33", "tp4_m19", "tp4_m20"])
def test_decode_dispatch_boundaries(inter, tokens, path):
    """decode 分派各交界的端到端结果与关键 kernel：M=4（Down 改用预量化槽对 kernel；M=3 一侧是基本测试）、
    TP2 M=11/12 与 TP4 M=19/20（Gate/Up 的 WG 数 M·topk·I/16 跨过 2560，改读预量化的 x）、M=32/33（改走排序的
    流式路径；E=33 时每个 expert 平均 11 行，排序块取 16）。

    基本测试的 decode 只有 M=3 与 MiMo 的 M=5 / 24（都不在这些交界上）；bench 的 M 只取 2 的幂与 48（没有 11/12、
    19/20、33），也不断言 kernel 序列：M=4、32 两例与 bench 同 M 同路径，留作交界的一侧。
    """
    w, prepared = _decode_weights(inter)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 33, TOPK, seed=90 + tokens)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    if path == "stream":
        _assert_path(_stream_path(16, False))
    else:
        _assert_path(_DECODE_PATHS[path])
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("topk, tokens, path", [(8, 16, "pp"), (17, 2, "nn")], ids=["topk8_m16", "topk17_m2"])
def test_decode_topk_variants(topk, tokens, path):
    """topk≠11：topk=8 是偶数，槽对 Down 3 个 wave 两轮、第二轮两个 wave 的槽对整对无效，且 M=16 时 Gate/Up 恰为
    2560 个 WG（按 >= 走预量化）；topk=17 大于 16，核内量化的 Down 改为 4 个 wave 轮流取槽，末轮 3 个槽无效。

    基本测试与 bench 的 Qwen 是 topk=11：槽对只有末对缺高半；MiMo 的 topk=8 走按 expert 去重的 decode（M=1 用单槽 Down），
    不经过槽对 Down；两者都不超过 16，核内量化的 Down 每个槽一个 wave。
    """
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 33, topk, seed=94 + topk)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(_DECODE_PATHS[path])
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
    out, ref = _a8w4_run(w, prepared, x, weights, ids)
    _assert_path(_DECODE_PATHS["np"])
    oracle.assert_close(out, ref)


def test_decode_down_fallback_w2_4gib():
    """w2 总字节数 >= 4 GiB 时，Down 不用预量化槽对 kernel（它用 32 位字节偏移寻址全部 expert），改走单槽核内量化 Down。

    真实 4 GiB 权重太大：用 stride 0 的视图让 w2 报告 2^32 字节（数据指针不变，kernel 只按指针与 expert 偏移读），
    验证 M=16 时的分派与单槽 Down 的结果；基本测试、bench 与其余用例的 w2 都远小于 4 GiB。
    """
    w, prepared = _decode_weights(320)
    big = dataclasses.replace(prepared, w2=prepared.w2.view(-1)[:1].expand(1 << 32))
    x, topk_weight, topk_ids = oracle.make_inputs(16, HIDDEN, 33, TOPK, seed=99)
    out, ref = _a8w4_run(w, big, x, topk_weight, topk_ids)
    _assert_path(["a8w4_gateup_decode_pq", "a8w4_down_decode"])
    oracle.assert_close(out, ref)


# 按 expert 去重的 decode（H=4096、I_pad%128==0、M≥2）的两种分派：M·I_pad 小时两个 kernel（逐路由 Gate/Up 顺带写计划并清零
# out，Down 在 WG 内量化 mid），其余四个 kernel（x 行量化 + 计划 + 清零 → 去重 Gate/Up → mid 行量化 → 去重 Down）。
_DEDUP_PATHS = {
    "dd2": ["a8w4_gateup_decode_plan", "a8w4_down_decode_dd"],
    "dd4": ["a8w4_row_quant_plan_decode", "a8w4_gateup_decode_dd", "a8w4_row_quant_decode", "a8w4_down_decode_dd"],
}


@functools.cache
def _mimo_small_weights(inter):
    """去重 decode 用例共用的 MiMo 形状权重（H=4096、topk=8 用），E 缩到 12：每个 expert 平均分到 2M/3 条路由。"""
    w = oracle.make_weights(12, MIMO_HIDDEN, inter, seed=200 + inter)
    return w, _a8w4_prepare(w)


def _spy_dedup(monkeypatch):
    """记录去重路径各 kernel 的构建参数（每组路由数、去重 Gate/Up 的 wave 数、Down 是否在 WG 内量化）。"""
    seen = {}

    def wrap(name, fields):
        original = getattr(moe_a8w4, name)

        def spy(*args):
            seen.update({key: args[i] for key, i in fields.items()})
            return original(*args)

        monkeypatch.setattr(moe_a8w4, name, spy)

    wrap("_gateup_plan_kernel", dict(group=4))
    wrap("_row_quant_plan", dict(group=2))
    wrap("_gateup_dd_kernel", dict(group=3, waves=4))
    wrap("_down_dd_kernel", dict(down_group=3, quant=5))
    return seen


@pytest.mark.parametrize("inter, tokens, path, group, waves", [
    (256, 2, "dd2", 2, None), (256, 11, "dd2", 2, None), (256, 12, "dd4", 2, 8), (256, 16, "dd4", 2, 8),
    (256, 17, "dd4", 3, 4), (512, 5, "dd2", 2, None), (512, 6, "dd4", 2, 8), (384, 2, "dd4", 2, 8),
], ids=["i256_m2", "i256_m11", "i256_m12", "i256_m16", "i256_m17", "i512_m5", "i512_m6", "i384_m2"])
def test_decode_dedup_dispatch(monkeypatch, inter, tokens, path, group, waves):
    """按 expert 去重的 decode 各交界的端到端结果、kernel 序列与构建参数：M·I_pad 跨过 3072 时两个 kernel 改为四个
    （I256 的 M=11/12、I512 的 M=5/6）；M=16/17 每组路由数 2→3、去重 Gate/Up 每 WG 8→4 个 wave；I384（I_pad%256≠0，
    Down 不能在 WG 内量化）M=2 也走四个 kernel。M=1 一侧（原 decode）由 bench 的 mimo_*_m1 覆盖。

    E 缩到 12（topk=8）：每个 expert 平均分到 2M/3 条路由，组满 2 / 3 条、尾组不满、同一 token 的各条路由分在不同组里原子
    累加都会出现；MiMo 的 E256 在小 M 时几乎每组只有 1 条路由。bench 的 MiMo 只有 2 的幂与 48，也不断言 kernel 序列；
    基本测试的 MiMo decode 只有 M=5 / 24。
    """
    seen = _spy_dedup(monkeypatch)
    w, prepared = _mimo_small_weights(inter)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, MIMO_HIDDEN, 12, MIMO_TOPK, seed=210 + tokens, shared=False)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(_DEDUP_PATHS[path])
    oracle.assert_close(out, ref)
    assert (seen["group"], seen.get("waves"), seen["down_group"], seen["quant"]) == (group, waves, group, path == "dd2")


@functools.cache
def _mimo_e256_weights():
    """MiMo TP8 满规模权重（E=256、H=4096、I=256），只给去重 decode 与流式的交界用例。"""
    w = oracle.make_weights(MIMO_EXPERTS, MIMO_HIDDEN, 256, seed=230)
    return w, _a8w4_prepare(w)


@pytest.mark.parametrize("tokens", [48, 49], ids=["m48", "m49"])
def test_decode_dedup_stream_boundary(tokens):
    """M>32 时按 expert 去重的 decode 只在每个 expert 平均不超过 1.5 条路由（2·M·topk ≤ 3E）时继续用：MiMo TP8（E256、
    topk8、I256）M=48 仍走去重的四个 kernel，M=49 改走流式（自写 prep 排序、排序块 16，Down 原子累加、没有 route_sum）。

    交界由 E 决定，只能用满规模的 E=256；bench 只有 M=48 与 64，也不断言 kernel 序列。
    """
    w, prepared = _mimo_e256_weights()
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, MIMO_HIDDEN, MIMO_EXPERTS, MIMO_TOPK, seed=230 + tokens,
                                                  shared=False)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    if tokens == 48:
        _assert_path(_DEDUP_PATHS["dd4"])
    else:
        _assert_path(["a8w4_stream_prep"] + _stream_path(16, False))
        assert "a8w4_route_sum" not in moe_a8w4.last_path
    oracle.assert_close(out, ref)


# ── 3.4 流式与 prefill：交界、grid 收缩、tile 参数 ────────────────────────────────────────────────────

def test_grid_shrink_boundary_e513():
    """E=513 满规模下流式 Gate/Up、Down 的 grid 收缩（M·topk <= E 时 task 数取 M·topk）交界两侧：M=46（506）、M=47（517）。

    基本测试（E=33 / 256）不在交界上；bench 在 33–63 只有 M=48（528 条路由 > E，不收缩，与 M=47 同侧）。这两个 M 的路由数
    少于 4E/3，排序块取 32，与 bench 的 M=48 同一路径。
    """
    w = oracle.make_weights(513, HIDDEN, 320, seed=31)
    prepared = _a8w4_prepare(w)
    for tokens in (46, 47):
        x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 513, TOPK, seed=32 + tokens)
        oracle.assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids))
        _assert_path(_stream_path(32, True))


@pytest.mark.parametrize("tokens, path", [(66, _stream_path(32, True)), (67, _prefill_path(128))],
                         ids=["m66", "m67"])
def test_stream_dispatch_boundary(tokens, path):
    """流式与 prefill 的交界（M·topk ≤ 22·E 走流式）：E=33、topk=11 时 M=66（每个 expert 平均 22 行）仍走流式
    （排序块 32、Gate/Up 尾部量化），M=67 改走 prefill，且默认 TILE_K=128。

    基本测试（流式 M=64 / 300，prefill M=257 / 1031 / 1600）与 bench 的各档都不在交界上。
    """
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, 33, TOPK, seed=tokens)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(path)
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("hidden, inter, tokens, path", [
    (2560, 160, 140, _stream_path(64, True)),
    (2560, 160, 141, _prefill_path(128)),
    (4096, 256, 62, _stream_path(32, False)),
    (4096, 256, 120, _stream_path(64, False)),
    (4096, 256, 121, _prefill_path(128)),
], ids=["h2560_i192_m140", "h2560_i192_m141", "h4096_i256_m62_bm32", "h4096_i256_m120", "h4096_i256_m121"])
def test_stream_rows_limits(hidden, inter, tokens, path):
    """流式上限按形状取：H≥4096 每个 expert 平均 ≤48 行（``STREAM_MAX_ROWS_WIDE``），H2560 且排序块 64 的 Gate/Up 自带
    量化（I_pad=192）≤56 行（``STREAM_MAX_ROWS_FUSED_QUANT``），交界两侧各一例（E=5、topk=2：M=140 / 141 为 56 / 56.4 行，
    M=120 / 121 为 48 / 48.4 行）。H4096·I256 的流式 Down 原子累加，排序块 32（M=62）时从 LDS 读激活（K%256==0）。

    基本测试与 bench 不在这两个交界上（22 行的交界见 test_stream_dispatch_boundary）；bench 的 MiMo TP8 M512 / 1024
    走这两种原子 Down，但不校验数值。E=5、topk=2 控制耗时。
    """
    w, prepared = _small_weights(hidden, inter)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, hidden, 5, 2, seed=300 + tokens, shared=False)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(path)
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("min_wgs, nsplit", [(5, 1), (8, 2), (1000, 4)], ids=["split1", "split2", "split4"])
def test_stream_prefill_down_split(monkeypatch, min_wgs, nsplit):
    """排序块 64、Down 不原子累加（I≥512）且 H≥4096 时，流式 Gate/Up 之后改用 prefill Down（顺带写 loc_ids）+ route_sum；
    prefill Down 按排序块数把 hidden 切给 1 / 2 / 4 个 WG（E=5、M=100 共 5 个块，``DOWN_MIN_WGS`` 取 5 / 8 / 默认 1000）。

    基本测试的 MiMo TP4 M1600 走 prefill，其 prefill 只用到切分 4（I≥256）与 1（I_pad=192）；bench 的 mimo_tp4_m1024
    走这条路径但不校验数值。E=5、topk=2 控制耗时。
    """
    monkeypatch.setattr(moe_a8w4, "DOWN_MIN_WGS", min_wgs)
    assert moe_a8w4._down_split(100, 2, 5, 4096, 512, 64) == nsplit
    w, prepared = _small_weights(4096, 512)
    x, topk_weight, topk_ids = oracle.make_inputs(100, 4096, 5, 2, seed=310, shared=False)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(["a8w4_stream_prep", "a8w4_gateup_stream_bm64", "a8w4_down_prefill_bm64", "a8w4_route_sum"])
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("inter, options", [
    (256, dict(block_m=128, block_n=256)),
], ids=["bm128_bn256"])
def test_tile_variants(inter, options):
    """给了 tile 参数时按参数选 prefill 的 tile：BLOCK_N=256（每个 wave 两个 16 行块）、BLOCK_M=128，
    没给的 tile_k 在 M≤2048 时取 256（M=300）。

    基本测试与 bench 只走默认 tile（BLOCK_M=64、BLOCK_N=128，每个 expert 行多时 TILE_K=128）。
    """
    w = oracle.make_weights(33, HIDDEN, inter, seed=36)
    x, topk_weight, topk_ids = oracle.make_inputs(300, HIDDEN, 33, TOPK, seed=37)
    oracle.assert_close(*_a8w4_run(w, _a8w4_prepare(w), x, topk_weight, topk_ids, **options))
    _assert_path(_prefill_path(256, block_m=128, block_n=256))


@functools.cache
def _half_block_weights():
    """半块用例共用的 E=9、H=512、I=64 权重（topk=2；最后两个 expert 只用来把 M 凑到 2048 / 2049）。"""
    w = oracle.make_weights(9, 512, 64, seed=131)
    return w, _a8w4_prepare(w)


@pytest.mark.parametrize("filler, options, block_m, tile_k", [
    (0, {}, 64, 128),
    (1854, dict(block_m=128), 128, 256),
    (1855, dict(block_m=64), 64, 128),
], ids=["default_m194", "bm128_m2048", "bm64_m2049"])
def test_prefill_half_block_boundary(filler, options, block_m, tile_k):
    """prefill 的 Down 在排序块第 BM/2 行是填充时只算前 BM/2 行、Gate/Up（BM64）按 16 行取整只算有效行（有效行在块首）：路由构造成各 expert
    恰有 32、33、64、65、96、97、1 行，尾块第 BM/2 行恰为填充或有效（BM64 的 32 / 33、96 / 97，BM128 的 64 / 65），
    K=512 时 Gate/Up 每个 128-K 步的 delta 字节 0–3 都会用到。同时核对 tile_k：默认分派到 prefill 用 128，
    给了 tile 参数时 M=2048 用 256、M=2049 用 128。

    随机路由的尾块行数是随机的，基本测试与 bench 不保证碰到恰好的交界，也不给 tile 参数；H=512、I=64、topk=2 控制耗时。
    """
    w, prepared = _half_block_weights()
    counts = torch.tensor([32, 33, 64, 65, 96, 97, 1, filler, filler])
    flat = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    tokens = flat.numel() // 2
    # 按 expert 排好的路由对折成两列：每个 expert 的行数都不超过 tokens，所以同一 token 的两个 expert 不同。
    topk_ids = torch.stack([flat[:tokens], flat[tokens:]], dim=1).to(torch.int32).contiguous()
    x, topk_weight, _ = oracle.make_inputs(tokens, 512, 9, 2, seed=132)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids, **options)
    _assert_path(_prefill_path(tile_k, block_m=block_m))
    oracle.assert_close(out, ref)


# ── 3.5 CUDA graph ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("options, path", [({}, _stream_path(32, True)), (dict(block_m=64), _prefill_path(256))],
                         ids=["stream", "prefill"])
def test_graph_replay(options, path):
    """整个调用可被 CUDA graph 捕获并重放（无 host 同步），重放读到的是新写入的输入：M=64 默认走流式路径
    （排序块 32、Gate/Up 尾部量化），给 tile 参数时走 prefill（TILE_K 256）。

    基本测试与 bench 都是 eager 调用，不检查 graph 捕获。
    """
    w = oracle.make_weights(33, HIDDEN, 320, seed=51)
    prepared = _a8w4_prepare(w)
    x, topk_weight, topk_ids = oracle.make_inputs(64, HIDDEN, 33, TOPK, seed=52)
    out = torch.empty_like(x)
    moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out, **options)  # 在捕获外编译
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out, **options)
    _assert_path(path)
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
    _assert_path(_DECODE_PATHS[path])
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


@pytest.mark.parametrize("tokens, path", [(5, "dd2"), (24, "dd4")], ids=["m5_dd2", "m24_dd4"])
def test_decode_dedup_graph_replay(tokens, path):
    """按 expert 去重的 decode 两种分派都可被 CUDA graph 捕获并重放：计划、out 清零与原子累加都在被捕获的 kernel 里，
    重放时按新写入的路由重新规划（路径内分配的 plan / mid / xq 由 graph 持有）。

    基本测试与 bench 都是 eager 调用；test_decode_graph_replay 只测原 decode 的三种分派。
    """
    w, prepared = _mimo_small_weights(256)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, MIMO_HIDDEN, 12, MIMO_TOPK, seed=260 + tokens, shared=False)
    out = torch.empty_like(x)
    moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out)  # 在捕获外编译
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out)
    _assert_path(_DEDUP_PATHS[path])
    x2, topk_weight2, topk_ids2 = oracle.make_inputs(tokens, MIMO_HIDDEN, 12, MIMO_TOPK, seed=261 + tokens,
                                                     shared=False)
    x.copy_(x2)
    topk_weight.copy_(topk_weight2)
    topk_ids.copy_(topk_ids2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    ref = oracle.reference(x2, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight2, topk_ids2,
                           activation="fp8")
    oracle.assert_close(out, ref)


# ── 3.6 报错与非目标形状回退 ───────────────────────────────────────────────────────────────────────────

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


@functools.cache
def _small_weights(hidden, inter):
    """形状回退用例共用的 E=5 权重（按 H、I 缓存）。"""
    w = oracle.make_weights(5, hidden, inter, seed=hidden + inter)
    return w, _a8w4_prepare(w)


@pytest.mark.parametrize("hidden, inter, tokens, path, shared, max_rows", [
    (4096, 64, 8, _DECODE_PATHS["np"], None, None),
    (4608, 64, 8, _prefill_path(256), None, None),
    (512, 1024, 4, _DECODE_PATHS["np"], None, None),
    (512, 1088, 4, _DECODE_PATHS["nn"], None, None),
    (512, 1024, 48, _stream_path(32, False), True, None),
    (512, 1024, 300, _stream_path(64, False), False, 120),
], ids=["h4096_decode", "h4608_prefill", "i1024_pair_down", "i1088_single_down", "i1024_bm32_shared",
        "i1024_bm64_unshared"])
def test_shape_fallbacks(monkeypatch, hidden, inter, tokens, path, shared, max_rows):
    """非目标形状按 kernel 上限回退，结果仍对：H<=4096 才走 decode（Gate/Up decode 每线程一片 16 B，至多 256 片），
    H=4608 的 M<=32 改走 prefill（每个 expert 行少，TILE_K 256）；槽对 Down 要把每行 I/16 片放进 64·WAVES 个 lane
    （topk=2 时 1 个 wave），I=1088 改用单槽 Down；流式 Down 只在共享激活行放得进 64 KiB LDS 时让 4 个 wave 共享
    （I=1024：排序块 32 放得下，64 放不下，后者若误选共享 tile 会在构建 kernel 时报错；默认分派的流式只到排序块 32，
    这里临时调大 STREAM_MAX_ROWS 让 M=300 走排序块 64）。

    前两类只有目标形状（H=2560 / 4096，I=192–512）：MiMo 的 H=4096 decode 按 expert 去重（I_pad%128==0），h4096_decode
    （I=64）测原 decode kernel 在 H 上限的一侧，其余上限碰不到；E=5、topk=2 控制耗时。
    """
    from pyhip.ops.moe.flydsl.moe_gemm_2stage import a8w4_common, a8w4_stream

    if max_rows is not None:
        monkeypatch.setattr(moe_a8w4, "STREAM_MAX_ROWS", max_rows)
    w, prepared = _small_weights(hidden, inter)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, hidden, 5, 2, seed=tokens)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(path)
    oracle.assert_close(out, ref)
    if shared is not None:
        block_m = moe_a8w4._stream_block_m(tokens, 2, 5)
        assert (a8w4_stream.down_stream_lds(inter, block_m, W=4) <= a8w4_common.LDS_BYTES) == shared


# ── 3.7 路由与输出的变体：同 token 重复 expert、原地输出 ───────────────────────────────────────────────

@pytest.mark.parametrize("case", ["dedup_m5", "dedup_m24", "stream_atomic", "stream_route_sum"])
def test_duplicate_ids_keep_every_route(case):
    """同一 token 的几个槽选同一 expert：按 expert 去重的 decode（两个 / 四个 kernel，同一 token 的几条路由可能落进同一组）
    与流式路径（自写 prep 排序给每条路由一行；I<512 的 Down 原子累加、I≥512 的 route_sum）都把每个槽各算一次再相加，
    与逐槽相加的参考一致。

    输入生成的路由互不重复，基本测试与 bench 碰不到；原 decode 的重复见 test_decode_duplicate_and_int64_ids，prefill 的
    aiter 排序只保留最后一个槽（入口文档「未覆盖」）。
    """
    if case.startswith("dedup"):
        (w, prepared), hidden, experts, topk, shared = _mimo_small_weights(256), MIMO_HIDDEN, 12, MIMO_TOPK, False
        tokens, path = (5, _DEDUP_PATHS["dd2"]) if case == "dedup_m5" else (24, _DEDUP_PATHS["dd4"])
    elif case == "stream_atomic":
        (w, prepared), hidden, experts, topk, shared = _decode_weights(320), HIDDEN, 33, TOPK, True
        tokens, path = 64, ["a8w4_stream_prep"] + _stream_path(32, True)
    else:
        (w, prepared), hidden, experts, topk, shared = _small_weights(512, 1024), 512, 5, 2, True
        tokens, path = 48, ["a8w4_stream_prep"] + _stream_path(32, False) + ["a8w4_route_sum"]
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, hidden, experts, topk, seed=240 + tokens, shared=shared)
    topk_ids[::2, 1] = topk_ids[::2, 0]
    if topk > 2:
        topk_ids[::3, topk - 2] = topk_ids[::3, 0]
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(path)
    oracle.assert_close(out, ref)


def test_prefill_duplicate_ids_keep_last_route():
    """prefill 遇到同一 token 的两个槽选同一 expert：aiter 排序只保留后一个槽，前一个槽的 loc 保持输入行量化写的 -1、
    被 sorted_sum 跳过（不读未初始化的行、不重复相加），结果等于把前一个槽的路由权重置 0 的参考。

    输入生成的路由互不重复，基本测试与 bench 碰不到；decode、流式（逐槽相加）见 test_duplicate_ids_keep_every_route。
    """
    w, prepared = _small_weights(512, 64)
    x, topk_weight, topk_ids = oracle.make_inputs(200, 512, 5, 2, seed=320, shared=False)
    topk_ids[::2, 1] = topk_ids[::2, 0]
    out = torch.full_like(x, float("nan"))
    moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out)
    _assert_path(_prefill_path(128))
    kept = topk_weight.clone()
    kept[::2, 0] = 0
    ref = oracle.reference(x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], kept, topk_ids, activation="fp8")
    oracle.assert_close(out, ref)


@pytest.mark.parametrize("case", ["dedup", "stream"])
def test_inplace_output(case):
    """output 就是 hidden_states（原地）：按 expert 去重的 decode 在 M·I 小时本走两个 kernel（逐路由 Gate/Up 一边读 x、
    一边清零 out，会读到已清零的 x），这时改走四个 kernel（计划 kernel 每行先读完 x 再清零）；流式的 prep 也在同一个
    WG 里先读完 x 的行再清零 out。结果与按原输入算的参考一致。

    前两类的 output 都另行分配。
    """
    if case == "dedup":
        (w, prepared), hidden, experts, topk, shared = _mimo_small_weights(256), MIMO_HIDDEN, 12, MIMO_TOPK, False
        tokens, path = 5, _DEDUP_PATHS["dd4"]
    else:
        (w, prepared), hidden, experts, topk, shared = _decode_weights(320), HIDDEN, 33, TOPK, True
        tokens, path = 64, ["a8w4_stream_prep"] + _stream_path(32, True)
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, hidden, experts, topk, seed=250 + tokens, shared=shared)
    ref = oracle.reference(x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight, topk_ids, activation="fp8")
    assert moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=x) is x
    _assert_path(path)
    assert oracle.path_activation(moe_a8w4.last_path) == "fp8"
    oracle.assert_close(x, ref)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# 区域 4｜其他覆盖 · 底座
# 只放区域 2 与 bench（区域 5）覆盖不了的情况，每个测试的 docstring 写明原因（docstring 里的“基本用例”指区域 2）：精度 /
# 激活变体、kernel 分支与 inter_dim 交界、尾块与 padding、graph replay、编译期报错、逐位格式，以及 tuned_moe 的
# 候选层门槛与固定配置报错。任务 bench 只计时 A8W4 新算子（对照是 Aiter 自身的 fused_moe），不经过这里的
# BF16 / FP8 底座路径，也不检查数值细节。容差与区域 2 相同（calc_diff <= 0.02，逐位用例为 0 误差）。
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

FP8_SMALL = dict(HIDDEN_SIZE=1024, INTER_SIZE=128, TP=1, E=8, TOPK=4)
FP8_SPLITK = dict(_impl="fly_decode", tile_m_gate=16, tile_m_down=16, tile_n_gate=64, tile_n_down=64)


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


def _round_bf16_reference(value):
    """定点测试遵循当前舍入模式（默认 RTE）；不改变完整 MoE 的独立参考和容差。"""
    if not (common._SIMPLIFIED_BF16_RTA or common._SIMPLIFIED_BF16_RTE):
        return value.bfloat16()
    bits = value.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bias = 0x8000 if common._SIMPLIFIED_BF16_RTA else 0x7FFF + ((bits >> 16) & 1)
    return ((bits + bias) >> 16).to(torch.int16).view(torch.bfloat16)


# ── 4.1 精度、激活与 kernel 变体 ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("precision, tokens", [("per_tensor", 4), ("bf16", 16)])
def test_fp8_splitk_variants(kernels, precision, tokens):
    """基本用例只测 PTPC、N64 tile 且 M·topk > E。这里补 per-tensor / BF16 权重的 scale 分支、
    N128 tile 的宽拷贝，以及 M·topk <= E（4×8 <= 64）时 launcher 改为每条路由一个 task。"""
    _fp8_check(_fp8_prepare(FP8_TP2, tokens, precision), FP8_SPLITK | dict(tile_n_gate=128, tile_n_down=128))
    assert kernels == [("gateup", "splitk", "default"), ("down", "splitk", "default")]


@pytest.mark.parametrize("tokens, hidden, inter, precision, activation, gate_n", [
    (3, 512, 128, "ptpc", "silu", 32),
    (8, 512, 128, "per_tensor", "swiglu", 64),
    (7, 32768, 64, "bf16", "silu", 64),
    (5, 32768, 64, "ptpc", "silu", 32),
], ids=["ptpc-m3", "per_tensor-swiglu-m8", "bf16-h32768-m7", "ptpc-h32768-m5"])
def test_fp8_batch1_clear_graph(tokens, hidden, inter, precision, activation, gate_n):
    """基本用例只跑一次 eager 的 M1 PTPC。这里检查 Gate/Up 融合清零写满输出且不越过两侧
    红区（H32768、I64 需要多轮清零）、2–8 token、per-tensor / BF16 / SwiGLU 的 batch1
    epilogue，以及 graph replay 时输入与路由变化后结果仍正确。"""
    model = dict(HIDDEN_SIZE=hidden, INTER_SIZE=inter, TP=1, E=4, TOPK=2)
    call = _fp8_prepare(model, tokens, precision, activation=activation)
    storage = torch.full((tokens * hidden + 64,), -123.0, dtype=torch.bfloat16)
    call["output"] = storage[32:-32].view(tokens, hidden)
    runner = _fp8_check(call, dict(_impl="fly_decode", decode_alg="batch1",
                                   tile_n_gate=gate_n, tile_n_down=64))
    assert (storage[:32] == -123).all() and (storage[-32:] == -123).all()
    _replay_graph(call, runner, storage)


@pytest.mark.parametrize("tokens, config", [
    pytest.param(2, FP8_SPLITK | dict(tile_m_gate=64, tile_m_down=64, tile_n_gate=128,
                                      tile_n_down=128), id="splitk-m2"),
    pytest.param(64, dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=64, tile_n_gate=128,
                          tile_n_down=128, tile_k_gate=64), id="prefill-m64"),
])
def test_fp8_swiglu(tokens, config):
    """基本用例只测 SiLU。SwiGLU 的 clamp 与 (up + 1) 在 split-K 和 prefill 的 Gate/Up
    epilogue 各有一份（batch1 的在 test_fp8_batch1_clear_graph）；limit=0.02 让 clamp 生效。"""
    _fp8_check(_fp8_prepare(FP8_SMALL, tokens, "bf16", activation="swiglu", swiglu_limit=.02), config)


@pytest.mark.parametrize("precision, hidden, inter, down_path, padding, tokens", [
    # per-tensor：FlyDSL absmax 量化、Gate/Up 输出预清零和单一激活 scale，基本用例只测 PTPC。
    pytest.param("per_tensor", 2560, 320, "default", None, FP8_PREFILL_TOKENS, id="per_tensor-default"),
    pytest.param("per_tensor", 2560, 320, "1x4_64x256", 128, FP8_PREFILL_TOKENS, id="per_tensor-1x4"),
    pytest.param("per_tensor", 2560, 192, "8x1_compact", 128, FP8_PREFILL_TOKENS, id="per_tensor-compact"),
    # 8x1 full kernel 按 K 分三类调度：K192（基本用例）、K128n 不携带 B 地址（K256）与携带（K640）。
    pytest.param("ptpc", 2560, 256, "8x1_compact", 128, FP8_PREFILL_TOKENS, id="compact-i256"),
    pytest.param("ptpc", 2560, 640, "8x1_compact", 128, FP8_PREFILL_TOKENS, id="compact-i640"),
    # 8x1 的 N 调度下界：N512 只有 4 个 N128 tile。
    pytest.param("ptpc", 512, 192, "8x1_compact", 128, FP8_PREFILL_TOKENS, id="compact-h512"),
    # 1x4 在 K640 时 LDS 接近 64 KiB；同时覆盖专用 Down 的 padding=0（调优候选之一）。
    pytest.param("ptpc", 2560, 640, "1x4_64x256", 0, FP8_PREFILL_TOKENS, id="1x4-i640-pad0"),
    # M·topk <= E（4×8 <= 64）时 prefill launcher 只发 M·topk 个 task，基本用例的 M 远大于 E。
    pytest.param("ptpc", 2560, 320, "default", None, 4, id="default-m4"),
])
def test_fp8_prefill_variants(kernels, task_counts, precision, hidden, inter, down_path, padding, tokens):
    """基本用例只测 PTPC、H2560、I320/I192、padding 128 与大 M。每行注释写明补的分支或交界。"""
    model = FP8_TP2 | dict(HIDDEN_SIZE=hidden, INTER_SIZE=inter)
    _fp8_check(_fp8_prepare(model, tokens, precision), _fp8_prefill(inter, down_path, padding))
    assert kernels == [("gateup", "prefill_1x4", "default"), ("down", "prefill_1x4", down_path)]
    if down_path == "8x1_compact" and get_cu_num() == 80:
        assert len(task_counts) == 1 and bool((task_counts[0] > 0).all()), task_counts


# ── 4.2 CUDA graph ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("inter, tokens, config", [
    pytest.param(320, 16, FP8_SPLITK, id="splitk"),
    pytest.param(192, FP8_PREFILL_TOKENS, _fp8_prefill(192, "8x1_compact", 128), id="prefill-8x1_compact"),
])
def test_fp8_graph_replay(inter, tokens, config):
    """基本用例只跑 eager。graph replay 时排序、loc_ids 与 compact 任务表都必须在设备端按
    新路由重建（不能有 host 同步或沿用捕获时的值）；batch1 的 graph 见 test_fp8_batch1_clear_graph。"""
    call = _fp8_prepare(FP8_TP2 | dict(INTER_SIZE=inter), tokens, "ptpc")
    _replay_graph(call, _fp8_check(call, config))


# ── 4.3 tuned_moe 候选门槛与报错 ───────────────────────────────────────────────────────────────────────

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
def test_fp8_candidate_gates(precision, inter, gate_ns, default_ms):
    """tuned_moe 候选层的两条门槛：prefill Gate/Up 的不完整 N tile、默认 Down 超过 64 KiB LDS 的
    M×I 激活。两者编译都不报错、launch 才失败，不能交给编译失败剪枝；基本用例与 bench 用固定
    配置，不经过候选枚举。专用 Down 固定 M64，不受 LDS 门槛影响。"""
    configs = _prefill_candidates(_fp8_prepare(FP8_TP2 | dict(INTER_SIZE=inter), 65, precision))
    assert {c["tile_n_gate"] for c in configs} == gate_ns
    assert {c["tile_m_down"] for c in configs if "down_path" not in c} == default_ms
    assert {c["down_path"] for c in configs if "down_path" in c} == (
        set() if precision == "bf16" else {"1x4_64x256", "8x1_compact"})


def test_fp8_lds_64k_boundary(kernels):
    """边界：默认 Down 的 M128×I256 BF16 激活恰为 64 KiB，候选层保留它，launch 与结果都必须正确；
    上一个测试只检查候选枚举，基本用例的 BF16 prefill 只用 M64。"""
    _fp8_check(_fp8_prepare(FP8_TP2 | dict(INTER_SIZE=256), FP8_PREFILL_TOKENS, "bf16"),
               dict(_impl="fly_prefill", tile_m_gate=128, tile_m_down=128, tile_n_gate=128,
                    tile_n_down=128, tile_k_gate=64))
    assert kernels == [("gateup", "prefill_1x4", "default"), ("down", "prefill_1x4", "default")]


def test_fp8_split_m_tile_rejected():
    """报错：Gate/Up 与 Down 共用同一份按 M tile 排序的结果，固定配置给出不同 M 时必须在启动
    kernel 前拒绝。候选层只枚举相同的 M，基本用例与 bench 走不到这个分支。"""
    runner = make_moe_runner(dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=128,
                                  tile_n_gate=128, tile_n_down=128, tile_k_gate=128))
    with pytest.raises(ValueError, match="share the sorting M tile"):
        runner(**_fp8_prepare(FP8_TP2, 65, "ptpc"))


@pytest.mark.parametrize("alg", ["batch1", "splitk"])
@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_fp8_gateup_rejects_incomplete_k(alg, kind):
    """报错：K 不是 256 的倍数时 batch1 / split-K Gate/Up 必须在编译期拒绝。tuned_moe 不在
    候选层过滤 K，靠编译失败剪枝；基本用例与 bench 只走合法形状。"""
    from pyhip.ops.moe.flydsl.moe_gemm_2stage import compile_moe_gemm1

    with pytest.raises(AssertionError, match="divisible by 256"):
        compile_moe_gemm1(N=256, K=384, weight_dtype=kind,
                          weight_quant_type="no" if kind == "bf16" else "ptpc",
                          TOPK=2, BLOCK_TILE_SIZE_M=16, BLOCK_TILE_SIZE_N=64, alg=alg, E=4)


# ── 4.4 逐位格式 ───────────────────────────────────────────────────────────────────────────────────────

def test_fp8_bf16_rounding_bits():
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
def test_fp8_down_rounds_before_packing(path):
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


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# 区域 5｜性能测试（脚本入口，commands.bench；pytest 不收集）
# 用法：PYTHONPATH=<repo>/src python tests/ops/moe/test_moe_a8w4.py [--cases tp2_m1 ...]
# 68 个 case = 4 个形状 × M 按 2 的幂细扫 {1, 2, 4, …, 32768} 再加 M=48（E513 下 M=33–62 走排序块 32 的流式段，
# 2 的幂跳过了它），名为 <形状>_m<M>，SiLU，BF16 进出：
#   tp2 / tp4：Qwen3.8-Flash-Next，H=2560，E=513（512 routed + 最后一个融合 shared，每个 token 的最后槽位必选），
#     topk=11，I=320 / 160（prepare 补零到 192；Aiter 用同一份 BF16 权重补零到 192）；对照 Aiter FP8 ptpc。
#   mimo_tp8 / mimo_tp4：MiMo-V2.6-Flash，H=4096，E=256，topk=8，均匀路由，I=256 / 512；Aiter 没有这两个形状的
#     调优表，只测 A8W4。
# case 的输入种子按它在 BENCH_CASES 里的序号取（Qwen 在前、MiMo 在后）；权重与输入由 ``pyhip.testing.moe_a8w4``
# 生成（BF16 权重逐 expert 生成），种子取环境变量 MF_SEED（默认 0）。
# 协议：有对照时两种实现在同一进程、同一协议下交替计时。每次调用前写 512 MiB flush buffer，再用 ``torch.cuda._sleep``
# 让 GPU 空转（至少 200 µs，且不短于 warmup 后实测的各实现 host 下发一次调用耗时较大者的 2 倍），使 host
# 提前排队，event 只计完整调用的 GPU 时间；不设默认设备（它的 TorchFunctionMode 会给每个 torch 调用加 host 开销）。
# warmup 3 次，30 次取中位，4 组输入轮换。被计时的 A8W4 输出按 ``moe_a8w4.check`` 校验，任一组不过即退出非零。
# Aiter 对照只用 ``aiter/configs/model_configs/qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv``（复制到临时目录后经
# AITER_CONFIG_FMOE 指定），每个 case 断言 dispatcher 用的就是表中那一行。
# 输出：``MF_METRIC case=<名> value=<us> unit=us flops=<F> bytes=<B>``（A8W4 中位数）；有对照的 case 另有
# ``MF_REF case=<名> impl=aiter_fp8_ptpc value=<us> flops=<F> bytes=<B> ratio=<A8W4/aiter> kernels=<名>``；
# flops 按补零后的 I 计 6·M·topk·I·H，bytes 是各自的最小数据量（被选中 expert 的权重与 scale + 输入读、输出写）；
# ``INFO`` 行给所走路径、校验结果、带宽与算力利用率（MI308X：HBM 5.3 TB/s，FP8 稠密约 606 TFLOPS）。
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

# 形状前缀: (H, E, topk, 真实 I, 计算用的补零 I, 融合 shared expert, 对照 Aiter)
BENCH_SHAPES = {
    "tp2": (2560, 513, 11, 320, 320, True, True),
    "tp4": (2560, 513, 11, 160, 192, True, True),
    "mimo_tp8": (4096, 256, 8, 256, 256, False, False),
    "mimo_tp4": (4096, 256, 8, 512, 512, False, False),
}
BENCH_TOKENS = tuple(1 << k for k in range(16))  # 1, 2, 4, …, 32768
BENCH_EXTRA_TOKENS = (48,)  # 排序块 32 的小 M 流式段
BENCH_CASES = ([f"{tp}_m{m}" for tp in ("tp2", "tp4") for m in BENCH_TOKENS]
               + [f"{tp}_m{m}" for tp in ("tp2", "tp4") for m in BENCH_EXTRA_TOKENS]
               + [f"{tp}_m{m}" for tp in ("mimo_tp8", "mimo_tp4") for m in BENCH_TOKENS + BENCH_EXTRA_TOKENS])
HBM_BYTES_PER_S, FP8_FLOPS = 5.3e12, 606e12
WARMUP, ITERS, GROUPS = 3, 30, 4
FLUSH_BYTES, IDLE_US = 512 << 20, 200
BENCH_TABLE = "qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv"


def _aiter_row(fm, tokens, hidden, inter, experts, topk):
    from aiter import ActivationType, QuantType, dtypes

    key = (fm.get_gfx_runtime(), fm.get_cu_num(), fm.get_padded_M(tokens), hidden, inter, experts, topk,
           str(ActivationType.Silu), str(dtypes.bf16), str(dtypes.fp8), str(dtypes.fp8),
           str(QuantType.per_Token), 1, 0)
    fm.get_2stage_cfgs(fm.get_padded_M(tokens), hidden, inter, experts, topk, dtypes.bf16, dtypes.fp8,
                       dtypes.fp8, QuantType.per_Token, True, ActivationType.Silu, False, 0, 0)
    row = fm.cfg_2stages[0].get(key)
    if row is None or not 0 < float(row["us"]) < math.inf:
        raise RuntimeError(f"{BENCH_TABLE} has no valid row for {key}")
    return row


def _row_kernels(row):
    name = str(row["kernelName1"])
    if int(row["run_1stage"]):
        return f"asm_1stage:{name}"
    return name if name.startswith("impl__") else f"{name}+{row['kernelName2']}"


def _uses_row(meta, row):
    """dispatcher 最终采用的 metadata 是否就是表中这一行（识别表里出现的 1-stage asm、impl__ 与 CK 两段式三类）。"""
    name = str(row["kernelName1"])
    if int(meta.block_m) != int(row["block_m"]) or bool(meta.run_1stage) != bool(int(row["run_1stage"])):
        return False
    if meta.run_1stage:
        return meta.stage1.keywords.get("kernelName") == name
    if meta.full_impl is not None and name.startswith("impl__"):
        configs = [c.cell_contents for c in meta.full_impl.__closure__ or () if isinstance(c.cell_contents, str)]
        return configs == [name.split("__", 2)[2]]
    if meta.full_impl is None and name.startswith("moe_ck2stages_"):
        return (meta.stage1.keywords.get("kernelName") == name
                and meta.stage2.keywords.get("kernelName") == str(row["kernelName2"]))
    return False


def _cycles_per_us():
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(1000)
    start.record()
    torch.cuda._sleep(1_000_000)
    end.record()
    torch.cuda.synchronize()
    return 1_000_000 / (start.elapsed_time(end) * 1e3)


def _host_us(fn, cycles_per_us):
    """一次调用的 host 下发耗时（中位数）；GPU 先空转，保证 host 不被队列反压。"""
    times = []
    for i in range(5):
        torch.cuda._sleep(int(20_000 * cycles_per_us))
        began = time.perf_counter()
        fn(i % GROUPS)
        times.append((time.perf_counter() - began) * 1e6)
        torch.cuda.synchronize()
    return statistics.median(times)


def _traffic(ids, per_expert, hidden):
    """最小数据量：被选中 expert 的全部权重（含 scale）+ 输入读与输出写（字节）。"""
    return ids.unique().numel() * per_expert + ids.shape[0] * hidden * 2 * 2


def _bytes_per_expert(weights, experts):
    values = weights.values() if isinstance(weights, dict) else vars(weights).values()
    return sum(t.numel() * t.element_size() for t in values if isinstance(t, torch.Tensor)) // experts


def _bench_case(name, weights, cycles_per_us, flush, seed):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType

    tp, tokens = name.rsplit("_m", 1)[0], int(name.rsplit("_m", 1)[1])
    hidden, experts, topk, inter, inter_pad, shared, compare = BENCH_SHAPES[tp]
    w, prepared, fp8_w = weights
    index = BENCH_CASES.index(name)
    groups = [oracle.make_inputs(tokens, hidden, experts, topk, seed=seed * 1000 + 100 + index * GROUPS + g,
                                 shared=shared)
              for g in range(GROUPS)]
    outputs = [torch.empty_like(x) for x, _, _ in groups]
    theirs_out = [None] * GROUPS

    def ours(g):
        x, topk_weight, topk_ids = groups[g]
        moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=outputs[g])

    def theirs(g):
        x, topk_weight, topk_ids = groups[g]
        theirs_out[g] = fm.fused_moe(x, fp8_w["w1"], fp8_w["w2"], topk_weight, topk_ids,
                                     activation=ActivationType.Silu, quant_type=QuantType.per_Token,
                                     w1_scale=fp8_w["w1_scale"], w2_scale=fp8_w["w2_scale"])

    impls = (("ours", ours), ("theirs", theirs)) if compare else (("ours", ours),)
    if compare:
        row = _aiter_row(fm, tokens, hidden, inter_pad, experts, topk)
        seen = []
        resolve = fm.get_2stage_cfgs

        def spy(*args, **kwargs):
            meta = resolve(*args, **kwargs)
            seen.append(meta)
            return meta

        fm.get_2stage_cfgs = spy
        try:
            theirs(0)
        finally:
            fm.get_2stage_cfgs = resolve
        expected = _row_kernels(row)
        if not seen or not _uses_row(seen[-1], row):
            raise RuntimeError(f"{name}: Aiter did not run the tuned row {expected} (block_m={row['block_m']}); "
                               f"got {seen[-1] if seen else None}")
    for i in range(WARMUP):
        for _, fn in impls:
            fn(i % GROUPS)
    # 空转要盖住 host 下发一次调用的时间，否则 host 开销会落进计时区间。
    host = {impl: _host_us(fn, cycles_per_us) for impl, fn in impls}
    idle_us = max(IDLE_US, 2 * max(host.values()))
    sleep_cycles = int(idle_us * cycles_per_us)

    events = {impl: [] for impl, _ in impls}
    for i in range(ITERS):
        for impl, fn in impls:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.zero_()
            torch.cuda._sleep(sleep_cycles)
            start.record()
            fn(i % GROUPS)
            end.record()
            events[impl].append((start, end))
    torch.cuda.synchronize()
    path = list(moe_a8w4.last_path)
    us = {impl: statistics.median(s.elapsed_time(e) * 1e3 for s, e in pairs) for impl, pairs in events.items()}

    mode = oracle.path_activation(path)
    worst = None
    for g, (x, topk_weight, topk_ids) in enumerate(groups):
        ref = oracle.reference(x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight, topk_ids,
                               activation=mode)
        result = oracle.check(outputs[g], ref)
        if not result["ok"]:
            print(f"FAIL case={name} group={g} check={result}", flush=True)
            sys.exit(1)
        if compare and not bool(torch.isfinite(theirs_out[g]).all()):
            print(f"FAIL case={name} group={g} aiter output is not finite", flush=True)
            sys.exit(1)
        if worst is None or result["err_ratio"] > worst["err_ratio"]:
            worst = result
        del ref

    flops = 6 * tokens * topk * inter_pad * hidden
    bytes_ours = statistics.mean(_traffic(ids, _bytes_per_expert(prepared, experts), hidden) for _, _, ids in groups)

    def util(t_us, nbytes):
        bw, fl = nbytes / (t_us * 1e-6), flops / (t_us * 1e-6)
        return (f"{bw / 1e12:.2f}TB/s({bw / HBM_BYTES_PER_S * 100:.1f}%) "
                f"{fl / 1e12:.1f}TFLOPS({fl / FP8_FLOPS * 100:.1f}%)")

    print(f"MF_METRIC case={name} value={us['ours']:.2f} unit=us flops={flops} bytes={bytes_ours:.0f}", flush=True)
    versus = ""
    if compare:
        bytes_theirs = statistics.mean(_traffic(ids, _bytes_per_expert(fp8_w, experts), hidden) for _, _, ids in groups)
        print(f"MF_REF case={name} impl=aiter_fp8_ptpc value={us['theirs']:.2f} flops={flops} bytes={bytes_theirs:.0f} "
              f"ratio={us['ours'] / us['theirs']:.3f} kernels={expected}", flush=True)
        versus = f" aiter={util(us['theirs'], bytes_theirs)}"
    print(f"INFO case={name} I={inter}->{inter_pad} path={'>'.join(path)} activation={mode} "
          f"a8w4={util(us['ours'], bytes_ours)}{versus} "
          f"worst_check=calc_diff:{worst['calc_diff']:.2e},err_ratio:{worst['err_ratio'] * 100:.2f}% "
          f"host_us={','.join(f'{k}:{v:.0f}' for k, v in host.items())} idle_us={idle_us:.0f}", flush=True)


def _bench_main():
    parser = argparse.ArgumentParser(description="A8W4 MoE 性能测试：pyhip fused_moe_a8w4 对照 Aiter FP8 ptpc")
    parser.add_argument("--cases", nargs="+", choices=BENCH_CASES, default=BENCH_CASES)
    args = parser.parse_args()
    seed = int(os.environ.get("MF_SEED", "0"))
    # 两个开关与调优表路径都在用到时才读，放在导入 aiter / FlyDSL 之后设置也生效（下面先清 Aiter 的配置缓存）。
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"  # 测当前源码，不复用旧磁盘编译产物。
    os.environ["AITER_ONLINE_TUNE"] = "0"
    table_src = Path(aiter.__file__).parent / "configs/model_configs" / BENCH_TABLE
    table = Path(tempfile.mkdtemp(prefix="bench_moe_a8w4_")) / BENCH_TABLE  # 副本：Aiter 的去重检查可能改写表文件
    shutil.copyfile(table_src, table)
    os.environ["AITER_CONFIG_FMOE"] = str(table)
    import aiter.fused_moe as fm
    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.shuffle import shuffle_weight

    AITER_CONFIGS.get_config_file.cache_clear()
    fm.cfg_2stages = None
    fm.get_2stage_cfgs.cache_clear()
    began = time.time()
    print(f"INFO seed={seed} aiter_table={table_src} warmup={WARMUP} iters={ITERS} groups={GROUPS}", flush=True)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    cycles_per_us = _cycles_per_us()
    for tp_index, tp in enumerate(BENCH_SHAPES):
        names = [n for n in args.cases if n.rsplit("_m", 1)[0] == tp]
        if not names:
            continue
        hidden, experts, topk, inter, inter_pad, shared, compare = BENCH_SHAPES[tp]
        w = oracle.make_weights(experts, hidden, inter, seed=seed * 1000 + tp_index,
                                fp8_inter=inter_pad if compare else None)
        prepared = moe_a8w4.prepare_a8w4_weights(w["w1"], w["w1_scale"], w["w2"], w["w2_scale"])
        fp8_w = None
        if compare:
            fp8_w = dict(w1=shuffle_weight(w.pop("w1_fp8"), layout=(16, 16)), w1_scale=w.pop("w1_fp8_scale"),
                         w2=shuffle_weight(w.pop("w2_fp8"), layout=(16, 16)), w2_scale=w.pop("w2_fp8_scale"))
        for name in names:
            _bench_case(name, (w, prepared, fp8_w), cycles_per_us, flush, seed)
        del w, prepared, fp8_w
        torch.cuda.empty_cache()
    print(f"INFO done in {time.time() - began:.0f}s", flush=True)


if __name__ == "__main__":
    _bench_main()
