"""moe_gemm_2stage 换成上游 ROCm/aiter@df3e0b2f gfx9_fp8 之后，pyhip 一侧的分流与适配。

- ``gemm.gemm_implementation`` 的每条规则，以及 ``compile_gemm`` 实际调用的是上游
  （``gemm1.compile_moe_gemm1`` / ``gemm2.compile_moe_gemm2``）还是 ``*_legacy`` 的 builder；
  上游 batch1 / compact launcher 换回 pyhip 调用约定的包装；分流在按设备的编译缓存里只判定一次。
- compact Down：pyhip 按设备 CU 数分配的任务缓冲，与上游按同一 CU 数的拆分策略容量一致。
- common.py：替代 aiter 的 ``get_gfx_runtime`` / ``get_cu_num`` / ``_run_compiled``；BF16 舍入开关
  （默认简化 RTA，``AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=0`` 切到 RTE）。
- quant.py（上游版本）：absmax 不再自行清零，amax 为 0 时输出 0。
- pyhip 调用方（src、tests、benchmarks、experiments）在原模块名下用到的符号；kernel 模块不导入 aiter。

分流与适配只构建 launcher（host 侧检查），不编译 kernel；只有舍入、``_run_compiled`` 与 quant 编译小 kernel。
Qwen3.8 形状上的端到端数值见 ``test_qwen38_ptpc.py``。
"""

import contextlib
import functools
import importlib.util
import inspect
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from pyhip.ops.moe.flydsl.moe_gemm_2stage import common, gemm, gemm1_legacy, gemm2_legacy

UPSTREAM, LEGACY = gemm.UPSTREAM, gemm.LEGACY
RTA_ENV = "AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED"


@pytest.fixture(autouse=True)
def _gpu_environment(monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("上游 gfx9_fp8 只用于 gfx942")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # 测当前源码，不复用旧磁盘编译产物。
    monkeypatch.delenv("MOE_PREFILL_TILE_K", raising=False)
    monkeypatch.delenv("CU_NUM", raising=False)


@pytest.fixture
def builds(monkeypatch):
    """``compile_gemm`` 换用全新的缓存，记录它调用的 builder：``[{impl, stage, kwargs, launcher}]``。"""
    calls = []

    def spy(impl, stage, builder):
        def build(**kwargs):
            record = dict(impl=impl, stage=stage, kwargs=kwargs)
            calls.append(record)
            record["launcher"] = builder(**kwargs)
            return record["launcher"]

        return build

    monkeypatch.setattr(gemm, "_compile_gemm_cached",
                        functools.cache(gemm._compile_gemm_cached.__wrapped__))
    monkeypatch.setattr(gemm, "compile_moe_gemm1", spy(UPSTREAM, "gateup", gemm.compile_moe_gemm1))
    monkeypatch.setattr(gemm, "compile_moe_gemm2", spy(UPSTREAM, "down", gemm.compile_moe_gemm2))
    monkeypatch.setattr(gemm1_legacy, "compile_moe_gemm1",
                        spy(LEGACY, "gateup", gemm1_legacy.compile_moe_gemm1))
    monkeypatch.setattr(gemm2_legacy, "compile_moe_gemm2",
                        spy(LEGACY, "down", gemm2_legacy.compile_moe_gemm2))
    return calls


# Qwen3.8 TP2 形状（H=2560，I=320，E=512，top-10）上 FP8 ptpc 的 tuned_moe 参数。
GATEUP = dict(N=640, K=2560, weight_dtype="fp8", weight_quant_type="ptpc", act_quant_type="ptpc",
              TOPK=10, E=512, BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=128, stage="gateup",
              alg="prefill_1x4", tile_k=128, METADATA_TILE_SIZE_M=64)
DOWN = dict(N=2560, K=320, weight_dtype="fp8", weight_quant_type="ptpc", act_quant_type="ptpc",
            TOPK=10, E=512, BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=128, stage="down",
            alg="prefill_1x4", USE_ATOMIC_WRITE=False)
GATEUP_BATCH1 = dict(GATEUP, alg="batch1", BLOCK_TILE_SIZE_M=16, BLOCK_TILE_SIZE_N=32, tile_k=None,
                     METADATA_TILE_SIZE_M=None, fused_down_clear=True)
DOWN_BATCH1 = dict(DOWN, alg="batch1", BLOCK_TILE_SIZE_M=16, BLOCK_TILE_SIZE_N=64, USE_ATOMIC_WRITE=True)
DOWN_1X4 = dict(DOWN, BLOCK_TILE_SIZE_N=256, down_path="1x4_64x256", down_output_padding_bytes=128)
DOWN_COMPACT = dict(DOWN, K=192, down_path="8x1_compact", down_output_padding_bytes=128)
BF16 = dict(weight_dtype="bf16", weight_quant_type="no", act_quant_type="no")
PER_TENSOR = dict(weight_quant_type="per_tensor", act_quant_type="per_tensor")


def _route(case_id, expected, base, arch="gfx942", **overrides):
    return pytest.param(dict(base, **overrides), arch, expected, id=case_id)


ROUTES = [
    # Qwen3.8 FLY_CONFIGS 用到的 Gate/Up 与 Down。
    _route("gateup-prefill", UPSTREAM, GATEUP),
    _route("gateup-splitk", UPSTREAM, GATEUP, alg="splitk", BLOCK_TILE_SIZE_M=16,
           METADATA_TILE_SIZE_M=16, tile_k=None),
    _route("gateup-batch1", UPSTREAM, GATEUP_BATCH1),
    _route("down-prefill", UPSTREAM, DOWN),
    _route("down-splitk", UPSTREAM, DOWN, alg="splitk", BLOCK_TILE_SIZE_M=16, USE_ATOMIC_WRITE=True),
    _route("down-batch1", UPSTREAM, DOWN_BATCH1),
    _route("down-1x4", UPSTREAM, DOWN_1X4),
    _route("down-compact", UPSTREAM, DOWN_COMPACT),
    # has_num_local_tokens 只有上游支持，先于其它条件；force_batch1_path 先把 alg 改成 batch1。
    _route("num-local-tokens-first", UPSTREAM, GATEUP_BATCH1, activation="situv2",
           has_num_local_tokens=True),
    _route("force-batch1", UPSTREAM, GATEUP_BATCH1, alg="prefill_1x4", K=2304, tile_k=256,
           force_batch1_path=True),
    # 上游只覆盖 gfx942 的 BF16/FP8 × SiLU/SwiGLU，Gate/Up 的 M 分块与排序一致。
    _route("gfx950-gateup", LEGACY, GATEUP, arch="gfx950"),
    _route("gfx950-down", LEGACY, DOWN_1X4, arch="gfx950"),
    _route("bf16", UPSTREAM, GATEUP, **BF16, tile_k=64),
    _route("fp4", LEGACY, GATEUP_BATCH1, weight_dtype="fp4", weight_quant_type="mxfp4",
           act_quant_type="no"),
    _route("swiglu", UPSTREAM, GATEUP, activation="swiglu"),
    _route("situv2", LEGACY, GATEUP, activation="situv2", situ_beta=.5, situ_linear_beta=2.0),
    _route("metadata-none", UPSTREAM, GATEUP, METADATA_TILE_SIZE_M=None),
    _route("metadata-ne-bm", LEGACY, GATEUP, METADATA_TILE_SIZE_M=256),
    _route("alg-unknown", LEGACY, GATEUP, alg="prefill"),
    _route("stage-unknown", LEGACY, DOWN, stage="combined"),
    # Gate/Up：不接受 Down 的参数；prefill 要求完整 N tile、偶数个 K tile，BF16 另查 BK 与 LDS。
    _route("gateup-down-path", LEGACY, GATEUP, down_path="1x4_64x256"),
    _route("gateup-padding", LEGACY, GATEUP, down_output_padding_bytes=128),
    _route("gateup-padding-0", UPSTREAM, GATEUP, down_output_padding_bytes=0),
    _route("gateup-n-tile", LEGACY, GATEUP, BLOCK_TILE_SIZE_N=256),
    _route("gateup-bk256", UPSTREAM, GATEUP, tile_k=256),
    _route("gateup-odd-k-tiles", LEGACY, GATEUP, K=2304, tile_k=256),
    _route("gateup-fp8-default-bk128", LEGACY, GATEUP, K=384, tile_k=None),
    _route("gateup-bf16-default-bk64", UPSTREAM, GATEUP, **BF16, K=384, tile_k=None),
    _route("gateup-bf16-bk256", LEGACY, GATEUP, **BF16, tile_k=256),
    _route("gateup-bf16-lds-64k", UPSTREAM, GATEUP, **BF16, BLOCK_TILE_SIZE_M=128,
           METADATA_TILE_SIZE_M=128, tile_k=128),
    _route("gateup-bf16-lds-over", LEGACY, GATEUP, **BF16, BLOCK_TILE_SIZE_M=256,
           METADATA_TILE_SIZE_M=256, tile_k=128),
    # default Down：不带 padding；prefill 要求 K%64、N%256，BF16 的 A tile 不超过 64 KiB。
    _route("down-padding", LEGACY, DOWN, down_output_padding_bytes=128),
    _route("down-padding-0", UPSTREAM, DOWN, down_output_padding_bytes=0),
    _route("down-prefill-k64", LEGACY, DOWN, K=352),
    _route("down-prefill-n256", LEGACY, DOWN, N=2432),
    _route("down-splitk-n128", UPSTREAM, DOWN, alg="splitk", BLOCK_TILE_SIZE_M=16, N=2432,
           USE_ATOMIC_WRITE=True),
    _route("down-bf16-lds-64k", UPSTREAM, DOWN, **BF16, K=512),
    _route("down-bf16-lds-over", LEGACY, DOWN, **BF16, K=576),
    # 1x4 / compact：prefill、FP8、M64，padding 0/32/64/128，K%64、N%256，LDS 不超过 64 KiB。
    _route("special-splitk", LEGACY, DOWN_1X4, alg="splitk"),
    _route("special-bf16", LEGACY, DOWN_1X4, **BF16),
    _route("special-m128", LEGACY, DOWN_1X4, BLOCK_TILE_SIZE_M=128),
    _route("special-padding-none", UPSTREAM, DOWN_1X4, down_output_padding_bytes=None),
    _route("special-padding-32", UPSTREAM, DOWN_1X4, down_output_padding_bytes=32),
    _route("special-padding-96", LEGACY, DOWN_1X4, down_output_padding_bytes=96),
    _route("special-k64", LEGACY, DOWN_1X4, K=352),
    _route("special-n256", LEGACY, DOWN_1X4, N=2432),
    _route("special-lds-ptpc-k832", UPSTREAM, DOWN_1X4, K=832),
    _route("special-lds-ptpc-k896", LEGACY, DOWN_1X4, K=896),
    _route("special-lds-per-tensor-k896", UPSTREAM, DOWN_1X4, **PER_TENSOR, K=896),
    _route("special-lds-per-tensor-k960", LEGACY, DOWN_1X4, **PER_TENSOR, K=960),
    _route("1x4-n128", LEGACY, DOWN_1X4, BLOCK_TILE_SIZE_N=128),
    # compact：N128 tile，K∈{192,256,384,512,640}（没有 Qwen3.8 TP2 的 320），N>=512，0<E<=2048。
    *[_route(f"compact-k{k}", UPSTREAM, DOWN_COMPACT, K=k) for k in (256, 384, 512, 640)],
    _route("compact-k320", LEGACY, DOWN_COMPACT, K=320),
    _route("compact-k448", LEGACY, DOWN_COMPACT, K=448),
    _route("compact-per-tensor", UPSTREAM, DOWN_COMPACT, **PER_TENSOR, K=640),
    _route("compact-n512", UPSTREAM, DOWN_COMPACT, N=512),
    _route("compact-n256", LEGACY, DOWN_COMPACT, N=256),
    _route("compact-n256-tile", LEGACY, DOWN_COMPACT, BLOCK_TILE_SIZE_N=256),
    _route("compact-e2048", UPSTREAM, DOWN_COMPACT, E=2048),
    _route("compact-e2049", LEGACY, DOWN_COMPACT, E=2049),
    _route("compact-e0", LEGACY, DOWN_COMPACT, E=0),
    _route("compact-e-none", LEGACY, DOWN_COMPACT, E=None),
    # 上游没有的 Down path。
    _route("8x1", LEGACY, DOWN, BLOCK_TILE_SIZE_M=256, down_path="8x1", down_output_padding_bytes=128),
    _route("8x1-m64", LEGACY, DOWN_COMPACT, down_path="8x1"),
    _route("unknown-path", LEGACY, DOWN_COMPACT, down_path="2x4"),
]


@pytest.mark.parametrize("params, arch, expected", ROUTES)
def test_compile_gemm_routes(builds, monkeypatch, params, arch, expected):
    monkeypatch.setattr(gemm, "get_gfx_runtime", lambda: arch)
    assert gemm.gemm_implementation(**params) == expected
    try:
        gemm.compile_gemm(**params)
    except Exception:  # noqa: BLE001
        # 判给上游的配置上游 builder 必须接受；legacy 可以拒绝两套都不支持的配置（报 pyhip 原来的错）。
        if expected == UPSTREAM and not params.get("has_num_local_tokens"):
            raise
    stage = "gateup" if params["stage"] == "gateup" else "down"
    assert [(c["impl"], c["stage"]) for c in builds] == [(expected, stage)]


ADAPTERS = [
    # (参数, 实现, 包装, builder 收到的参数)
    pytest.param(GATEUP_BATCH1, UPSTREAM, "batch1", dict(fused_down_clear=True, has_num_local_tokens=False),
                 id="upstream-gateup-batch1"),
    pytest.param(DOWN_BATCH1, UPSTREAM, "batch1",
                 dict(down_output_padding_bytes=0, USE_ATOMIC_WRITE=True, has_num_local_tokens=False),
                 id="upstream-down-batch1"),
    pytest.param(dict(GATEUP_BATCH1, has_num_local_tokens=True), UPSTREAM, None,
                 dict(has_num_local_tokens=True), id="upstream-gateup-num-local-tokens"),
    pytest.param(dict(DOWN_BATCH1, has_num_local_tokens=True), UPSTREAM, None,
                 dict(has_num_local_tokens=True), id="upstream-down-num-local-tokens"),
    pytest.param(dict(GATEUP, tile_k=256), UPSTREAM, None, dict(tile_k=256, alg="prefill_1x4"),
                 id="upstream-gateup-prefill"),
    pytest.param(dict(DOWN, down_output_padding_bytes=None), UPSTREAM, None,
                 dict(down_output_padding_bytes=0, USE_ATOMIC_WRITE=False), id="upstream-down-default"),
    pytest.param(DOWN_1X4, UPSTREAM, None, dict(down_path="1x4_64x256", down_output_padding_bytes=128),
                 id="upstream-down-1x4"),
    pytest.param(DOWN_COMPACT, UPSTREAM, "compact",
                 dict(down_path="8x1_compact", down_output_padding_bytes=128, E=512),
                 id="upstream-down-compact"),
    # legacy 的 launcher 本来就是 pyhip 的调用约定（batch1、compact 都不包装），并收到 legacy 独有的参数。
    pytest.param(dict(GATEUP_BATCH1, **BF16, activation="situv2", situ_beta=.5, situ_linear_beta=2.0,
                      swiglu_limit=float("inf")), LEGACY, None,
                 dict(situ_beta=.5, situ_linear_beta=2.0, mxfp4_gate_up_interleaved=True,
                      fused_down_clear=True, METADATA_TILE_SIZE_M=None), id="legacy-gateup-situv2-batch1"),
    pytest.param(dict(GATEUP, METADATA_TILE_SIZE_M=256), LEGACY, None,
                 dict(METADATA_TILE_SIZE_M=256, tile_k=128), id="legacy-gateup-metadata-m256"),
    pytest.param(dict(DOWN_COMPACT, K=320), LEGACY, None,
                 dict(down_path="8x1_compact", down_output_padding_bytes=128, E=512),
                 id="legacy-down-compact-k320"),
    pytest.param(dict(DOWN, down_output_padding_bytes=None, arch="gfx950"), LEGACY, None,
                 dict(down_output_padding_bytes=None, USE_ATOMIC_WRITE=False), id="legacy-down-default"),
]


@pytest.mark.parametrize("params, implementation, wrapper, expected_kwargs", ADAPTERS)
def test_compile_gemm_adapts_launchers(builds, monkeypatch, params, implementation, wrapper, expected_kwargs):
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.gemm2_8x1_compact_legacy import device_cu_count

    params = dict(params)
    arch = params.pop("arch", "gfx942")  # 只改分流看到的架构；builder 仍按本机 gfx942 构建。
    monkeypatch.setattr(gemm, "get_gfx_runtime", lambda: arch)
    # CU_NUM 只改变上游的 get_cu_num()；compact 的任务缓冲（allocate_task_buffers）按设备真实 CU 数分配。
    monkeypatch.setenv("CU_NUM", "40")
    launcher = gemm.compile_gemm(**params)
    [record] = builds
    assert record["impl"] == implementation
    assert {key: record["kwargs"][key] for key in expected_kwargs} == expected_kwargs
    if wrapper is None:
        assert launcher is record["launcher"]
        return
    closure = inspect.getclosurevars(launcher.func).nonlocals
    assert closure["launch"] is record["launcher"]
    hints = record["launcher"].compile_hints
    assert {key: launcher.compile_hints.get(key) for key in hints} == hints
    if wrapper == "batch1":
        # 上游 batch1 取路由数 batch*TOPK，并多一个 token 数指针（不设 has_num_local_tokens 时不读）。
        assert closure["TOPK"] == params["TOPK"]
        assert list(inspect.signature(launcher.func).parameters) == [
            "p_input", "p_weight", "p_output", "p_topk_ids", "p_clear_or_weights", "p_w_scale", "M", "stream"]
    else:
        assert closure["cu_count"] == device_cu_count() != 40
        assert "cu_count" not in inspect.signature(launcher.func).parameters


def test_compile_gemm_routes_once_per_device(builds, monkeypatch):
    decisions = []
    route = gemm.gemm_implementation

    def counted(**params):
        decisions.append(params)
        return route(**params)

    monkeypatch.setattr(gemm, "gemm_implementation", counted)
    first = gemm.compile_gemm(**DOWN_BATCH1)
    assert gemm.compile_gemm(**DOWN_BATCH1) is first
    # force_batch1_path 在进缓存前改写 alg，与 alg="batch1" 共用一项。
    assert gemm.compile_gemm(**dict(DOWN_BATCH1, alg="splitk", force_batch1_path=True)) is first
    assert len(decisions) == len(builds) == 1
    key = gemm.get_device_cache_key()
    monkeypatch.setattr(gemm, "get_device_cache_key", lambda: ("other-gpu", *key[1:]))
    assert gemm.compile_gemm(**DOWN_BATCH1) is not first
    assert len(decisions) == len(builds) == 2
    monkeypatch.setattr(gemm, "get_device_cache_key", lambda: key)
    assert gemm.compile_gemm(**DOWN_BATCH1) is first
    assert len(decisions) == len(builds) == 2


def test_compile_gemm_prefill_tile_k_env(builds, monkeypatch):
    monkeypatch.setenv("MOE_PREFILL_TILE_K", "256")
    gemm.compile_gemm(**dict(GATEUP, tile_k=None))
    gemm.compile_gemm(**GATEUP)
    gemm.compile_gemm(**dict(DOWN, tile_k=None))
    # BK 在分流前解析：K=2304 默认 BK128 走上游，环境变量给 BK256 后是奇数个 K tile，判给 legacy。
    with contextlib.suppress(Exception):
        gemm.compile_gemm(**dict(GATEUP, K=2304, tile_k=None))
    assert [(c["impl"], c["stage"], c["kwargs"]["tile_k"]) for c in builds] == [
        (UPSTREAM, "gateup", 256), (UPSTREAM, "gateup", 128), (UPSTREAM, "down", None),
        (LEGACY, "gateup", 256)]


def test_compact_task_buffers_match_upstream_split():
    from pyhip.ops.moe.flydsl.moe_gemm_2stage import gemm2_8x1_compact as compact
    from pyhip.ops.moe.flydsl.moe_gemm_2stage import gemm2_8x1_compact_legacy as legacy

    for cu_count in (1, 4, 5, 80, 304):
        for experts in (1, 9, 512, 2048):
            # 排序后的 M64 块数：0、少于 4 块、Qwen3.8 TP4 T=8200 与 T=65536。
            for blocks in (0, 1, 3, 4, 5, 37, 1785, 10744):
                assert legacy.task_capacities(blocks, experts, cu_count=cu_count) == \
                    compact.task_capacities(blocks, experts, cu_count), (cu_count, experts, blocks)
    sorted_expert_ids = torch.empty(1785, dtype=torch.int32, device="cuda")
    pyhip = compact.allocate_task_buffers(sorted_expert_ids, 512)  # tuned_moe 的调用方式
    upstream = compact.allocate_task_buffers_upstream(sorted_expert_ids, 512, legacy.device_cu_count())
    assert [tuple(t.shape) for t in pyhip] == [tuple(t.shape) for t in upstream]


def test_chip_info_stand_ins(monkeypatch):
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    assert common.get_gfx_runtime() == properties.gcnArchName.split(":")[0].lower() == "gfx942"
    assert common.get_cu_num() == properties.multi_processor_count
    monkeypatch.setenv("CU_NUM", "0")
    assert common.get_cu_num() == properties.multi_processor_count
    # 上游 Down（1x4、compact 的满块）按 CU 数识别 MI308X 的拓扑；CU_NUM 与 aiter 一样可以覆盖。
    monkeypatch.setenv("CU_NUM", "80")
    assert common.get_cu_num() == 80 and common.get_down_device_config() == (True, 4)
    monkeypatch.setenv("CU_NUM", "304")
    assert common.get_down_device_config() == (False, 8)
    monkeypatch.setattr(common, "get_gfx_runtime", lambda: "gfx950")
    with pytest.raises(RuntimeError, match="gfx942"):
        common.get_down_device_config()


@pytest.mark.parametrize("value, rta", [(None, True), ("1", True), ("True", True), ("0", False)],
                         ids=["unset", "1", "True", "0"])
def test_bf16_rounding_switch(monkeypatch, value, rta):
    from pyhip.ops.moe.flydsl.moe_gemm_2stage import common_legacy

    if value is None:
        monkeypatch.delenv(RTA_ENV, raising=False)
    else:
        monkeypatch.setenv(RTA_ENV, value)
    # 开关在导入时读取：另载一份 common.py 看它的取值，不改动已导入的模块。
    spec = importlib.util.spec_from_file_location("_moe_gemm_2stage_common_probe", common.__file__)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe._SIMPLIFIED_BF16_RTA is rta
    assert probe._SIMPLIFIED_BF16_RTE is False
    # legacy kernel 用的是同一个函数，跟随同一开关。
    assert common_legacy._f32_to_bf16 is common._f32_to_bf16


@pytest.mark.parametrize("rta", [True, False], ids=["rta", "rte"])
def test_f32_to_bf16_follows_switch(monkeypatch, rta):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.common import _f32_to_bf16, torch_tensor_to_pointer as ptr

    # 开关在 kernel 追踪时读取；这里的 kernel 每次新建，按当前开关编译。
    monkeypatch.setattr(common, "_SIMPLIFIED_BF16_RTA", rta)
    monkeypatch.setattr(common, "_SIMPLIFIED_BF16_RTE", False)
    upper = [0, 1, 0x7f, 0x80, 0x3f80, 0x3f81, 0x7f7f, 0x8000, 0x8001, 0xbf80, 0xbf81, 0xff7f]
    bits = [(hi << 16) | lo for hi in upper for lo in (0, 1, 0x7fff, 0x8000, 0x8001, 0xffff)]
    bits += [0x7f800000, 0xff800000, 0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff]
    bits += [0] * (512 - len(bits))
    raw = torch.tensor(bits, dtype=torch.int64)
    x_cpu = raw.to(torch.int32).view(torch.float32)
    if rta:
        # 简化 RTA：加 0x8000 后截断（NaN 可能变成 Inf，与 kernel 逐位相同）。
        expected = ((raw + 0x8000) >> 16).to(torch.int16).view(torch.bfloat16)
    else:
        expected = x_cpu.bfloat16()  # PyTorch 就近舍入到偶数（RTE），NaN 仍为 NaN。
    x = x_cpu.cuda()
    vector_out = torch.empty(x.shape, dtype=torch.bfloat16, device="cuda")
    scalar_out = torch.empty_like(vector_out)

    @flyc.kernel
    def convert(A: fx.Pointer, B: fx.Pointer, C: fx.Pointer):
        base = fx.thread_idx.x * 8
        values = fx.make_view(A + base, fx.make_layout(8, 1)).load()
        fx.make_view(B + base, fx.make_layout(8, 1)).store(_f32_to_bf16(values))
        for i in fx.range_constexpr(8):
            C[base + i] = _f32_to_bf16(A[base + i])

    @flyc.jit
    def launch(A: fx.Pointer, B: fx.Pointer, C: fx.Pointer, stream: fx.Stream):
        convert(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    common._run_compiled(launch, ptr(x), ptr(vector_out), ptr(scalar_out), torch.cuda.current_stream())
    tie = bits.index(0x3f808000)  # 偶数 LSB 的中点：RTE 保持 0x3f80，RTA 进到 0x3f81。
    for result in (vector_out.cpu(), scalar_out.cpu()):
        assert int(result[tie].view(torch.int16)) == (0x3f81 if rta else 0x3f80)
        if rta:
            assert torch.equal(result.view(torch.int16), expected.view(torch.int16))
        else:
            nan = expected.isnan()
            assert torch.equal(result.isnan(), nan)
            assert torch.equal(result[~nan].view(torch.int16), expected[~nan].view(torch.int16))


def test_run_compiled_compiles_once_and_recovers(monkeypatch):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as ptr

    @flyc.kernel
    def copy(X: fx.Pointer, Y: fx.Pointer):
        Y[fx.thread_idx.x] = X[fx.thread_idx.x]

    @flyc.jit
    def launch(X: fx.Pointer, Y: fx.Pointer, stream: fx.Stream):
        copy(X, Y).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    def current_context():
        try:
            return ir.Context.current
        except Exception:  # noqa: BLE001
            return None

    def leaky_compile(exe, *args):
        # flyc.compile 失败时会留下已进入的 ir.Context。
        ir.Context().__enter__()
        raise RuntimeError("compile failed")

    compile_ = flyc.compile
    stream = torch.cuda.current_stream()
    x, y = torch.arange(64, dtype=torch.float32, device="cuda"), torch.zeros(64, device="cuda")
    monkeypatch.setattr(flyc, "compile", leaky_compile)
    with pytest.raises(RuntimeError, match="compile failed"):
        common._run_compiled(launch, ptr(x), ptr(y), stream)
    assert current_context() is None and getattr(launch, "_cf", None) is None
    compiled = []
    monkeypatch.setattr(flyc, "compile", lambda exe, *args: compiled.append(exe) or compile_(exe, *args))
    common._run_compiled(launch, ptr(x), ptr(y), stream)  # 首次编译并执行
    common._run_compiled(launch, ptr(-x), ptr(y), stream)  # 之后复用编译结果
    torch.cuda.synchronize()
    assert compiled == [launch]
    torch.testing.assert_close(y, -x, atol=0, rtol=0)


def test_flydsl_quant_per_tensor_contract():
    from pyhip.ops.moe.flydsl.moe_gemm_splitk import flydsl_absmax, flydsl_quant_per_tensor

    fp8 = torch.float8_e4m3fnuz
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = (torch.randn(8 * 1031, generator=gen, device="cuda") * 3).bfloat16()
    amax = torch.zeros(1, dtype=torch.float32, device="cuda")
    flydsl_absmax()(x, amax)
    assert amax.item() == x.float().abs().max().item()
    # 上游版本不在内部清零 Amax：结果是传入值与本次 absmax 的较大者，由调用方清零。
    stale = torch.full_like(amax, 1e4)
    flydsl_absmax()(x, stale)
    assert stale.item() == 1e4
    q = torch.empty(x.shape, dtype=fp8, device="cuda")
    flydsl_quant_per_tensor(fp8)(x, amax, q)
    expected = (x.float() * (240.0 / amax)).clamp(-240, 240).to(fp8)
    # kernel 乘 rcp 近似值：编码最多差一个 FP8 步长。
    step = (q.view(torch.uint8).int() - expected.view(torch.uint8).int()).abs()
    assert int(step.max()) <= 1 and float((step != 0).float().mean()) < 1e-2
    # amax 为 0（全零输入）时 scale 取 0：输出全 0，不出现 NaN（FNUZ 的 0x80）。
    zeros = torch.zeros_like(x)
    zero_amax = torch.zeros_like(amax)
    flydsl_absmax()(zeros, zero_amax)
    q.fill_(1)
    flydsl_quant_per_tensor(fp8)(zeros, zero_amax, q)
    assert zero_amax.item() == 0 and not q.view(torch.uint8).any()


PYHIP_SYMBOLS = {
    # 模块（相对 pyhip.ops.moe.flydsl）-> pyhip 调用方在这里用到的名字
    "moe_gemm_splitk": ("compile_gemm", "flydsl_absmax", "flydsl_quant_per_tensor", "invert_sorted_ids",
                        "sorted_sum"),
    "moe_gemm_2stage": ("compile_gemm", "compile_moe_gemm1", "compile_moe_gemm2", "compile_moe_reduction",
                        "flydsl_absmax", "flydsl_quant_per_tensor", "sorted_sum"),
    "moe_gemm_2stage.common": ("down_device_config_from_name", "get_device_cache_key", "rocdl_aux",
                               "torch_tensor_to_pointer", "_f32_to_bf16", "_run_compiled"),
    "moe_gemm_2stage.gemm": ("compile_gemm", "gemm_implementation"),
    "moe_gemm_2stage.gemm1": ("compile_moe_gemm1", "_build_moe_gemm1"),
    "moe_gemm_2stage.gemm2": ("compile_moe_gemm2",),
    "moe_gemm_2stage.gemm2_8x1": ("_build_moe_gemm2_8x1", "_build_moe_gemm2_8x1_upstream"),
    "moe_gemm_2stage.gemm2_8x1_compact": ("_build_moe_gemm2_8x1_compact", "allocate_task_buffers",
                                          "_build_moe_gemm2_8x1_compact_upstream",
                                          "allocate_task_buffers_upstream"),
    "moe_gemm_2stage.moe_reduce": ("compile_moe_reduction", "invert_sorted_ids", "sorted_sum",
                                   "sorted_sum_upstream", "_run_compiled"),
}


def test_pyhip_symbols_keep_their_modules():
    import importlib

    from pyhip.codegen.flydsl import helpers
    from pyhip.ops.moe.flydsl import moe_gemm_2stage as package
    from pyhip.ops.moe.flydsl import moe_gemm_splitk as splitk
    from pyhip.ops.moe.flydsl.moe_gemm_2stage import (
        gemm1, gemm1_legacy, gemm2_8x1, gemm2_8x1_compact, gemm2_8x1_compact_legacy, gemm2_8x1_legacy,
        moe_reduce,
    )

    for module, names in PYHIP_SYMBOLS.items():
        loaded = importlib.import_module(f"pyhip.ops.moe.flydsl.{module}")
        missing = [name for name in names if not hasattr(loaded, name)]
        assert not missing, (module, missing)
    assert set(PYHIP_SYMBOLS["moe_gemm_2stage"]) <= set(package.__all__)
    # 包导出的是按设备缓存的 compile_gemm，不是上游 __init__ 里不带缓存的版本。
    assert package.compile_gemm is splitk.compile_gemm is gemm.compile_gemm
    assert callable(gemm.compile_gemm.cache_clear) and callable(gemm.compile_gemm.cache_info)
    # 原名转导出 pyhip（legacy）实现；上游同名实现带 _upstream 后缀。
    assert gemm1._build_moe_gemm1 is gemm1_legacy._build_moe_gemm1
    assert gemm2_8x1._build_moe_gemm2_8x1 is gemm2_8x1_legacy._build_moe_gemm2_8x1
    assert gemm2_8x1._build_moe_gemm2_8x1_upstream is not gemm2_8x1._build_moe_gemm2_8x1
    for name in ("_build_moe_gemm2_8x1_compact", "allocate_task_buffers"):
        assert getattr(gemm2_8x1_compact, name) is getattr(gemm2_8x1_compact_legacy, name)
        assert getattr(gemm2_8x1_compact, f"{name}_upstream") is not getattr(gemm2_8x1_compact, name)
    assert "cu_count" in inspect.signature(gemm2_8x1_compact.allocate_task_buffers_upstream).parameters
    assert splitk.sorted_sum is moe_reduce.sorted_sum is not moe_reduce.sorted_sum_upstream
    assert inspect.signature(moe_reduce.sorted_sum).parameters["row_padding_bytes"].default is None
    assert moe_reduce.compile_moe_reduction(topk=10, model_dim=2560, row_padding_bytes=128) is \
        moe_reduce.sorted_sum(10, 2560, 128)
    assert common.rocdl_aux is helpers.rocdl_aux


def test_kernel_modules_import_without_aiter():
    code = textwrap.dedent("""
        import importlib, importlib.abc, pkgutil, sys

        class NoAiter(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "aiter":
                    raise ImportError(f"moe_gemm_2stage must not import {name}")

        sys.meta_path.insert(0, NoAiter())
        import pyhip.ops.moe.flydsl.moe_gemm_2stage as package
        import pyhip.ops.moe.flydsl.moe_gemm_splitk
        for module in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package.__name__}.{module.name}")
        assert not [name for name in sys.modules if name.split(".")[0] == "aiter"]
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stderr[-4000:]
