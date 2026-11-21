"""A8W4 decode 的 M4 复核补充：alg=gemv（T≤16 的 decode，VALU 上 f32 乘加、不发 MFMA）在已有用例之外缺的覆盖。

参考与容差同 ``test_a8w4_decode.py``（``moe_a8w4_mxfp4.reference_moe`` + M0 的 ``check_output``，不放宽）。
已有用例覆盖了配置表的 gemv 行（T=1 gn32 / dn64、T=2..16 gn128 / dn64，TP2 / TP4，含 CUDA graph），也就覆盖了
Down 序言不整轮（K/2 = 160 / 96 个 dword 分给 64 个 lane）与奇数个 K tile 的收尾，这里不重复。

- ``config`` 指定配置表以外的 gemv tile：Gate/Up gn64（每个 CTA 32 对 gate / up 通道，r=2 的 delta 选择）、
  Down dn128（每个 CTA 两个 64 行 delta 块，写出两轮）。
- 同一输入下 gemv 与 batch1（M1–M3 配置表的 decode 行，经 ``config`` 指定）都满足容差。每个 token 只保留一条
  路由（权重 1，其余 0）时，Down 的每个输出元素只有一次非零原子加，结果与加法顺序无关，两者只差 fp32 累加顺序：
  逐元素差不超过 1 个 BF16 ulp（按元素自身的量级，接近 0 的元素按该 token RMS 的量级），且 99% 以上逐位相同。
  不比较带全部路由的输出：Down 的 BF16 原子加顺序不定，同一算法连跑两次也有约一半元素不同。
"""

import pytest
import torch

from pyhip.ops.moe.flydsl import a8w4_ptpc as A
from pyhip.ops.moe.flydsl.a8w4_ptpc import moe as entry
from pyhip.testing import moe_a8w4_mxfp4 as X
from pyhip.testing import moe_a8w4_ptpc as M

GEMV_TILES = dict(path="decode", alg="gemv", tile_n_gate=64, tile_n_down=128)
TILE_CASES = [(2, 16), (4, 5)]
VERSUS_CASES = [(2, 16), (4, 1)]


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
            cache[tp] = X.make_mxfp4_weights(tp, seed=30 + tp, device="cuda")
        return cache[tp]

    yield get
    cache.clear()


def _record_compiles(monkeypatch):
    compiled = []
    original = entry.compile_decode_gemm

    def compile_decode_gemm(**params):
        compiled.append((params["stage"], params["alg"], params["BLOCK_TILE_SIZE_N"]))
        return original(**params)

    monkeypatch.setattr(entry, "compile_decode_gemm", compile_decode_gemm)
    return compiled


def _stages(config):
    return [("gateup", config["alg"], config["tile_n_gate"]), ("down", config["alg"], config["tile_n_down"])]


def _ordered(x):
    """BF16 位型映射到单调整数轴（-0 与 +0 都是 0），相邻两个 BF16 值差 1。"""
    bits = x.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


@pytest.mark.parametrize("tp, tokens", TILE_CASES, ids=[f"tp{tp}-t{t}" for tp, t in TILE_CASES])
def test_gemv_config_tiles(gpu, shard, monkeypatch, tp, tokens):
    raw, prep = shard(tp)
    hidden, weight, ids = M.make_inputs(tokens, seed=3000 + tokens)
    output = torch.full_like(hidden, float("nan"))
    compiled = _record_compiles(monkeypatch)
    assert A.fused_moe(hidden, prep, weight, ids, output=output, config=GEMV_TILES) is output
    assert compiled == _stages(GEMV_TILES), compiled
    M.check_output(X.reference_moe(hidden, raw, weight, ids), output, f"TP{tp} T={tokens} {GEMV_TILES}")


@pytest.mark.parametrize("tp, tokens", VERSUS_CASES, ids=[f"tp{tp}-t{t}" for tp, t in VERSUS_CASES])
def test_gemv_matches_batch1(gpu, shard, monkeypatch, tp, tokens):
    raw, prep = shard(tp)
    gemv = A.select_config(tokens)
    assert gemv["alg"] == "gemv", gemv
    # M1–M3 配置表的 decode 行：T=1 batch1 gn32，2..16 batch1 gn64，Down 都是 dn64。
    batch1 = dict(gemv, alg="batch1", tile_n_gate=32 if tokens == 1 else 64, tile_n_down=64)
    hidden, weight, ids = M.make_inputs(tokens, seed=4000 + tokens)
    single = torch.zeros_like(weight)
    single[torch.arange(tokens), torch.arange(tokens) % ids.shape[1]] = 1.0
    compiled = _record_compiles(monkeypatch)
    reference = X.reference_moe(hidden, raw, weight, ids)
    outputs = {}
    for config in (gemv, batch1):
        for name, route_weight in (("all", weight), ("single", single)):
            compiled.clear()
            output = torch.full_like(hidden, float("nan"))
            assert A.fused_moe(hidden, prep, route_weight, ids, output=output, config=config) is output
            assert compiled == _stages(config), compiled
            outputs[config["alg"], name] = output
        M.check_output(reference, outputs[config["alg"], "all"], f"TP{tp} T={tokens} {config}")

    got, base = outputs["gemv", "single"], outputs["batch1", "single"]
    assert bool(torch.isfinite(got).all()) and bool(torch.isfinite(base).all())
    ulps = (_ordered(got) - _ordered(base)).abs()
    diff = (got.float() - base.float()).abs()
    rms = base.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    near_zero = diff <= rms * 2.0 ** -8
    assert bool(((ulps <= 1) | near_zero).all()), (
        f"TP{tp} T={tokens}: {int(((ulps > 1) & ~near_zero).sum())} elements differ by more than 1 BF16 ulp, "
        f"max {int(ulps.max())} ulp, max |diff| / token RMS {float((diff / rms).max()):.2e}")
    same = float((ulps == 0).float().mean())
    assert same >= 0.99, f"TP{tp} T={tokens}: only {same:.4f} of the elements are bitwise equal"
