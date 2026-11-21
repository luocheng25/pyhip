"""A8W4（mxfp4 权重）MoE 的 M1 复核补充：``test_a8w4_decode.py`` 没覆盖到的入口契约与分支。

参考与容差同 ``test_a8w4_decode.py``（``moe_a8w4_mxfp4.reference_moe`` + M0 的 ``check_output``，不放宽）。

- CPU：``select_config`` / ``decode_max_tokens`` 的查表规则；``prepare_weights`` 其余报错分支，
  以及按专家分块（``chunk``，含不整除的最后一块）不改变结果。
- GPU：``fused_moe`` 的输入检查在启动任何 kernel 之前报错；T=0 不编译、不启动，直接返回；
  配置表 decode 各档的下界（当前 T=2 batch1 gn64、T=17 split-K），其中 TP4 不传 ``output``；
  ``config`` 指定配置表以外的 tile（batch1 gn128/dn128、split-K gn64/dn64），覆盖 delta 选择的其余分支
  （Gate/Up 每 16 行块一个 N repeat 的 r=4 batch1、r=2 split-K，Down 每 CTA 两个 / 一个 64 行 delta 块）；
  ``compile_decode_gemm`` 按设备缓存。
"""

import pytest
import torch

from pyhip.ops.moe.flydsl import a8w4_ptpc as A
from pyhip.ops.moe.flydsl.a8w4_ptpc import moe as entry
from pyhip.testing import moe_a8w4_mxfp4 as X
from pyhip.testing import moe_a8w4_ptpc as M


def _decode_rows(configs=A.CONFIGS):
    """配置表 decode 行的 (下界, 上界, 配置)。"""
    rows, low = [], 1
    for limit, config in configs:
        if config["path"] != "decode":
            break
        rows.append((low, limit, config))
        low = limit + 1
    return rows


# test_a8w4_decode.py 已测 T=1/8/16/32 与 decode 上界；这里补其余各档的下界。
EDGE_TOKENS = sorted({low for low, _, _ in _decode_rows()} - {1, 8, 16, 32, A.decode_max_tokens()})
EDGE_CASES = [(tp, tokens) for tp in (2, 4) for tokens in EDGE_TOKENS]
OVERRIDES = [
    (4, 8, dict(path="decode", alg="batch1", tile_n_gate=128, tile_n_down=128)),
    (2, 32, dict(path="decode", alg="splitk", tile_n_gate=64, tile_n_down=64)),
]


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
            cache[tp] = X.make_mxfp4_weights(tp, seed=20 + tp, device="cuda")
        return cache[tp]

    yield get
    cache.clear()


def _small_raw(experts=2, hidden=256, inter=64, seed=0):
    """CPU 上的小形状自然布局 mxfp4（GPU 用例里默认设备是 cuda，所以显式指定 CPU）。"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w1 = torch.randint(0, 256, (experts, 2 * inter, hidden // 2), generator=gen, device="cpu").to(torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, inter // 2), generator=gen, device="cpu").to(torch.uint8)
    s1 = X.random_codes((experts, 2 * inter, hidden // 32), gen=gen, device="cpu")
    s2 = X.random_codes((experts, hidden, inter // 32), gen=gen, device="cpu")
    return w1, s1, w2, s2


def _record_compiles(monkeypatch):
    compiled = []
    original = entry.compile_decode_gemm

    def compile_decode_gemm(**params):
        compiled.append((params["stage"], params["alg"], params["BLOCK_TILE_SIZE_N"]))
        return original(**params)

    monkeypatch.setattr(entry, "compile_decode_gemm", compile_decode_gemm)
    return compiled


# ---------------------------------------------------------------- CPU

def test_select_config_rules():
    table = ((4, dict(path="decode", alg="batch1", tile_n_gate=64, tile_n_down=64)),
             (64, dict(path="decode", alg="splitk", tile_n_gate=128, tile_n_down=128)),
             (256, dict(path="prefill")))
    assert [A.select_config(t, table).get("alg") for t in (0, 1, 4, 5, 64)] == ["batch1"] * 3 + ["splitk"] * 2
    assert A.select_config(65, table) == {"path": "prefill"}
    assert A.select_config(10 ** 6, table) == {"path": "prefill"}  # 超过所有上界时取最后一行
    assert A.decode_max_tokens(table) == 64
    # 默认表：每个 decode 行的上下界都选到这一行，上界 + 1 是下一行；返回副本，改它不影响配置表。
    rows = _decode_rows()
    assert rows and A.decode_max_tokens() == rows[-1][1]
    for low, high, config in rows:
        assert A.select_config(low) == config and A.select_config(high) == config
    assert A.select_config(A.decode_max_tokens() + 1)["path"] == "prefill"
    picked = A.select_config(1)
    picked["alg"] = "changed"
    assert A.select_config(1)["alg"] != "changed"


def test_prepare_rejects_other_invalid():
    w1, s1, w2, s2 = _small_raw()
    with pytest.raises(ValueError, match="must be"):
        A.prepare_weights(w1[0], s1, w2, s2)
    with pytest.raises(ValueError, match="inconsistent"):
        A.prepare_weights(w1[:, :-1], s1[:, :-1], w2, s2)  # 2I 为奇数
    bad = s1.clone()
    bad[0, 3, 1] = 255
    with pytest.raises(ValueError, match="w1_scale.*255"):
        A.prepare_weights(w1, bad, w2, s2)
    # I=48：形状自洽但 I % 32 != 0。
    w1b, s1b, w2b, s2b = _small_raw(inter=64)
    w1b, s1b = w1b[:, :96], s1b[:, :96]
    w2b, s2b = w2b[..., :24], s2b[..., :1]
    with pytest.raises(ValueError, match="I % 32"):
        A.prepare_weights(w1b, s1b, w2b, s2b)


def test_prepare_chunk_does_not_change_result():
    raw = _small_raw(experts=3, inter=96, seed=3)
    base = A.prepare_weights(*raw)
    for chunk in (1, 2):  # chunk=2 时最后一块只有 1 个专家
        got = A.prepare_weights(*raw, chunk=chunk)
        for field in ("w1", "w1_delta", "w1_factor", "w2", "w2_delta", "w2_factor"):
            assert torch.equal(getattr(got, field), getattr(base, field)), (chunk, field)


# ---------------------------------------------------------------- GPU

def test_fused_moe_rejects_invalid_inputs(gpu, monkeypatch):
    raw = _small_raw()
    prep = A.prepare_weights(*(t.cuda() for t in raw))
    prep_cpu = A.prepare_weights(*raw)
    tokens, hidden_size = 4, 256
    hidden = torch.randn(tokens, hidden_size).to(torch.bfloat16)
    ids = torch.zeros(tokens, 2, dtype=torch.int32)
    weight = torch.full((tokens, 2), 0.5)
    sentinel = torch.full_like(hidden, float("nan"))
    good = dict(hidden_states=hidden, weights=prep, topk_weight=weight, topk_ids=ids, output=sentinel)
    cases = [
        (TypeError, "prepare_weights", dict(weights=(prep.w1, prep.w2))),
        (ValueError, "BF16", dict(hidden_states=hidden.half())),
        (ValueError, "BF16", dict(hidden_states=hidden[0])),
        (ValueError, "BF16", dict(hidden_states=hidden.cpu())),
        (ValueError, "mismatch", dict(hidden_states=torch.zeros(tokens, 512, dtype=torch.bfloat16),
                                      output=None)),
        (ValueError, "mismatch", dict(weights=prep_cpu)),
        (ValueError, "topk", dict(topk_ids=ids.long())),
        (ValueError, "topk", dict(topk_weight=weight.to(torch.bfloat16))),
        (ValueError, "topk", dict(topk_ids=ids[:-1], topk_weight=weight[:-1])),
        (ValueError, "topk", dict(topk_weight=weight[:, :1])),
        (ValueError, "topk", dict(topk_ids=ids[0], topk_weight=weight[0])),
        (ValueError, "contiguous", dict(hidden_states=torch.zeros(tokens, 2 * hidden_size,
                                                                  dtype=torch.bfloat16)[:, ::2])),
        (ValueError, "contiguous", dict(topk_ids=torch.zeros(2, tokens, dtype=torch.int32).t())),
        (ValueError, "contiguous", dict(output=torch.empty(hidden_size, tokens, dtype=torch.bfloat16).t())),
        (ValueError, "output", dict(output=torch.empty(tokens, hidden_size))),
        (ValueError, "output", dict(output=torch.empty(tokens + 1, hidden_size, dtype=torch.bfloat16))),
    ]
    compiled = _record_compiles(monkeypatch)
    for error, match, change in cases:
        with pytest.raises(error, match=match):
            A.fused_moe(**{**good, **change})
    assert compiled == [], compiled
    assert bool(sentinel.isnan().all()), "inputs must be checked before any launch"


def test_empty_input_returns_without_launch(gpu, monkeypatch):
    prep = A.prepare_weights(*(t.cuda() for t in _small_raw()))
    hidden = torch.empty(0, 256, dtype=torch.bfloat16)
    ids = torch.empty(0, 2, dtype=torch.int32)
    weight = torch.empty(0, 2)
    compiled = _record_compiles(monkeypatch)
    out = A.fused_moe(hidden, prep, weight, ids)
    assert out.shape == (0, 256) and out.dtype == torch.bfloat16 and out.is_cuda
    given = torch.empty_like(hidden)
    assert A.fused_moe(hidden, prep, weight, ids, output=given) is given
    assert compiled == [], compiled


@pytest.mark.parametrize("tp, tokens", EDGE_CASES, ids=[f"tp{tp}-t{t}" for tp, t in EDGE_CASES])
def test_decode_tier_lower_edges(gpu, shard, monkeypatch, tp, tokens):
    raw, prep = shard(tp)
    config = A.select_config(tokens)
    assert config["path"] == "decode" and A.select_config(tokens - 1) != config, (tokens, config)
    hidden, weight, ids = M.make_inputs(tokens, seed=1000 + tokens)
    compiled = _record_compiles(monkeypatch)
    if tp == 2:
        output = torch.full_like(hidden, float("nan"))
        assert A.fused_moe(hidden, prep, weight, ids, output=output) is output
    else:  # 不传 output：新建 BF16 [T, H]
        output = A.fused_moe(hidden, prep, weight, ids)
        assert output.shape == hidden.shape and output.dtype == torch.bfloat16
    assert compiled == [("gateup", config["alg"], config["tile_n_gate"]),
                        ("down", config["alg"], config["tile_n_down"])], compiled
    M.check_output(X.reference_moe(hidden, raw, weight, ids), output, f"TP{tp} T={tokens} {config}")


@pytest.mark.parametrize("tp, tokens, config", OVERRIDES,
                         ids=[f"tp{tp}-t{t}-{c['alg']}-gn{c['tile_n_gate']}-dn{c['tile_n_down']}"
                              for tp, t, c in OVERRIDES])
def test_config_override_tiles(gpu, shard, monkeypatch, tp, tokens, config):
    raw, prep = shard(tp)
    hidden, weight, ids = M.make_inputs(tokens, seed=2000 + tokens)
    output = torch.full_like(hidden, float("nan"))
    compiled = _record_compiles(monkeypatch)
    assert A.fused_moe(hidden, prep, weight, ids, output=output, config=config) is output
    assert compiled == [("gateup", config["alg"], config["tile_n_gate"]),
                        ("down", config["alg"], config["tile_n_down"])], compiled
    M.check_output(X.reference_moe(hidden, raw, weight, ids), output, f"TP{tp} T={tokens} {config}")


def test_compile_decode_gemm_cached_per_device(gpu, monkeypatch):
    from pyhip.ops.moe.flydsl.a8w4_ptpc import decode as D

    params = dict(stage="down", N=M.H, K=192, TOPK=M.TOPK, BLOCK_TILE_SIZE_M=16,
                  BLOCK_TILE_SIZE_N=64, alg="batch1", E=M.E)
    first = D.compile_decode_gemm(**params)
    assert D.compile_decode_gemm(**params) is first
    monkeypatch.setattr(D, "get_device_cache_key", lambda: ("another device",))
    assert D.compile_decode_gemm(**params) is not first
    with pytest.raises(AssertionError):
        D.compile_decode_gemm(**{**params, "BLOCK_TILE_SIZE_M": 32})
