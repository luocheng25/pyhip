"""A8W4（mxfp4 权重）MoE 的 M3 复核补充：M3 新增的 prefill 逻辑的专门用例（已有用例只在统计上覆盖）。

参考与容差同 ``test_a8w4_prefill.py``（``moe_a8w4_mxfp4.reference_moe(act_quant=True)`` + M0 的 ``check_output``，
不放宽），并同样确认输出离 BF16 激活的参考远 10 倍以上。"M2 路径"指同一配置改成 ``half_m=False``、
``prefetch_gate=prefetch_down=1``（量化仍是 FlyDSL kernel），M3 的分派与预取只改变计算哪些行、何时读权重，
不改变每个输出元素的运算顺序，所以两者逐位相同。

- half_m 块内分派（配置表 prefill 档默认打开）：排序块第 33 行（``sorted_ids[32]``）是补齐行时，Gate/Up 与 Down
  只算前 32 行。路由按指定行数构造（``_routing``）：专家行数 31、32（交界，只算前 32 行）、33（交界另一侧，
  整块）、64、65、96、97 和 1，其余专家各几行，所以同一次调用里两种块都有。直接按 ``_fused_moe_prefill`` 的
  顺序启动两级 kernel，``mid`` 与 ``routes`` 预填 NaN 位型 ``SENTINEL``：
  * Down：只算 32 行的块不写第 33–64 行（保持 ``SENTINEL``），整块写满 64 行（补齐行也写），据此逐块断言
    走了哪条路径，并与按 ``sorted_ids`` 推出的预期一致；
  * Gate/Up：两条路径都只按 token 写有效行，从输出看不出走了哪条，靠上面的构造保证两种块都出现；断言所有
    有效行都写了（没有 ``SENTINEL``），且 ``mid``、``routes`` 的有效行与 M2 路径逐位相同；
  * 端到端 ``fused_moe``：与 M2 路径逐位相同，并对 FP32 参考满足容差。
  Qwen3.8 形状 TP2（Down K=320，5 个 K core）、TP4（K=192，3 个 K core）各一次，T=256。
- 权重预取 2 步在 K tile 少的形状上（E=8、H=512、topk=2、T=128，``config=`` 指定 prefill 档，同样的分派检查）：
  Down 的 K=64（只有 1 个 K core，预取深度取 min(2, 1)）与 K=128（2 个 K core，下一 N tile 的两个 K core
  全部在本 tile 内发出、跨 N 循环携带）；Gate/Up 的 BK256（共 2 个 K tile，只在序言预取）与 BK128（4 个）。
  已有用例覆盖、这里不重复的：TP4 的 Down K=192（3 个 K core）由 ``test_a8w4_prefill.py`` 的
  ``test_prefill_matches_reference[tp4-*]`` 覆盖；32 行 tile 的 Gate/Up 在 2 个 K tile 上预取 2 步也由
  ``test_a8w4_prefill_edges.py::test_prefill_min_hidden_max_inter`` 覆盖（其路由里专家 1 有 91 行，末块 27 行）。
- ``quant="aiter"``（M2 的 Aiter per-token 量化，只经 ``config=`` 可达）：TP2 T=256 对参考满足容差。
- FlyDSL 按行量化（``compile_prefill_quant``，生产经 ``moe._quantize_rows`` 调用）与参考 ``quantize_rows`` 的
  q、scale 逐位相同：宽度 2560（输入，每行 64 个 lane、5 步）、320 / 192（TP2 / TP4 的 Gate/Up 输出宽度，
  8 个 lane）、256 / 128（32 / 16 个 lane）；行数 1、10、33、1027（不是每块行数 4 或 32 的倍数）；数据含
  +0 / −0 全零行（amax=0，scale=0）、±BF16 最大值与 1.0 混合的行（小值下溢为 0）、amax=240（scale 恰为 1）
  且含 FP8 中点与次正规数的行（就近偶数舍入）、每行量级 2^-20–2^20 的随机行。直接启动 kernel 时输出缓冲
  比 ``rows`` 长：最后一块越界的行不写 q、不写 scale。
"""

import pytest
import torch

from pyhip.ops.moe.flydsl import a8w4_ptpc as A
from pyhip.ops.moe.flydsl.a8w4_ptpc import moe as entry
from pyhip.ops.moe.flydsl.a8w4_ptpc import prefill as P
from pyhip.testing import moe_a8w4_mxfp4 as X
from pyhip.testing import moe_a8w4_ptpc as M

PREFILL = A.select_config(A.decode_max_tokens() + 1)
BLOCK = PREFILL["tile_m"]  # 64
HALF = BLOCK // 2
SENTINEL = 0x7FAB  # BF16 NaN 位型：kernel 没写的位置保持这个值
# 专家 -> 路由行数（其余专家只有几行）：32 / 33 是只算前 32 行与整块的交界。
QWEN_COUNTS = {0: 32, 1: 33, 2: 31, 3: 64, 4: 65, 5: 96, 6: 97, 7: 1}
SMALL_COUNTS = {0: 32, 1: 33, 2: 64, 3: 97}
QUANT_COLS = {2560: "h2560-input", 320: "i320-tp2-mid", 192: "i192-tp4-mid", 256: "i256", 128: "i128"}
QUANT_ROWS = (1, 10, 33, 1027)


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


def _m2_path(config):
    return dict(config, half_m=False, prefetch_gate=1, prefetch_down=1)


def _check(hidden, raw, weight, ids, output, msg):
    _, logits_diff, _ = M.check_output(X.reference_moe(hidden, raw, weight, ids, act_quant=True), output, msg)
    _, bf16_diff, _ = M.output_errors(X.reference_moe(hidden, raw, weight, ids, act_quant=False), output)
    assert bf16_diff > 10 * logits_diff, (msg, logits_diff, bf16_diff)


def _routing(tokens, experts, topk, counts, *, seed):
    """int32 ``topk_ids``：专家 e 恰有 ``counts[e]`` 条路由（在不同 token 上），每个 token 的其余槽位从其它
    专家里随机取（同一 token 内不重复），最后打乱每个 token 的槽位顺序。"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.full((tokens, topk), -1, dtype=torch.int64, device="cpu")
    used = torch.zeros(tokens, dtype=torch.int64, device="cpu")
    for expert, rows in sorted(counts.items(), key=lambda item: -item[1]):
        chosen = torch.argsort(used + torch.rand(tokens, generator=gen, device="cpu"))[:rows]  # 空槽多的先
        assert int(used[chosen].max()) < topk
        ids[chosen, used[chosen]] = expert
        used[chosen] += 1
    others = torch.tensor([e for e in range(experts) if e not in counts], device="cpu")
    for token in range(tokens):
        free = topk - int(used[token])
        ids[token, topk - free:] = others[torch.randperm(len(others), generator=gen, device="cpu")[:free]]
    ids = torch.gather(ids, 1, torch.argsort(torch.rand((tokens, topk), generator=gen, device="cpu"), dim=1))
    assert torch.bincount(ids.flatten(), minlength=experts)[list(counts)].tolist() == list(counts.values())
    return ids.to(torch.int32).cuda().contiguous()


def _filled(shape):
    return torch.full(shape, SENTINEL, dtype=torch.int16).view(torch.bfloat16)


def _untouched(t):
    return t.view(torch.int16) == SENTINEL


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _stages(prep, hidden, weight, ids, config):
    """按 ``moe._fused_moe_prefill`` 的顺序直接启动排序、量化与两级 kernel（``mid``、``routes`` 预填
    ``SENTINEL``），返回 ``(sorted_ids, sorted_expert_ids, 有效排序行数, mid, routes)``。"""
    from pyhip.ops.moe._aiter_compat import moe_sorting
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.common import _run_compiled

    tokens, hidden_size = hidden.shape
    topk, inter, experts = ids.shape[1], prep.inter_padded, prep.experts
    padding, stream = config["padding"], torch.cuda.current_stream()
    routing = moe_sorting(ids, weight, experts, hidden_size, hidden.dtype, BLOCK, accumulate=False)[:4]
    tasks = routing[2].numel()
    common = dict(TOPK=topk, BLOCK_TILE_SIZE_M=BLOCK, E=experts, half_m=config["half_m"])
    gateup = P.compile_prefill_gemm(stage="gateup", N=2 * inter, K=hidden_size, BLOCK_TILE_SIZE_N=config["tile_n_gate"],
                                    tile_k=config["tile_k_gate"], prefetch=config["prefetch_gate"], **common)
    down = P.compile_prefill_gemm(stage="down", N=hidden_size, K=inter, BLOCK_TILE_SIZE_N=config["tile_n_down"],
                                  padding=padding, prefetch=config["prefetch_down"], **common)
    x, x_scale = entry._quantize_rows(hidden, stream)
    mid = _filled((tokens, topk, inter))
    _run_compiled(gateup, *map(entry._ptr, (x, prep.w1, mid, *routing, prep.w1_factor, prep.w1_pdelta, x_scale)),
                  tokens, tasks, stream)
    mid_q, mid_scale = entry._quantize_rows(mid.view(tokens * topk, inter), stream)
    routes = _filled((tasks * BLOCK, hidden_size + padding // 2))
    _run_compiled(down, *map(entry._ptr, (mid_q, prep.w2, routes, *routing, prep.w2_factor, prep.w2_pdelta,
                                          mid_scale)), tokens, tasks, stream)
    torch.cuda.synchronize()
    return routing[0], routing[2], int(routing[3][0]), mid, routes[:, :hidden_size]


def _expected_kinds(rows):
    """一个专家 ``rows`` 行路由的各排序块：后半全是补齐的只算前 32 行（"half"），否则整块（"full"）。"""
    tail = rows % BLOCK
    return ["full"] * (rows // BLOCK) + ([] if tail == 0 else ["half" if tail <= HALF else "full"])


def _check_dispatch(prep, raw, hidden, weight, ids, config, msg):
    """断言 half_m 的逐块路径（Down 可观测）、有效行与 M2 路径逐位相同，以及端到端结果；返回各块的路径。"""
    tokens, experts = hidden.shape[0], prep.experts
    sorted_ids, block_experts, valid_rows, mid, routes = _stages(prep, hidden, weight, ids, config)
    _, _, m2_rows, m2_mid, m2_routes = _stages(prep, hidden, weight, ids, _m2_path(config))
    assert valid_rows == m2_rows and valid_rows % BLOCK == 0
    blocks = valid_rows // BLOCK
    valid = (sorted_ids[:valid_rows].view(blocks, BLOCK) & 0xFFFFFF) < tokens
    half = ~valid[:, HALF]  # kernel 的判据：第 33 行是补齐行
    # 排序把补齐放在每个专家末块的末尾：第 33 行是补齐时，后 32 行全是补齐。
    assert not bool((valid[:, HALF:] & half[:, None]).any())
    counts = torch.bincount(ids.flatten().long(), minlength=experts).tolist()
    owner = block_experts[:blocks].tolist()
    kinds = ["half" if h else "full" for h in half.tolist()]
    for expert in range(experts):
        got = [kind for kind, e in zip(kinds, owner) if e == expert]
        assert got == _expected_kinds(counts[expert]), (msg, expert, counts[expert], got)
    assert {"half", "full"} <= set(kinds), (msg, kinds)

    # Down：只算 32 行的块不写后 32 行；整块写满 64 行（含补齐行）；M2 路径全部整块。
    tiles = routes[:valid_rows].view(blocks, BLOCK, -1)
    assert bool(_untouched(tiles[half, HALF:]).all()), msg
    assert not bool(_untouched(tiles[~half]).any()), msg
    assert not bool(_untouched(m2_routes[:valid_rows]).any()), msg
    # 所有有效行都写了，且与 M2 路径逐位相同。
    rows = valid.flatten()
    assert not bool(_untouched(mid).any()) and _bits_equal(mid, m2_mid), msg
    assert not bool(_untouched(routes[:valid_rows][rows]).any()), msg
    assert _bits_equal(routes[:valid_rows][rows], m2_routes[:valid_rows][rows]), msg

    output = torch.full_like(hidden, float("nan"))
    assert A.fused_moe(hidden, prep, weight, ids, output=output, config=config) is output
    m2_output = A.fused_moe(hidden, prep, weight, ids, config=_m2_path(config))
    assert _bits_equal(output, m2_output), msg
    _check(hidden, raw, weight, ids, output, msg)
    return kinds


# ---------------------------------------------------------------- half_m 分派（Qwen3.8 形状）

@pytest.fixture(scope="module")
def shard():
    """``shard(tp) -> (raw, prepared)``，每个 TP 只生成一次。"""
    cache = {}

    def get(tp):
        if tp not in cache:
            cache[tp] = X.make_mxfp4_weights(tp, seed=50 + tp, device="cuda")
        return cache[tp]

    yield get
    cache.clear()


@pytest.mark.parametrize("tp", (2, 4), ids=("tp2", "tp4"))
def test_half_m_dispatch_boundary(gpu, shard, tp):
    tokens = 256
    assert A.select_config(tokens) == PREFILL and PREFILL["half_m"]
    raw, prep = shard(tp)
    hidden, weight, _ = M.make_inputs(tokens, seed=800 + tp)
    ids = _routing(tokens, M.E, M.TOPK, QWEN_COUNTS, seed=810 + tp)
    kinds = _check_dispatch(prep, raw, hidden, weight, ids, PREFILL, f"TP{tp} T={tokens} {PREFILL}")
    assert kinds.count("full") == 1 + 1 + 1 + 1 + 2  # 33、64、65（首块）、96（首块）、97 两块


def test_quant_aiter_option(gpu, shard):
    tokens, config = 256, dict(PREFILL, quant="aiter")
    raw, prep = shard(2)
    hidden, weight, ids = M.make_inputs(tokens, seed=830)
    output = torch.full_like(hidden, float("nan"))
    assert A.fused_moe(hidden, prep, weight, ids, output=output, config=config) is output
    _check(hidden, raw, weight, ids, output, f"TP2 T={tokens} {config}")


# ---------------------------------------------------------------- 预取 2 步：K tile 少的形状

def _small_raw(experts, hidden, inter, *, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w1 = torch.randint(0, 256, (experts, 2 * inter, hidden // 2), generator=gen, device="cuda").to(torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, inter // 2), generator=gen, device="cuda").to(torch.uint8)
    s1 = X.random_codes((experts, 2 * inter, hidden // 32), gen=gen, device="cuda")
    s2 = X.random_codes((experts, hidden, inter // 32), gen=gen, device="cuda")
    return w1, s1, w2, s2


SMALL_CASES = [(64, 256), (128, 128)]  # (I = Down 的 K, Gate/Up 的 BK)


@pytest.mark.parametrize("inter, tile_k", SMALL_CASES, ids=[f"i{i}-bk{k}" for i, k in SMALL_CASES])
def test_prefetch_two_with_few_k_tiles(gpu, inter, tile_k):
    experts, hidden_size, topk, tokens = 8, 512, 2, 128
    config = dict(PREFILL, tile_k_gate=tile_k)
    assert config["prefetch_gate"] == 2 and config["prefetch_down"] == 2
    raw = _small_raw(experts, hidden_size, inter, seed=840 + inter)
    prep = A.prepare_weights(*raw)
    assert prep.inter_padded == inter
    gen = torch.Generator(device="cuda").manual_seed(850 + inter)
    hidden = torch.randn((tokens, hidden_size), generator=gen, device="cuda").to(torch.bfloat16)
    weight = torch.randn((tokens, topk), generator=gen, device="cuda").softmax(dim=-1).contiguous()
    ids = _routing(tokens, experts, topk, SMALL_COUNTS, seed=860 + inter)
    _check_dispatch(prep, raw, hidden, weight, ids, config, f"E={experts} H={hidden_size} I={inter} {config}")


# ---------------------------------------------------------------- FlyDSL 按行量化

def _quant_inputs(rows, cols, *, seed):
    """随机行（每行量级 2^-20–2^20）；行数 ≥ 5 时前 5 行换成特殊行。"""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    magnitude = torch.exp2(torch.randint(-20, 21, (rows, 1), generator=gen, device="cuda").float())
    x = (torch.randn((rows, cols), generator=gen, device="cuda") * magnitude).to(torch.bfloat16)
    if rows >= 5:
        big = torch.finfo(torch.bfloat16).max
        x[0] = 0.0
        x[1] = -0.0
        x[2] = 1.0
        x[2, ::3], x[2, 1::3] = big, -big
        # amax=240，scale 恰为 1：FP8 中点按就近偶数舍入，次正规数 k·2^-10 与下溢。
        ties = [1.0625, 1.1875, -1.0625, -1.1875, 1.5 * 2 ** -10, 0.5 * 2 ** -10, 2 ** -11, -3 * 2 ** -11]
        x[3] = 0.0
        x[3, 0] = 240.0
        x[3, 1:1 + len(ties)] = torch.tensor(ties, dtype=torch.bfloat16)
        x[4] = -x[3]
    return x


@pytest.mark.parametrize("cols", list(QUANT_COLS), ids=list(QUANT_COLS.values()))
def test_quant_rows_matches_reference_bitwise(gpu, cols):
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.common import _run_compiled

    stream = torch.cuda.current_stream()
    launch, rows_per_block = P.compile_prefill_quant(cols)
    assert rows_per_block == 256 // P._quant_lanes_per_row(cols)
    for rows in QUANT_ROWS:
        x = _quant_inputs(rows, cols, seed=rows * 10007 + cols)
        ref_q, ref_scale = M.quantize_rows(x)
        msg = f"cols={cols} rows={rows}"
        q, scale = entry._quantize_rows(x, stream)
        torch.cuda.synchronize()
        assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8)), msg
        assert torch.equal(scale.view(torch.int32), ref_scale.view(-1).view(torch.int32)), msg
        if rows >= 5:
            assert ref_scale[:2].eq(0).all() and ref_scale[3:5].eq(1).all(), msg  # 构造确实命中这些情况
        # 直接启动、输出缓冲比 rows 长：最后一块里越界的行不写。
        blocks = -(-rows // rows_per_block)
        extra = blocks * rows_per_block - rows + 1
        q_buf = torch.full((rows + extra, cols), 0x5A, dtype=torch.uint8)
        scale_buf = torch.full((rows + extra,), -7.0, dtype=torch.float32)
        _run_compiled(launch, entry._ptr(x), entry._ptr(q_buf), entry._ptr(scale_buf), rows, blocks, stream)
        torch.cuda.synchronize()
        assert torch.equal(q_buf[:rows], ref_q.view(torch.uint8)), msg
        assert torch.equal(scale_buf[:rows], ref_scale.view(-1)), msg
        assert bool((q_buf[rows:] == 0x5A).all()) and bool((scale_buf[rows:] == -7.0).all()), msg
