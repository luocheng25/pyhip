"""A8W4 MoE bench（任务冻结后受保护）：pyhip ``fused_moe_a8w4`` 对照 Aiter FP8 ptpc。

34 个 case：TP2（I=320）与 TP4（I=160，prepare 补零到 192；Aiter 用同一份 BF16 权重补零到 192）
× M 按 2 的幂细扫 {1, 2, 4, …, 32768}，另加 M=48（E513 下 M=33–62 走排序块 32 的流式段，
2 的幂跳过了它），名为 tp2_m1 … tp4_m32768、tp2_m48、tp4_m48。H=2560，E=513（512 routed +
最后一个融合 shared，每个 token 的最后槽位必选），topk=11，SiLU，BF16 进出；权重与输入由
``pyhip.testing.moe_a8w4`` 生成（BF16 权重逐 expert 生成），种子取环境变量 MF_SEED（默认 0）。

协议：两种实现在同一进程、同一协议下交替计时。每次调用前写 512 MiB flush buffer，再用
``torch.cuda._sleep`` 让 GPU 空转（至少 200 µs，且不短于 warmup 后实测的两种实现 host 下发一次调用
耗时较大者的 2 倍），使 host 提前排队，event 只计完整调用的 GPU 时间；不设默认设备（它的
TorchFunctionMode 会给每个 torch 调用加 host 开销）。warmup 3 次，30 次取中位，4 组输入轮换。
被计时的 A8W4 输出按 ``moe_a8w4.check`` 校验，任一组不过即退出非零。Aiter 对照只用
``aiter/configs/model_configs/qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv``（复制到临时目录后经
AITER_CONFIG_FMOE 指定），每个 case 断言 dispatcher 用的就是表中那一行。

输出：``MF_METRIC case=<名> value=<us> unit=us flops=<F> bytes=<B>``（A8W4 中位数）；
``MF_REF case=<名> impl=aiter_fp8_ptpc value=<us> flops=<F> bytes=<B> ratio=<A8W4/aiter> kernels=<名>``；
flops 按补零后的 I 计 6·M·topk·I·H，bytes 是各自的最小数据量（被选中 expert 的权重与 scale + 输入读、输出写）；
``INFO`` 行给所走路径、校验结果、带宽与算力利用率（MI308X：HBM 5.3 TB/s，FP8 稠密约 606 TFLOPS）。
用法：``PYTHONPATH=<repo>/src python benchmarks/moe/bench_moe_a8w4.py [--cases tp2_m1 ...]``
"""

import argparse
import importlib.util
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time

HIDDEN, EXPERTS, TOPK = 2560, 513, 11
TOKENS = tuple(1 << k for k in range(16))  # 1, 2, 4, …, 32768
EXTRA_TOKENS = (48,)  # 排序块 32 的小 M 流式段；追加在末尾，已有 case 的输入种子（按序号）不变
INTER = {"tp2": (320, 320), "tp4": (160, 192)}  # (真实 I, 计算用的补零 I)
CASES = [f"{tp}_m{m}" for tp in INTER for m in TOKENS] + [f"{tp}_m{m}" for tp in INTER for m in EXTRA_TOKENS]
HBM_BYTES_PER_S, FP8_FLOPS = 5.3e12, 606e12
WARMUP, ITERS, GROUPS = 3, 30, 4
FLUSH_BYTES, IDLE_US = 512 << 20, 200
TABLE = "qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv"

os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"  # 测当前源码，不复用旧磁盘编译产物。
os.environ["AITER_ONLINE_TUNE"] = "0"
_table_src = Path(importlib.util.find_spec("aiter").submodule_search_locations[0]) / "configs/model_configs" / TABLE
_table = Path(tempfile.mkdtemp(prefix="bench_moe_a8w4_")) / TABLE  # 副本：Aiter 的去重检查可能改写表文件
shutil.copyfile(_table_src, _table)
os.environ["AITER_CONFIG_FMOE"] = str(_table)

import torch  # noqa: E402


def _aiter_row(fm, tokens, inter):
    from aiter import ActivationType, QuantType, dtypes

    key = (fm.get_gfx_runtime(), fm.get_cu_num(), fm.get_padded_M(tokens), HIDDEN, inter, EXPERTS, TOPK,
           str(ActivationType.Silu), str(dtypes.bf16), str(dtypes.fp8), str(dtypes.fp8),
           str(QuantType.per_Token), 1, 0)
    fm.get_2stage_cfgs(fm.get_padded_M(tokens), HIDDEN, inter, EXPERTS, TOPK, dtypes.bf16, dtypes.fp8,
                       dtypes.fp8, QuantType.per_Token, True, ActivationType.Silu, False, 0, 0)
    row = fm.cfg_2stages[0].get(key)
    if row is None or not 0 < float(row["us"]) < math.inf:
        raise RuntimeError(f"{TABLE} has no valid row for {key}")
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


def _traffic(ids, per_expert):
    """最小数据量：被选中 expert 的全部权重（含 scale）+ 输入读与输出写（字节）。"""
    return ids.unique().numel() * per_expert + ids.shape[0] * HIDDEN * 2 * 2


def _bytes_per_expert(weights):
    values = weights.values() if isinstance(weights, dict) else vars(weights).values()
    return sum(t.numel() * t.element_size() for t in values if isinstance(t, torch.Tensor)) // EXPERTS


def run_case(name, weights, cycles_per_us, flush, seed):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType

    from pyhip.ops.moe.flydsl import moe_a8w4
    from pyhip.testing import moe_a8w4 as oracle

    tp, tokens = name.split("_m")[0], int(name.split("_m")[1])
    inter, inter_pad = INTER[tp]
    w, prepared, fp8_w = weights
    index = CASES.index(name)
    groups = [oracle.make_inputs(tokens, HIDDEN, EXPERTS, TOPK, seed=seed * 1000 + 100 + index * GROUPS + g)
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

    row = _aiter_row(fm, tokens, inter_pad)
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
        ours(i % GROUPS)
        theirs(i % GROUPS)
    # 空转要盖住 host 下发一次调用的时间，否则 host 开销会落进计时区间。
    host = {impl: _host_us(fn, cycles_per_us) for impl, fn in (("ours", ours), ("theirs", theirs))}
    idle_us = max(IDLE_US, 2 * max(host.values()))
    sleep_cycles = int(idle_us * cycles_per_us)

    events = {impl: [] for impl in ("ours", "theirs")}
    for i in range(ITERS):
        for impl, fn in (("ours", ours), ("theirs", theirs)):
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
        if not bool(torch.isfinite(theirs_out[g]).all()):
            print(f"FAIL case={name} group={g} aiter output is not finite", flush=True)
            sys.exit(1)
        if worst is None or result["err_ratio"] > worst["err_ratio"]:
            worst = result
        del ref

    flops = 6 * tokens * TOPK * inter_pad * HIDDEN
    bytes_ours = statistics.mean(_traffic(ids, _bytes_per_expert(prepared)) for _, _, ids in groups)
    bytes_theirs = statistics.mean(_traffic(ids, _bytes_per_expert(fp8_w)) for _, _, ids in groups)

    def util(t_us, nbytes):
        bw, fl = nbytes / (t_us * 1e-6), flops / (t_us * 1e-6)
        return (f"{bw / 1e12:.2f}TB/s({bw / HBM_BYTES_PER_S * 100:.1f}%) "
                f"{fl / 1e12:.1f}TFLOPS({fl / FP8_FLOPS * 100:.1f}%)")

    print(f"MF_METRIC case={name} value={us['ours']:.2f} unit=us flops={flops} bytes={bytes_ours:.0f}", flush=True)
    print(f"MF_REF case={name} impl=aiter_fp8_ptpc value={us['theirs']:.2f} flops={flops} bytes={bytes_theirs:.0f} "
          f"ratio={us['ours'] / us['theirs']:.3f} kernels={expected}", flush=True)
    print(f"INFO case={name} I={inter}->{inter_pad} path={'>'.join(path)} activation={mode} "
          f"a8w4={util(us['ours'], bytes_ours)} aiter={util(us['theirs'], bytes_theirs)} "
          f"worst_check=calc_diff:{worst['calc_diff']:.2e},err_ratio:{worst['err_ratio'] * 100:.2f}% "
          f"host_us=a8w4:{host['ours']:.0f},aiter:{host['theirs']:.0f} idle_us={idle_us:.0f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    seed = int(os.environ.get("MF_SEED", "0"))
    import aiter.fused_moe as fm
    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.shuffle import shuffle_weight

    from pyhip.ops.moe.flydsl import moe_a8w4
    from pyhip.testing import moe_a8w4 as oracle

    AITER_CONFIGS.get_config_file.cache_clear()
    fm.cfg_2stages = None
    fm.get_2stage_cfgs.cache_clear()
    began = time.time()
    print(f"INFO seed={seed} aiter_table={_table_src} warmup={WARMUP} iters={ITERS} groups={GROUPS}", flush=True)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    cycles_per_us = _cycles_per_us()
    for tp_index, tp in enumerate(INTER):
        names = [n for n in args.cases if n.startswith(tp + "_")]
        if not names:
            continue
        inter, inter_pad = INTER[tp]
        w = oracle.make_weights(EXPERTS, HIDDEN, inter, seed=seed * 1000 + tp_index, fp8_inter=inter_pad)
        prepared = moe_a8w4.prepare_a8w4_weights(w["w1"], w["w1_scale"], w["w2"], w["w2_scale"])
        fp8_w = dict(w1=shuffle_weight(w.pop("w1_fp8"), layout=(16, 16)), w1_scale=w.pop("w1_fp8_scale"),
                     w2=shuffle_weight(w.pop("w2_fp8"), layout=(16, 16)), w2_scale=w.pop("w2_fp8_scale"))
        for name in names:
            run_case(name, (w, prepared, fp8_w), cycles_per_us, flush, seed)
        del w, prepared, fp8_w
        torch.cuda.empty_cache()
    print(f"INFO done in {time.time() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main()
