"""Qwen3.8-Flash-Next 单卡分片的 per-token FP8（ptpc）MoE 延迟：tuned_moe FlyDSL 对照 Aiter。

Case：TP2（I=320）/ TP4（I=160，补零到 192，见 pyhip.testing.moe_a8w4_ptpc）
× T ∈ {1, 8, 16, 32, 128, 1024, 8192, 65536}，共 16 个；H=2560，E=512，top-10，SiLU，
均匀随机路由、固定种子。

实现（``--impl`` 逗号分隔，第一个是指标）：
  fly_ptpc    tuned_moe 的 FlyDSL ptpc 候选，经 ``make_moe_runner`` 固定为
              ``moe_a8w4_ptpc.FLY_CONFIGS`` 中的配置（不调优、不回退）
  aiter_ptpc  ``aiter.fused_moe.fused_moe(quant_type=per_Token)``，由 Aiter 自选 kernel；
              TP4 与 fly_ptpc 使用同一份补零到 I=192 的权重

计时协议：每个实现用同样的权重和输入。权重 ``--sets``（>=4）组；输入轮换至少 ``--sets`` 组，
小 T 再加组数，让两次复用之间触达的专家权重 >= 512 MiB，先从 L2/MALL 中挤出。先跑
``--warmup``（>=2）轮，再连续跑 ``--warmup-seconds`` 让频率稳定；之后取 ``--iters``（>=20）个样本，
每个样本是下一组输入上的一次调用，前面放一个 sleep kernel（把主机下发时间挡在计时窗口外），
用 GPU event 计时并同步，报告中位数。decode（T<=32）每组输入捕获一个 CUDA graph 并按 replay 计时；
prefill 为 eager。

``--check`` 与 ``--sweep`` 还会记下 tuned_moe 的两次 compile_gemm 被 gemm.py 分流到 upstream 还是 legacy。

``--mf`` 对第一个实现输出 ``MF_METRIC case=<tp2|tp4>_t<T> value=<us> unit=us``，其余实现输出
``REF impl=<名> case=<...> value=<us>``；否则打印表格。``--check`` 先用 FP32 参考检查每个实现
（T=65536 每 16 个 token 取一个）。``--sweep`` 不计指标，而是对每个 case 计时 tuned_moe 生成的
全部 FlyDSL 候选，按中位数排序输出，用于更新 ``FLY_CONFIGS``。
"""

import argparse
import math
import os
import statistics
import sys
import time

from pyhip.testing import moe_a8w4_ptpc as M

IMPLS = ("fly_ptpc", "aiter_ptpc")
CASES = [(tp, tokens) for tp in (2, 4) for tokens in M.BENCH_TOKENS]
GRAPH_MAX_TOKENS = 32
EVICT_BYTES = 512 << 20
MAX_INPUT_SETS = 64
CHECK_STEP = {65536: 16}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", default="fly_ptpc,aiter_ptpc",
                        help="逗号分隔的实现，第一个是指标")
    parser.add_argument("--cases", default="all", help="逗号分隔的 case 名，如 tp2_t1（默认全部 16 个）")
    parser.add_argument("--mf", action="store_true", help="输出 MF_METRIC / REF 行")
    parser.add_argument("--warmup", type=int, default=2, help="预热轮数（>=2）")
    parser.add_argument("--warmup-seconds", type=float, default=1.0, help="采样前连续运行的秒数")
    parser.add_argument("--iters", type=int, default=20, help="计时样本数（>=20）")
    parser.add_argument("--sets", type=int, default=4, help="权重组数（>=4）")
    parser.add_argument("--check", action="store_true", help="计时前用 FP32 参考检查每个实现")
    parser.add_argument("--sweep", action="store_true", help="计时全部 FlyDSL 候选（不输出指标）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.impl = [name.strip() for name in args.impl.split(",") if name.strip()]
    unknown = sorted(set(args.impl) - set(IMPLS))
    if unknown or not args.impl:
        parser.error(f"unknown implementation(s) {unknown}; choose from {IMPLS}")
    names = {M.case_name(tp, tokens): (tp, tokens) for tp, tokens in CASES}
    if args.cases == "all":
        args.cases = list(CASES)
    else:
        try:
            args.cases = [names[name.strip()] for name in args.cases.split(",") if name.strip()]
        except KeyError as exc:
            parser.error(f"unknown case {exc}; choose from {sorted(names)}")
    if args.sweep:
        args.warmup, args.iters, args.warmup_seconds = 2, 10, 0.2
    elif args.warmup < 2 or args.iters < 20 or args.sets < 4:
        parser.error("the protocol needs --warmup >= 2, --iters >= 20 and --sets >= 4")
    return args


def note(message):
    print(f"# {message}", file=sys.stderr, flush=True)


class Sleeper:
    """主机下发被计时调用期间让 GPU 保持忙碌。"""

    def __init__(self):
        import torch

        self.torch = torch
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        cycles = 2_000_000
        for _ in range(2):
            torch.cuda.synchronize()
            start.record()
            torch.cuda._sleep(cycles)
            end.record()
            torch.cuda.synchronize()
        self.cycles_per_us = cycles / (start.elapsed_time(end) * 1000)

    def __call__(self, us):
        self.torch.cuda._sleep(int(us * self.cycles_per_us))


def make_weight_sets(tp, count, seed, keep_raw):
    import torch

    sets = []
    for index in range(count):
        raw, kernel = M.make_weights(tp, seed=seed * 1000 + tp * 100 + index,
                                     keep_raw=keep_raw and index == 0)
        sets.append((raw, kernel))
    torch.cuda.synchronize()
    return sets


def input_set_count(tp, tokens, minimum):
    touched = M.touched_experts(tokens) * 3 * M.KERNEL_INTER[tp] * M.H
    return max(minimum, min(MAX_INPUT_SETS, math.ceil(EVICT_BYTES / touched)))


def make_inputs(tokens, count, seed):
    import torch

    inputs = []
    for index in range(count):
        hidden, weight, ids = M.make_inputs(tokens, seed=seed * 1_000_003 + tokens * 1009 + index * 17)
        inputs.append((hidden, weight, ids, torch.empty_like(hidden)))
    return inputs


def make_call(impl, kernel, inputs, config):
    """返回在一组输入上运行 ``impl`` 的无参函数。"""
    hidden, weight, ids, output = inputs
    if impl == "fly_ptpc":
        runner = M.fly_runner(config)
        call = M.moe_call(kernel, hidden, weight, ids, output)
        return lambda: runner(**call)
    import aiter
    from aiter.fused_moe import fused_moe

    w1, w2, w1_scale, w2_scale = kernel
    return lambda: fused_moe(hidden, w1, w2, weight, ids, activation=aiter.ActivationType.Silu,
                             quant_type=aiter.QuantType.per_Token,
                             w1_scale=w1_scale, w2_scale=w2_scale)


def time_impl(calls, *, graph, warmup, iters, sleeper, warmup_seconds):
    """``calls`` 轮流运行的中位延迟（微秒）。"""
    import torch

    for _ in range(warmup):
        for call in calls:
            call()
    torch.cuda.synchronize()
    runners = calls
    graphs = []
    if graph:
        for call in calls:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call()
            graphs.append(g)
        runners = [g.replay for g in graphs]
    # 主机下发一次调用的时间；sleep kernel 必须比它长。
    enqueue_us = 0.0
    for _ in range(warmup):
        for run in runners:
            begin = time.perf_counter()
            run()
            enqueue_us = max(enqueue_us, (time.perf_counter() - begin) * 1e6)
        torch.cuda.synchronize()
    lead_us = max(200.0, 2 * enqueue_us + 100)
    deadline = time.perf_counter() + warmup_seconds
    while time.perf_counter() < deadline:
        for run in runners:
            run()
        torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for sample in range(max(iters, len(runners))):
        sleeper(lead_us)
        start.record()
        runners[sample % len(runners)]()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    del graphs
    return statistics.median(samples)


class Reference:
    """按数据流缓存一组输入的 FP32 参考；大 T 只比较等间隔抽取的 token。"""

    def __init__(self, raw, inputs):
        import torch

        hidden, weight, ids, _ = inputs
        tokens = hidden.shape[0]
        step = CHECK_STEP.get(tokens, 1)
        self.rows = torch.cat([torch.arange(0, tokens - 1, step, device=hidden.device),
                               torch.tensor([tokens - 1], device=hidden.device)])
        self.args = (hidden[self.rows], raw, weight[self.rows], ids[self.rows])
        self.cache = {}

    def errors(self, output, act_quant):
        if act_quant not in self.cache:
            self.cache[act_quant] = M.reference_moe(*self.args, act_quant=act_quant)
        return M.output_errors(self.cache[act_quant], output[self.rows])


def gemm_routes(run):
    """运行 ``run()``，同时记录 tuned_moe 每次 compile_gemm 被分流到 upstream 还是 legacy。"""
    from pyhip.ops.moe.flydsl import moe_gemm_splitk as fly
    from pyhip.ops.moe.flydsl.moe_gemm_2stage.gemm import gemm_implementation

    original, routes = fly.compile_gemm, []

    def compile_gemm(**params):
        routes.append(gemm_implementation(**params))
        return original(**params)

    fly.compile_gemm = compile_gemm
    try:
        return run(), "/".join(routes)
    finally:
        fly.compile_gemm = original


def check_impl(impl, call, reference, config):
    # Aiter 的 ptpc kernel 对两级输入都按 token 量化。
    act_quant = True if impl == "aiter_ptpc" else M.act_quant(config)
    return reference.errors(call(), act_quant)


def passes(errors):
    beyond, logits_diff, token_err = errors
    return beyond == 0 and logits_diff < M.LOGITS_DIFF_LIMIT and token_err < M.TOKEN_ERR_LIMIT


def run(args):
    import torch

    torch.cuda.set_device(0)
    sleeper = Sleeper()
    results = {}
    for tp in sorted({tp for tp, _ in args.cases}):
        weight_sets = make_weight_sets(tp, args.sets, args.seed, keep_raw=args.check or args.sweep)
        for case_tp, tokens in args.cases:
            if case_tp != tp:
                continue
            case = M.case_name(tp, tokens)
            count = input_set_count(tp, tokens, args.sets)
            inputs = make_inputs(tokens, count, args.seed)
            kernels = [weight_sets[index % len(weight_sets)][1] for index in range(count)]
            reference = (Reference(weight_sets[0][0], inputs[0])
                         if args.check or args.sweep else None)
            timing = dict(graph=tokens <= GRAPH_MAX_TOKENS, warmup=args.warmup, iters=args.iters,
                          sleeper=sleeper, warmup_seconds=args.warmup_seconds)
            if args.sweep:
                results[case] = sweep_case(case, kernels, inputs, reference, timing)
            else:
                config = M.fly_config(tp, tokens)
                row = results.setdefault(case, {"config": M.describe(config), "input_sets": count})
                for impl in args.impl:
                    try:
                        calls = [make_call(impl, kernels[index], inputs[index], config)
                                 for index in range(count)]
                        if args.check:
                            row[f"{impl}_err"], route = gemm_routes(
                                lambda: check_impl(impl, calls[0], reference, config))
                            if impl == "fly_ptpc":
                                row["route"] = route
                            note(f"{impl} {case} vs FP32: beyond/logits_diff/token_err="
                                 f"{row[f'{impl}_err']}")
                        row[impl] = time_impl(calls, **timing)
                    except Exception as exc:  # 记录失败，继续其它 case
                        torch.cuda.synchronize()
                        row[impl] = math.nan
                        note(f"{impl} {case} failed: {type(exc).__name__}: {str(exc)[:300]}")
                    emit(args, case, impl, row)
            del inputs, kernels, reference
            torch.cuda.empty_cache()
        del weight_sets
        torch.cuda.empty_cache()
    return results


def sweep_case(case, kernels, inputs, reference, timing):
    """计时 tuned_moe 对本 case 生成的全部 FlyDSL 候选，返回按中位数排序的列表。"""
    import torch

    hidden, weight, ids, output = inputs[0]
    candidates = M.fly_candidates(M.moe_call(kernels[0], hidden, weight, ids, output))
    rows = []
    for config in candidates:
        try:
            calls = [make_call("fly_ptpc", kernels[index], inputs[index], config)
                     for index in range(len(inputs))]
            errors, route = gemm_routes(lambda: check_impl("fly_ptpc", calls[0], reference, config))
            if not passes(errors):
                note(f"sweep {case} {M.describe(config)} rejected: errors={errors}")
                continue
            us = time_impl(calls, **timing)
        except Exception as exc:  # 编译或形状限制不满足的候选直接跳过
            torch.cuda.synchronize()
            note(f"sweep {case} {M.describe(config)} failed: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        rows.append((us, config, errors, route))
        print(f"SWEEP case={case} us={us:.2f} route={route} {M.describe(config)} errors={errors}",
              flush=True)
    rows.sort(key=lambda row: row[0])
    if rows:
        print(f"BEST case={case} us={rows[0][0]:.2f} config={rows[0][1]}", flush=True)
    return rows


def emit(args, case, impl, row):
    if not args.mf:
        return
    value = row[impl]
    if impl == args.impl[0]:
        print(f"MF_METRIC case={case} value={value:.3f} unit=us", flush=True)
    else:
        print(f"REF impl={impl} case={case} value={value:.3f}", flush=True)


def print_table(args, results):
    if args.sweep:
        for case, rows in results.items():
            print(f"{case}: {len(rows)} passing candidates")
            for us, config, _, route in rows[:5]:
                print(f"  {us:10.2f} us  {route:<17} {M.describe(config)}")
        return
    speedup = args.impl[:2] == ["fly_ptpc", "aiter_ptpc"]
    header = f"{'case':<10} {'fly_ptpc config [gemm.py 分流]':<84} {'sets':>4} "
    header += " ".join(f"{impl + '(us)':>16}" for impl in args.impl)
    if speedup:
        header += f" {'aiter/fly':>9}"
    print(header)
    for case, row in results.items():
        config = f"{row['config']} [{row['route']}]" if "route" in row else row["config"]
        line = f"{case:<10} {config:<84} {row['input_sets']:>4} "
        line += " ".join(f"{row.get(impl, math.nan):>16.2f}" for impl in args.impl)
        if speedup:
            line += f" {row['aiter_ptpc'] / row['fly_ptpc']:>8.3f}x"
        print(line)
        for impl in args.impl:
            if f"{impl}_err" in row:
                beyond, logits_diff, token_err = row[f"{impl}_err"]
                print(f"{'':<10} {impl} vs FP32: beyond={beyond:.2e} "
                      f"logits_diff={logits_diff:.2e} token_err={token_err:.3f}")


def main(argv=None):
    args = parse_args(argv)
    if args.mf:
        # Aiter 的 C++ kernel 加载器会写 stdout；让 stdout 只有 MF_METRIC/REF 行。
        os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
    results = run(args)
    if not args.mf:
        print_table(args, results)
    if args.sweep:
        return 0 if all(results.values()) else 1
    failed = [case for case, row in results.items() if math.isnan(row[args.impl[0]])]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
