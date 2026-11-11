"""Calibrate the QSA union/direct route model (Platform) for BF16 K/V or FP8 K/V per tensor / per token.

measure  For every layout, head count and mode, run the plan with several routings: no plan (every row
         direct) and per-tile ratios with the global route off (union on the tiles K2 proposes). After K1-K3
         (and FP8's Q quantization) cudaPerf times the union phase (K5) and the direct phase (K6 pack + K7
         direct) separately, and the plan's counts, costs and active flags give the route-model features:
         the longest union tile M and the union steps S (N64), TASK_SHARE, the direct BN32 steps (sum, longest
         row) and the packed blocks.
fit      Least-squares fits of the Platform terms on those samples, with their median / p90 errors:
           union  = union_fixed + union_step * max(M, S * TASK_SHARE)
           direct = direct_fixed + pack_block * packed * HK + max(row_step * Dmax, direct_cu_step * HK / CU * Dsum)
sweep    Full calls (K1-K3, quantization, union, direct) with the global route on, for union_ratio candidates
         and the given model; picks the ratio with the lowest mean regret against each layout's best.

Layouts: benchmarks/qsa/_synthetic.py prefills (no prefix), the op test's shared / random chunk selections, and
optionally real selections dumped from SGLang (--dumps: TP0 block IDs [rows, 512] per layer and chunk). K/V
values are random: the routes depend only on the selections. Results stay under mytest/mydata.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.qsa._synthetic import make_case
from pyhip.ops.qsa.flydsl import sparse_prefill_common as common
from pyhip.testing.misc import cudaPerf
from tests.ops.qsa import sparse_prefill_op_test as op
from tests.ops.qsa._attention import _metadata, _runtime
from tests.ops.qsa._benchmark import DATA, default_output

torch.set_default_device(None)

FP8 = torch.float8_e4m3fnuz
MODES = ("bf16", "tensor", "token")
RATIOS = (0.7, 1.0, 1.4, 1.7, 2.0, 2.5, 3.5, math.inf)
SYNTHETIC = (
    ("syn_r1024", 1024, "mixed"),
    ("syn_r2048", 2048, "mixed"),
    ("syn_r4096", 4096, "mixed"),
    ("syn_r8192", 8192, "mixed"),
    ("syn_r12000", 12000, "mixed"),
    ("syn_r12000_l3", 12000, "l3like"),
    ("syn_r12000_l47", 12000, "l47like"),
    ("syn_r16384", 16384, "mixed"),
)
CHUNKS = (
    ("chunk_1024_p30000_random", (1024,), (30000,), "random"),
    ("chunk_1024_p30000_shared", (1024,), (30000,), "shared"),
    ("chunk_4096_p8192_shared", (4096,), (8192,), "shared"),
    ("chunk_4096_p126976_shared", (4096,), (126976,), "shared"),
    ("chunk_4096_p126976_random", (4096,), (126976,), "random"),
    ("chunk_16384_p16384_shared", (16384,), (16384,), "shared"),
    ("ragged_4", (300, 129, 1001, 63), (0, 517, 4093, 12345), "shared"),
    ("serving_3", (480, 1000, 301), (11392, 3, 4001), "shared"),
)


def _indices_from_blocks(blocks, prefix):
    """int32 [rows, 2051] of one request from block IDs [rows, 512] (-1 padded) and its prefix."""
    rows = blocks.shape[0]
    visible = prefix + torch.arange(1, rows + 1)
    complete = (visible // 4).clamp(max=512)
    # Strided columns past a row's complete blocks hold its causal tail, not blocks.
    blocks = torch.where(torch.arange(512) < complete[:, None], blocks.long(), -1)
    assert int((blocks >= 0).sum(1).sub(complete).abs().max()) == 0, "dump blocks per row"
    tokens = (blocks[:, :, None] * 4 + torch.arange(4)).reshape(rows, 2048)
    index = torch.full((rows, 2051), -1, dtype=torch.int32)
    keep = torch.arange(2048) < 4 * complete[:, None]
    index[:, :2048] = torch.where(keep, tokens, -1).to(torch.int32)
    tail = torch.arange(3)
    has = tail < (visible % 4)[:, None]
    position = 4 * complete[:, None] + tail
    value = ((visible // 4) * 4)[:, None] + tail
    row = torch.arange(rows)[:, None].expand(-1, 3)
    index[row[has], position[has]] = value[has].to(torch.int32)
    return index


def layouts(args):
    """(name, query_lens, prefix_lens, indices builder) of the selected layout sets."""
    out = []
    if "synthetic" in args.sets:
        for name, rows, profile in SYNTHETIC:
            out.append((name, (rows,), (0,), lambda rows=rows, profile=profile: make_case(
                rows, 12, profile=profile, seed=args.seed)[0].indices))
    if "chunks" in args.sets:
        for name, queries, prefixes, selection in CHUNKS:
            def build(queries=queries, prefixes=prefixes, selection=selection):
                generator = torch.Generator().manual_seed(args.seed)
                return op.make_indices(queries, prefixes, selection, generator, 512)
            out.append((name, queries, prefixes, build))
    if args.dumps:
        # The dump run's status.json maps files to its prompts.
        status = Path(args.dumps).parent / "status.json"
        prompts = {}
        if status.is_file():
            for request in json.loads(status.read_text())["requests"]:
                prompts.update({name: request["name"] for name in request["files"]})
        for path in sorted(Path(args.dumps).glob("*.pt")):
            if args.dump_files and path.name not in args.dump_files:
                continue
            record = torch.load(path)
            name = f"real_{prompts.get(path.name, 'dump')}_" + path.stem.split("_", 1)[1]
            if args.dump_layers and record["layer"] not in args.dump_layers:
                continue
            out.append((name, tuple(record["query_lens"]), tuple(record["prefix_lens"]),
                        lambda record=record: _indices_from_blocks(record["blocks"], record["prefix_lens"][0])))
    return out


def _quantize(x, per_token):
    x = x.float()
    amax = x.abs().amax(-1) if per_token else x.abs().amax().reshape(1)
    scale = (amax / 240.0).clamp_min(1e-12)
    x8 = (x / (scale[..., None] if per_token else scale)).clamp(-240.0, 240.0).to(FP8)
    return x8, scale.contiguous()


class Case:
    """One layout, head count and mode on the GPU, with its routings."""

    def __init__(self, name, queries, prefixes, indices, heads, mode, device, seed):
        rows, total = sum(queries), sum(queries) + sum(prefixes)
        generator = torch.Generator(device=device).manual_seed(seed)
        q = torch.randn((rows, heads, 256), generator=generator, device=device, dtype=torch.bfloat16)
        k = torch.randn((total, 1, 256), generator=generator, device=device, dtype=torch.bfloat16)
        v = torch.randn(k.shape, generator=generator, device=device, dtype=torch.bfloat16)
        self.name, self.heads, self.mode, self.device = name, heads, mode, device
        self.value = _metadata(q, k, v, indices.to(device), queries, prefixes)
        self.fp8 = None if mode == "bf16" else mode
        self.ks = self.vs = None
        if self.fp8:
            k8, self.ks = _quantize(k, mode == "token")
            v8, self.vs = _quantize(v, mode == "token")
            self.value.k, self.value.v = k8, v8
        self.out = torch.empty_like(q)
        value = self.value
        self.workspace = _runtime._Workspace(value.q, value.k, value.v, value.indices, value.query_lens,
                                             value.prefix_lens, value.scale, None, self.fp8)
        tokens = (value.indices >= 0).sum(1)
        self.row_steps = ((tokens + 31) // 32).cpu().numpy()
        # Blocks of the requests with queries, as allocate_plan counts the packed KV.
        self.packed = sum(-(-(q + p) // 4) for q, p in zip(queries, prefixes) if q)

    def _bind(self):
        v = self.value
        return self.workspace.bind(v.q, v.k, v.v, v.indices, None, self.ks, self.vs)

    def route(self, ratio, routing=False, model=None):
        """A routing: ratio None = no union plan, else a plan with this per-tile ratio, the global route
        on or off and the given route model (keys as Platform; default the platform's)."""
        inputs = self._bind()
        plan = None
        if ratio is not None:
            plan = _runtime.prepare.allocate_plan(inputs=inputs, query_tile=32)
            plan.union_ratio, plan.routing = ratio, routing
            if model is not None:
                hk, cus = inputs.k.shape[1], common.get_cu_num()
                plan.route_model = (model["union_fixed"], model["union_step"], model["row_step"],
                                    model["route_margin"])
                plan.route_costs = (plan.route_costs[0], model["direct_cu_step"] * hk / cus,
                                    model["direct_fixed"] + model["pack_block"] * self.packed * hk)
        if self.fp8:
            direct = _runtime.direct_fp8.prepare(inputs=inputs, union=plan, per_token=self.fp8 == "token")
        else:
            direct = _runtime.direct.prepare(inputs=inputs, union=plan)
        direct.source_blocks = self.workspace.block_indices
        return SimpleNamespace(plan=plan, direct=direct, inputs=inputs)

    def phases(self, route):
        """(prepare, union, direct) callables of a routing."""
        v, rt, inputs = self.value, _runtime, route.inputs

        def prepare():
            rt.prepare.run(inputs=inputs, plan=route.plan)
            if self.fp8:
                rt.union_fp8.quantize_q(v.q, inputs.q8, inputs.qs)

        def union():
            if route.plan is not None:
                (rt.union_fp8 if self.fp8 else rt.union).run(inputs=inputs, plan=route.plan, out=self.out)

        def direct():
            (rt.direct_fp8 if self.fp8 else rt.direct).run(inputs=inputs, prepared=route.direct, out=self.out)

        return prepare, union, direct

    def features(self, route):
        plan, rows, packed = route.plan, len(self.row_steps), self.packed
        if plan is None:
            return dict(rows=rows, tiles=0, union_tiles=0, union_rows=0, M=0, S=0, task_share=0.0,
                        Dsum=int(self.row_steps.sum()), Dmax=int(self.row_steps.max(initial=0)),
                        direct_rows=rows, packed=packed, hk=1)
        counts = plan.counts.cpu().numpy()
        costs = plan.costs.cpu().numpy().reshape(-1, 3)
        active = plan.active.cpu().numpy() == 1
        sizes = plan.metadata.cpu().numpy().reshape(-1, 5)[:, 1]
        steps = (counts + 15) // 16
        return dict(rows=rows, tiles=int(plan.num_tiles), union_tiles=int(active.sum()),
                    union_rows=int(sizes[active].sum()), M=int(steps[active].max(initial=0)),
                    S=int(steps[active].sum()), task_share=float(plan.route_costs[0]), grid=int(plan.grid),
                    query_tile=int(plan.query_tile), Dsum=int(costs[~active, 0].sum()),
                    Dmax=int(costs[~active, 1].max(initial=0)), direct_rows=int(rows - sizes[active].sum()),
                    packed=packed, hk=1)


def _time(timer, fn):
    with timer:
        fn()
    return timer.latencies[-1] * 1e6


def measure(args):
    folder = args.output
    folder.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    timer = cudaPerf(name="qsa_route_calibration", verbose=0)
    if not timer.enable:
        raise RuntimeError("CUDAPERF disabled timing")
    records = []
    sink = (folder / "samples.jsonl").open("x")
    for name, queries, prefixes, build in layouts(args):
        if args.only and name not in args.only:
            continue
        indices = build()
        for heads in args.heads:
            for mode in args.modes:
                case = Case(name, queries, prefixes, indices, heads, mode, device, args.seed)
                settings = [("noplan", None)] + [(f"ratio_{r:g}", r) for r in RATIOS]
                for setting, ratio in settings:
                    if ratio is not None and sum(queries) < 2:
                        continue
                    route = case.route(ratio)
                    prepare, union, direct = case.phases(route)
                    for _ in range(2):
                        prepare(), union(), direct()
                    torch.cuda.synchronize()
                    features = case.features(route)
                    tu, td, tt = [], [], []
                    for _ in range(args.samples):
                        prepare()
                        tu.append(_time(timer, union))
                        td.append(_time(timer, direct))
                        tt.append(_time(timer, lambda: (prepare(), union(), direct())))
                    record = dict(layout=name, heads=heads, mode=mode, setting=setting, ratio=ratio,
                                  query_lens=list(queries), prefix_lens=list(prefixes), **features,
                                  union_us=statistics.median(tu), direct_us=statistics.median(td),
                                  total_us=statistics.median(tt), union_min=min(tu), direct_min=min(td),
                                  total_min=min(tt))
                    assert bool(torch.isfinite(case.out.float()).all()), (name, heads, mode, setting)
                    records.append(record)
                    sink.write(json.dumps(record, default=float) + "\n")
                    sink.flush()
                    print(f"{name} H{heads} {mode} {setting}: union {record['union_us']:.1f} "
                          f"direct {record['direct_us']:.1f} total {record['total_us']:.1f} us "
                          f"(M {features['M']} S {features['S']} Dsum {features['Dsum']} Dmax {features['Dmax']})",
                          flush=True)
                del case
                torch.cuda.empty_cache()
    sink.close()
    return records


def sweep(args):
    """Full calls (K1-K3, quantization, union, direct) with the global route on: the platform's own route
    (config "platform") and the --model constants with each --ratios value, interleaved per sample."""
    folder = args.output
    folder.mkdir(parents=True, exist_ok=False)
    models = json.loads(args.model.read_text())
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    timer = cudaPerf(name="qsa_route_sweep", verbose=0)
    if not timer.enable:
        raise RuntimeError("CUDAPERF disabled timing")
    sink = (folder / "sweep.jsonl").open("x")
    for name, queries, prefixes, build in layouts(args):
        if args.only and name not in args.only:
            continue
        indices = build()
        for heads in args.heads:
            for mode in args.modes:
                case = Case(name, queries, prefixes, indices, heads, mode, device, args.seed)
                platform = common.platform()
                configs = {"platform": case.route(platform.union_ratio, routing=True)}
                for ratio in args.ratios:
                    configs[f"model_{ratio:g}"] = case.route(ratio, routing=True, model=models[mode])
                calls = {}
                for key, route in configs.items():
                    prepare, union, direct = case.phases(route)
                    calls[key] = lambda prepare=prepare, union=union, direct=direct: (prepare(), union(), direct())
                    for _ in range(2):
                        calls[key]()
                features = {key: case.features(route) for key, route in configs.items()}
                times = {key: [] for key in configs}
                keys = list(configs)
                for sample in range(args.samples):
                    order = keys[sample % len(keys):] + keys[: sample % len(keys)]
                    for key in order if sample % 2 == 0 else order[::-1]:
                        times[key].append(_time(timer, calls[key]))
                record = dict(layout=name, heads=heads, mode=mode, query_lens=list(queries),
                              prefix_lens=list(prefixes), totals={k: statistics.median(v) for k, v in times.items()},
                              union_tiles={k: f["union_tiles"] for k, f in features.items()},
                              tiles=features["platform"]["tiles"])
                sink.write(json.dumps(record, default=float) + "\n")
                sink.flush()
                best = min(record["totals"].values())
                print(f"{name} H{heads} {mode}: " + " ".join(
                    f"{k} {v:.0f}({v / best - 1:+.1%},u{record['union_tiles'][k]})" for k, v in record["totals"].items()),
                    flush=True)
                del case, configs, calls
                torch.cuda.empty_cache()
    sink.close()


def _errors(predicted, measured):
    rel = np.abs(np.asarray(predicted) - np.asarray(measured)) / np.maximum(np.asarray(measured), 1e-9)
    return dict(median=float(np.median(rel)), p90=float(np.percentile(rel, 90)), max=float(rel.max()), n=len(rel))


def fit(records, cus):
    """Least-squares route terms of one mode (relative-error weights)."""
    union = [r for r in records if r["S"] > 0]
    x = np.array([max(r["M"], r["S"] * r["task_share"]) for r in union], dtype=float)
    y = np.array([r["union_us"] for r in union])
    w = 1.0 / y
    a, b = np.linalg.lstsq(np.stack([np.ones_like(x), x], 1) * w[:, None], y * w, rcond=None)[0]
    result = dict(union_fixed=float(a), union_step=float(b), union_error=_errors(a + b * x, y))
    direct = [r for r in records if r["Dsum"] > 0]
    y = np.array([r["direct_us"] for r in direct])
    w = 1.0 / y
    best = None
    for row_step in np.linspace(0.05, 2.0, 79):
        for cu_step in np.linspace(0.02, 1.0, 99):
            term = np.array([max(row_step * r["Dmax"], cu_step * r["hk"] / cus * r["Dsum"]) for r in direct])
            packed = np.array([r["packed"] * r["hk"] for r in direct], dtype=float)
            design = np.stack([np.ones_like(term), packed], 1)
            coef = np.linalg.lstsq(design * w[:, None], (y - term) * w, rcond=None)[0]
            pred = design @ coef + term
            loss = float(np.sum(((pred - y) * w) ** 2))
            if best is None or loss < best[0]:
                best = (loss, row_step, cu_step, coef, pred)
    _, row_step, cu_step, coef, pred = best
    result.update(direct_fixed=float(coef[0]), pack_block=float(coef[1]), row_step=float(row_step),
                  direct_cu_step=float(cu_step), direct_error=_errors(pred, y))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--gpu", type=int, default=0)
    m.add_argument("--modes", nargs="+", choices=MODES, default=["tensor", "token"])
    m.add_argument("--heads", nargs="+", type=int, choices=(12, 6, 3), default=[12, 6, 3])
    m.add_argument("--sets", nargs="*", choices=("synthetic", "chunks"), default=["synthetic", "chunks"])
    m.add_argument("--dumps", type=Path, help="directory of SGLang selection dumps (*.pt)")
    m.add_argument("--dump-layers", nargs="*", type=int, default=[])
    m.add_argument("--dump-files", nargs="*", default=[], help="dump file names to keep (all by default)")
    m.add_argument("--only", nargs="*", default=[], help="layout names to keep (all by default)")
    m.add_argument("--samples", type=int, default=10)
    m.add_argument("--seed", type=int, default=20261111)
    m.add_argument("--output", type=Path, default=default_output("route_calibration"))
    w = sub.add_parser("sweep")
    for flag, kwargs in (
        ("--gpu", dict(type=int, default=0)),
        ("--modes", dict(nargs="+", choices=MODES, default=["tensor", "token"])),
        ("--heads", dict(nargs="+", type=int, choices=(12, 6, 3), default=[12, 6, 3])),
        ("--sets", dict(nargs="*", choices=("synthetic", "chunks"), default=["synthetic", "chunks"])),
        ("--dumps", dict(type=Path)),
        ("--dump-layers", dict(nargs="*", type=int, default=[])),
        ("--dump-files", dict(nargs="*", default=[])),
        ("--only", dict(nargs="*", default=[])),
        ("--samples", dict(type=int, default=20)),
        ("--seed", dict(type=int, default=20261111)),
        ("--model", dict(type=Path, required=True, help="JSON: mode -> Platform route terms")),
        ("--ratios", dict(nargs="+", type=float, default=[1.7, 2.0, 2.25, 2.5])),
        ("--output", dict(type=Path, default=default_output("route_sweep"))),
    ):
        w.add_argument(flag, **kwargs)
    f = sub.add_parser("fit")
    f.add_argument("samples", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command in ("measure", "sweep"):
        if not args.output.resolve().is_relative_to(DATA.resolve()):
            parser.error("Results must stay under mytest/mydata")
        (measure if args.command == "measure" else sweep)(args)
    else:
        records = [json.loads(line) for path in args.samples for line in path.read_text().splitlines()]
        cus = 80
        report = {}
        for mode in sorted({r["mode"] for r in records}):
            report[mode] = fit([r for r in records if r["mode"] == mode], cus)
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
