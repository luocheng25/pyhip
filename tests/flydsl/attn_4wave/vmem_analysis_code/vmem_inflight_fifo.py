# SPDX-License-Identifier: MIT
"""CPU-only finite-admission overlay on the seven immutable ATT payloads.

Capacity is a user-adjustable modeling parameter, defaulting to 12 per CU.
Full-load measurements supply latency proxies, NOT the FIFO capacity.
Historical event timestamps, models, frozen source and manifests stay intact.
"""

import argparse
import base64
from collections import defaultdict
import gzip
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import re
import tarfile

import numpy as np

from vmem_inflight_deficit import gap_partition
from vmem_inflight import contiguous
from vmem_inflight_timeline import embedded_font


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SLUGS = ("pyhip", "flydsl_bn32", "flydsl_bn64", "m256_persist", "m256", "m128", "m128_persist")
MAX_BINS = 1_000_000
DEFAULT_CAPACITY = 12
ASSUMPTIONS = (
    "仅为内存到达/服务的反事实模拟：保留ATT成功issue作为到达，按原事件顺序FCFS；"
    "每CU所有wave共享一个槽位池，每条VMEM占1槽，包括metadata/atomic，不是按字节占槽。"
    "容量满则等最早完成释放；同刻先完成再接纳；服务完成可乱序，接纳不越过前面的请求。"
    "容量是用户可调的模拟参数，不从满载平均等效占用量推导，并非已测硬件FIFO深度；"
    "没有证据证明真实A/B/C共享同一个FIFO。满载探针L已有路径等待，再次加队列可能重复计入背压；"
    "不重排计算、依赖、屏障或CTA接纳，不能用于预测实际kernel时间/加速。"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def packed(value):
    return gzip.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(), mtime=0)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def schedule(data, latency_ns, capacity):
    """Independent reference: explicit server-release vector, not a heap."""
    if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 8192):
        raise ValueError("capacity must be an integer in [1,8192] or None")
    ns = [latency_ns[kind["key"]] for kind in data["classes"]]
    if any(not math.isfinite(value) or not 1 <= value <= 10000 for value in ns):
        raise ValueError("each latency must be finite and in [1,10000] ns")
    if not math.isfinite(data["nsPerCycle"]) or data["nsPerCycle"] <= 0:
        raise ValueError("invalid ATT calibration")
    lengths = np.asarray(ns) / data["nsPerCycle"]
    count = len(data["events"])
    starts, expires, delays = np.zeros(count), np.zeros(count), np.zeros(count)
    free = np.full(capacity or 1, -np.inf)
    prior, arrival_before = -np.inf, -np.inf
    for i, event in enumerate(data["events"]):
        arrival, kind = event[0], event[8] if len(event) > 8 else event[4]
        if not math.isfinite(arrival) or arrival < arrival_before:
            raise ValueError("ATT arrivals must be finite and sorted")
        slot = int(np.argmin(free))
        start = max(arrival, prior, float(free[slot])) if capacity is not None else arrival
        starts[i] = start
        expires[i] = start + lengths[kind]
        delays[i] = start - arrival
        if capacity is not None:
            free[slot] = expires[i]
        prior, arrival_before = start, arrival
    # Completions sort before admissions at a tied time: half-open lifetimes.
    edges = sorted([(float(t), -1) for t in expires] + [(float(t), 1) for t in starts])
    active = peak = 0
    for _, change in edges:
        active += change
        assert active >= 0
        peak = max(peak, active)
    assert active == 0 and (capacity is None or peak <= capacity)
    return {"starts": starts, "expires": expires, "delays": delays, "latencies": lengths,
            "capacity": capacity, "peakActive": peak, "delayed": int(np.count_nonzero(delays)),
            "totalDelay": float(delays.sum()), "maxDelay": float(delays.max()) if count else 0.,
            "lastCompletion": max(data["begin"], float(expires.max())) if count else data["begin"]}


def reference_model(data, latency_ns, capacity):
    plan = schedule(data, latency_ns, capacity)
    begin, step = data["begin"], data["step"]
    bins = math.ceil((max(data["end"], plan["lastCompletion"]) - begin) / step)
    if bins > MAX_BINS:
        raise ValueError("more than 1,000,000 bins; no events silently truncated")
    kinds = len(data["classes"])
    counts, gbs = np.zeros((kinds, bins)), np.zeros((kinds, bins))
    # Direct per-event fractional intersections, independent of JS delta bins.
    for i, event in enumerate(data["events"]):
        start, finish = plan["starts"][i], plan["expires"][i]
        l = max(0, math.floor((start - begin) / step))
        r = min(bins, math.ceil((finish - begin) / step))
        if r <= l:
            continue
        times = begin + np.arange(l, r) * step
        overlaps = np.maximum(0., np.minimum(times + step, finish) - np.maximum(times, start)) / step
        kind, latency_kind = event[4], event[8] if len(event) > 8 else event[4]
        counts[kind, l:r] += overlaps
        gbs[kind, l:r] += overlaps * event[5] / latency_ns[data["classes"][latency_kind]["key"]]
    # Integrate the waiting intervals with vectorized start/end boundary flux.
    waiting_edges, flux = np.zeros(bins), np.zeros(bins + 1)
    for sign, times in ((1, [e[0] for e in data["events"]]), (-1, plan["starts"])):
        positions = (np.asarray(times) - begin) / step
        for pos in positions:
            if pos < 0:
                flux[0] += sign
            elif pos < bins:
                index = math.floor(pos)
                waiting_edges[index] += sign * (index + 1 - pos)
                flux[index + 1] += sign
    waiting = waiting_edges + np.cumsum(flux[:-1])
    waiting[np.abs(waiting) < 1e-9] = 0
    end = begin + bins * step
    assert np.all(waiting >= 0)
    fifo = {key: value for key, value in plan.items() if key not in ("starts", "expires", "delays")}
    fifo.update(peakQueued=float(waiting.max()), captureEnd=data["end"], tailCycles=end - data["end"])
    result = {"latencies": plan["latencies"], "bandwidth": gbs.sum(axis=0) * 256 / 1000,
              "count": counts.sum(axis=0), "classGBs": gbs, "classCount": counts,
              "starts": plan["starts"], "expires": plan["expires"], "delays": plan["delays"],
              "waitingCount": waiting, "end": end, "fifo": fifo}
    if data["events"] and data["events"][0][0] >= begin:
        actual_bytes = result["bandwidth"].sum() * step * data["nsPerCycle"] * 1000 / 256
        assert math.isclose(actual_bytes, sum(e[5] for e in data["events"]), abs_tol=1e-4, rel_tol=1e-11)
        assert math.isclose(waiting.sum() * step, plan["totalDelay"], abs_tol=1e-5, rel_tol=1e-10)
        assert math.isclose(counts.sum() * step, sum(plan["expires"] - plan["starts"]), abs_tol=1e-5, rel_tol=1e-10)
    return result


def reference_diagnostic(data, values, domain):
    shifted = [[float(values["starts"][i]), *e[1:]] for i, e in enumerate(data["events"])]
    begin, end = data["interior"] if domain == "interior" else (data["begin"], data["end"] if domain == "capture" else values["end"])
    step = data["step"]
    first, stop = (begin - data["begin"]) // step, (end - data["begin"]) // step
    bandwidth = np.asarray(values["bandwidth"])[first:stop]
    deficits = np.maximum(0, data["target"] - bandwidth)
    low = deficits > 0
    total = math.fsum(deficits) * step
    low_cycles = int(low.sum()) * step

    # Independent extended-precision integral, not the historical double
    # cumsum: long capacity-12 tails amplify its roundoff at PC boundaries.
    # Browser uses compensated double sums; tolerances remain unchanged.
    prefix = np.r_[np.longdouble(0), np.cumsum(deficits, dtype=np.longdouble) * step]
    low_prefix = np.r_[0, np.cumsum(low) * step]

    def area_at(times, *, low_only=False):
        offsets = np.clip(np.asarray(times) - begin, 0, end - begin)
        indices = np.minimum((offsets // step).astype(int), len(bandwidth) - 1)
        samples, accumulated = (low, low_prefix) if low_only else (deficits, prefix)
        return accumulated[indices] + samples[indices] * (offsets - indices * step)

    gaps = gap_partition(shifted, begin, end)
    left = np.array([g["begin"] for g in gaps])
    right = np.array([g["end"] for g in gaps])
    gap_areas = area_at(right) - area_at(left)
    gap_lows = area_at(right, low_only=True) - area_at(left, low_only=True)
    groups = defaultdict(list)
    for i, gap in enumerate(gaps):
        gap.update(deficit=float(max(0, gap_areas[i])), lowCycles=float(max(0, gap_lows[i])))
        groups[tuple(gap["pcs"])].append(gap)
    hotspots = []
    for pcs, rows in groups.items():
        area = math.fsum(g["deficit"] for g in rows)
        example = max(rows, key=lambda g: g["deficit"])
        supplier_ids = {i for g in rows for i in g["nextIds"]}
        hotspots.append({"pcs": list(pcs), "deficit": area, "share": area / total if total else 0,
                         "lowCycles": math.fsum(g["lowCycles"] for g in rows), "gapCount": len(rows),
                         "lowGapCount": sum(g["lowCycles"] > 1e-7 for g in rows),
                         "waveTasks": len({tuple(data["events"][i][2:4]) for i in supplier_ids}),
                         "example": {key: example[key] for key in ("begin", "end", "nextIds", "deficit")}})
    hotspots.sort(key=lambda row: (-row["deficit"], row["pcs"]))
    for rank, row in enumerate(hotspots, 1):
        row["rank"] = rank
    regions = []
    event_times = values["starts"]
    for l, r in contiguous(low):
        start, finish = begin + l * step, begin + r * step
        area = math.fsum(deficits[l:r]) * step
        lo, hi = np.maximum(left, start), np.minimum(right, finish)
        mask = hi > lo
        pieces = area_at(hi[mask]) - area_at(lo[mask])
        by_pc = defaultdict(list)
        for i, piece in zip(np.flatnonzero(mask), pieces):
            by_pc[tuple(gaps[i]["pcs"])].append(float(max(0, piece)))
        pc_areas = {pcs: math.fsum(parts) for pcs, parts in by_pc.items()}
        top_pcs = min(pc_areas, key=lambda pcs: (-pc_areas[pcs], pcs))
        regions.append({"begin": start, "end": finish, "cycles": finish - start,
                        "deficit": area, "share": area / total if total else 0,
                        "meanBandwidth": math.fsum(bandwidth[l:r]) / (r - l), "minBandwidth": float(bandwidth[l:r].min()),
                        "index": first + l + int(np.argmin(bandwidth[l:r])),
                        "issued": int(np.searchsorted(event_times, finish) - np.searchsorted(event_times, start)),
                        "topPcs": list(top_pcs), "topShare": pc_areas[top_pcs] / area if area else 0})
    regions.sort(key=lambda row: (-row["deficit"], row["begin"]))
    for rank, row in enumerate(regions, 1):
        row["rank"] = rank
    assert math.isclose(math.fsum(r["deficit"] for r in hotspots), total, abs_tol=1e-7, rel_tol=0)
    assert math.isclose(math.fsum(r["deficit"] for r in regions), total, abs_tol=1e-7, rel_tol=0)
    return {"begin": begin, "end": end, "cycles": end - begin, "deficit": total, "lowCycles": low_cycles,
            "lowFraction": low_cycles / (end - begin), "deficitFraction": total / (data["target"] * (end - begin)),
            "meanBandwidth": math.fsum(bandwidth) / len(bandwidth), "regions": regions, "hotspots": hotspots, "gaps": gaps}


def configuration(data, measured, source, default_capacity=DEFAULT_CAPACITY):
    if isinstance(default_capacity, bool) or not isinstance(default_capacity, int) or not 1 <= default_capacity <= 8192:
        raise ValueError("default capacity must be an integer in [1,8192]")
    presets = {}
    for name in ("m256", "m128"):
        row = next(r for r in measured["comparisons"] if r["label"] == name + "_full")
        assert row["CU_divisor"] == 256 and row["physical_FIFO_capacity_slots"] is None
        ns, equivalent = row["sampled_PMC_all_probe_latency_ns"], row["equivalent_requests_per_CU_by_kind"]
        for kind in "ABC":
            assert math.isclose(row["bulk_payload_mean_CU_GBs"][kind] * ns[kind] / 1024, equivalent[kind], rel_tol=1e-12)
        total = sum(equivalent.values())
        assert math.isclose(total, row["equivalent_requests_per_CU_total"], rel_tol=1e-12)
        presets[name] = {"estimate": total, "estimateByKind": equivalent,
                         "latencyNs": {"b_dma": ns["B"], "a_load": ns["A"], "c_store": ns["C"],
                                       "metadata": ns["A"], "atomic": data["latencyNs"]["mean"]["atomic"]},
                         "sourceLabel": row["label"], "gridCTAs": row["grid_CTAs"], "groups": row["groups"],
                         "order": row["order"], "sampledPMCTBs": row["sampled_PMC_TBs"],
                         "latencyPopulation": "same counted sampled-PMC run: ALL probes, not full256-only subset"}
    default = "m128" if data["family"] == "m128" else "m256"
    proxy = data["family"] in ("legacy", "pyhip")
    provenance = (f"容量默认{default_capacity}条/CU，可独立调整；这是用户指定的模拟参数，不是39/37平均占用的取整值，也不是实测硬件深度。"
                  f"延迟仍采用{default.upper()}同组满载sampled-PMC的ALL探针L；切换延迟预设不改容量。"
                  "不是取全256区间子集L混配全dispatch带宽。"
                  + ("本页为PyHIP/Legacy：借用M256比例参数，非该kernel/DMA/cache策略专门测量。" if proxy else
                     "A/B/C比例和策略来自同家族合成满载实验，不是该ATT逐请求退休时间。")
                  + f" 数据源sha256：{digest(source)}；metadata沿A/B/C代理，atomic保留历史Mean L。")
    return {"defaultPreset": default, "defaultCapacity": default_capacity, "presets": presets, "proxy": proxy, "assumptions": ASSUMPTIONS,
            "provenance": provenance, "measurementPath": str(source), "measurementSHA256": digest(source),
            "capacityPolicy": "independent user parameter; not inferred from measured average occupancy", "slotUnit": "one captured wave-VMEM"}


def verify_inputs(old, archive):
    report = json.loads((old / "verified.json").read_text())
    assert report["status"] == "PASS"
    checked, archived = {}, {}
    with tarfile.open(archive) as snapshots:
        for group, relative in (("output_sha256", True), ("input_sha256", False), ("protected_sha256", False)):
            for name, expected in report[group].items():
                path = (old / name).resolve() if relative else Path(name)
                if digest(path) == expected:
                    checked[str(path)] = expected
                else:
                    # Only old generator sources may be resolved through the
                    # pre-change archive. Raw artifacts never bypass checks.
                    assert path.parent == HERE and group == "input_sha256", str(path)
                    member = str(path.relative_to(ROOT))
                    stream = snapshots.extractfile(member)
                    assert stream is not None and hashlib.sha256(stream.read()).hexdigest() == expected, member
                    archived[str(path)] = expected
    checked[str((old / "verified.json").resolve())] = digest(old / "verified.json")
    return checked, archived


def make_reference(data, ns, capacity):
    model = reference_model(data, ns, capacity)
    diagnostics = {}
    for domain in ("interior", "capture", "full"):
        d = reference_diagnostic(data, model, domain)
        diagnostics[domain] = {k: v for k, v in d.items() if k != "gaps"}
    return {"ns": ns, "capacity": capacity, "model": plain(model), "diagnostics": diagnostics}


def build(old_root, source, archive, output, font, default_capacity=DEFAULT_CAPACITY):
    from plotly.offline import get_plotlyjs

    assert not output.exists(), output
    measured = json.loads(source.read_text())
    measurement_manifest = json.loads(source.with_name("verified.json").read_text())
    assert measured["status"] == measurement_manifest["status"] == "PASS"
    assert digest(source) == measurement_manifest["output_sha256"][source.name]
    for path, expected in measurement_manifest["input_sha256"].items():
        assert digest(path) == expected, path
    output.mkdir(parents=True)
    app_paths = [HERE / n for n in ("vmem_inflight_latency.js", "vmem_inflight_fifo.js", "vmem_inflight_timeline.js")]
    template_path = HERE / "vmem_inflight_timeline.template.html"
    license_path = ROOT / "tests/flydsl/moe_8w_down/figures/FONT-LICENSE.txt"
    license_text = license_path.read_text()
    app, template, plotly = "\n".join(p.read_text() for p in app_paths), template_path.read_text(), get_plotlyjs()
    inputs = {str(p): digest(p) for p in [Path(__file__).resolve(), *app_paths, template_path, source, source.with_name("verified.json"), archive, font, license_path,
                                         HERE / "vmem_inflight_deficit.py", HERE / "vmem_inflight_timeline.py", HERE / "vmem_inflight_supply.py", HERE / "vmem_inflight.py"]}
    rows = []
    for slug in SLUGS:
        old, dest = old_root / slug, output / slug
        checked, archived = verify_inputs(old, archive)
        data = json.loads(gzip.decompress((old / "timeline_data.json.gz").read_bytes()))
        assert data["schema"] == 3 and data["adjustableLatency"]
        assert data["events"][0][0] >= data["begin"]
        data["fifoConfig"] = configuration(data, measured, source, default_capacity)
        data["schema"] = 4
        params = data["fifoConfig"]["presets"][data["fifoConfig"]["defaultPreset"]]
        reference = make_reference(data, params["latencyNs"], data["fifoConfig"]["defaultCapacity"])
        comparison = make_reference(data, params["latencyNs"], None)
        # Keep full default arrays outside the page for independent browser QA.
        # The page computes these itself; no baseline arrays are overwritten.
        extra = []
        for capacity, b, a, c in ((7, 300., 500., 900.), (61, 900.125, 700.5, 150.25)):
            ns = {**params["latencyNs"], "b_dma": b, "a_load": a, "c_store": c, "metadata": a}
            extra.append(make_reference(data, ns, capacity))
        packed_data = packed(data)
        encoded_font = embedded_font(font, template + app + json.dumps(data, ensure_ascii=False))
        replacements = {"FONT": encoded_font, "PLOTLY": plotly, "PAYLOAD": base64.b64encode(packed_data).decode(),
                "APP": app, "FONT_LICENSE": escape(license_text)}
        html = re.sub(r"@@(FONT|PLOTLY|PAYLOAD|APP|FONT_LICENSE)@@", lambda m: replacements[m[1]], template)
        dest.mkdir()
        (dest / "inflight_timeline.html").write_text(html)
        (dest / "timeline_data.json.gz").write_bytes(packed_data)
        (dest / "fifo_reference.json.gz").write_bytes(packed({"default": reference, "unbounded": comparison, "custom": extra}))
        summary = {"candidate": data["candidate"], "slug": slug, "family": data["family"],
                   "vmem_count": len(data["events"]), "original_bins": len(data["models"]["mean"]["bandwidth"]),
                   "fifo_bins": len(reference["model"]["bandwidth"]), "config": data["fifoConfig"],
                   "fifo": reference["model"]["fifo"], "diagnostics": reference["diagnostics"],
                   "sameLUnboundedInterior": comparison["diagnostics"]["interior"],
                   "originalMeanInterior": data["diagnostics"]["mean"]["interior"], "nsPerCycle": data["nsPerCycle"]}
        write_json(dest / "summary.json", summary)
        outputs = {p.name: digest(p) for p in dest.iterdir() if p.is_file()}
        write_json(dest / "verified.json", {"status": "PASS", "schema": "fifo-model-v1", "input_sha256": {**inputs, **checked},
                                            "archived_input_sha256": archived, "output_sha256": outputs,
                                            "note": "CPU direct-overlap and independent slot-vector reference; browser checks are separate."})
        rows.append(summary)
        print("FIFO_BUILD_PASS", slug, "capacity", data["fifoConfig"]["defaultCapacity"], "events", len(data["events"]), "bins", summary["fifo_bins"], flush=True)
    index = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>七配置 · FIFO背压在途带宽</title><style>body{margin:32px;background:#101620;color:#e1e9f6;font:15px/1.7 system-ui,sans-serif}main{max-width:1400px;margin:auto}a{color:#92c6ff}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #30415a;text-align:left}.table{overflow:auto}.note{padding:16px;background:#192332;border-left:3px solid #82baff}small{color:#b5c5db}</style><main>
<h1>七配置 · 有限FIFO背压在途带宽</h1><p class="note">按CU共享容量接纳，满则等待已有请求完成；只有服务窗口贡献带宽。容量为独立可变参数，不是满载平均占用量或实测硬件FIFO深度。全部页面可独立修改B/A/C延迟和容量，并切换同L无背压或原始Mean对照；切换延迟预设不改变容量。</p>
<p>表中缺口是原内部时间域的FIFO模型结果。当前容量与L若使上界低于5目标，“100%低于目标 / 单一连续低谷”是参数限定，不是全时段硬件缺供给的证据；每页显示按当前参数重算的上界。</p>
<div class="table"><table><thead><tr><th>配置</th><th>默认容量 / CU</th><th>B / A / C 延迟 ns</th><th>FIFO缺口 D/(5T)</th><th>接纳等待请求</th><th>模型尾部 µs</th></tr></thead><tbody>"""
    index = index.replace("<div class=\"table\">", f"<p><b>全部七页默认{default_capacity}条/CU</b>；输入范围1～8192整数，应用按钮或Enter重算；恢复默认及重新打开页面回到{default_capacity}。延迟沿用上一版，不随容量换算。</p><div class=\"table\">", 1)
    for row in rows:
        config, f = row["config"], row["fifo"]
        p = config["presets"][config["defaultPreset"]]
        ns = p["latencyNs"]
        index += (f'<tr><td><a href="{row["slug"]}/inflight_timeline.html">{escape(row["candidate"])}</a></td>'
                  f'<td>{config["defaultCapacity"]}（可调整模型参数）</td>'
                  f'<td>{ns["b_dma"]:.3f} / {ns["a_load"]:.3f} / {ns["c_store"]:.3f}</td>'
                  f'<td>{row["diagnostics"]["interior"]["deficitFraction"]:.3%}</td>'
                  f'<td>{f["delayed"]} / {row["vmem_count"]}</td><td>{f["tailCycles"] * row["nsPerCycle"] / 1000:.3f}</td></tr>')
    baseline_href = Path(os.path.relpath(old_root / "index.html", output)).as_posix()
    index += (f'</tbody></table></div><p><a href="{escape(baseline_href)}">旧七页无背压发布（完整保留）</a></p>'
              f'<p>{escape(ASSUMPTIONS)}</p><small>5 TB/s目标；16-cycle逐bin正缺口，不以issue-stall或等待时长排名。表格是模型结果，不是实测HBM或kernel性能比较。'
              f'测量输入SHA256 {digest(source)}</small>'
              f'<details><summary>内嵌字体版权与SIL许可证</summary><pre>{escape(license_text)}</pre></details></main></html>')
    index_font = embedded_font(font, index)
    index = index.replace("<style>", f"<style>@font-face{{font-family:TimelineCJK;src:url(data:font/woff;base64,{index_font})}}", 1)
    index = index.replace("font:15px/1.7 system-ui", "font:15px/1.7 TimelineCJK,system-ui")
    (output / "index.html").write_text(index)
    write_json(output / "suite_summary.json", {"status": "PASS", "schema": "fifo-suite-v1", "cases": rows,
                                               "original_html_root": str(old_root), "gpu_runs": 0})
    outputs = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    write_json(output / "verified.json", {"status": "PASS", "input_sha256": inputs, "output_sha256": outputs, "pages": 7})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--default-capacity", type=int, default=DEFAULT_CAPACITY,
                        help="Editable per-CU FIFO capacity on first load/reset (default: 12)")
    args = parser.parse_args()
    if not 1 <= args.default_capacity <= 8192:
        parser.error("--default-capacity must be in [1,8192]")
    build(args.old_root.resolve(), args.measurements.resolve(), args.archive.resolve(), args.output.resolve(), args.font.resolve(), args.default_capacity)