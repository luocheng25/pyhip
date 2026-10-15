"""Owned SGLang server runs for the Qwen3.8 MonoKernel decode hooks (native vs MonoKernel arms).

The server mirrors the user's launcher (/opt/sglang/scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh:
AITER attention/MoE, chunked prefill 16384, radix cache off, 32 running requests, decode graphs up to
32, no PLE embedding offload) but takes the visible GPUs as an argument (the launcher pins 0-3).
The ``mono`` arm loads the ``qwen38_monokernel`` plugin (benchmarks/monokernel/plugin, SGLang source
unchanged) with ``PYHIP_MONOKERNEL=1`` from a snapshot of ``src/pyhip`` copied into the run folder
(``source/``), so later edits cannot reach a running server; every run gets a fresh FlyDSL cache.

Modes:
* ``validate``: eager server (no CUDA graphs), ``PYHIP_MONOKERNEL_VALIDATE=1``; real-text prompts at
  concurrency 1 and 4 decode 64 tokens; every MonoKernel layer call is compared with the native layer
  on the same input and states (server log ``MonoKernel validate``), summarized per layer.
* ``bench``: CUDA graphs, validation off; ``sglang.bench_serving`` random 32 x 12000 -> 350 at
  concurrency 1, 2, 4 and 8 (the protocol of the earlier QSA service studies).

    .venv/bin/python benchmarks/monokernel/serve.py --arm mono --mode validate --tp 2 --devices 0,1
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "mytest/mydata"
PLUGIN = Path(__file__).resolve().parent / "plugin"
MODEL = Path("/models/Qwen3.8-Flash-Next-PTPC-FP8")
SERVED = "Qwen/Qwen3.8-Flash-Next-PTPC-FP8"
READY = "The server is fired up and ready to roll!"
CONCURRENCIES = (1, 2, 4, 8)
VALIDATE_LINE = re.compile(r"MonoKernel validate (\{.*\})")


def gpu_state():
    cards = {}
    for card in Path("/sys/class/drm").glob("card[0-9]*"):
        path = card / "device"
        if (path / "gpu_busy_percent").is_file():
            cards[path.resolve().name.lower()] = {
                key: int((path / key).read_text()) for key in ("gpu_busy_percent", "mem_info_vram_used",
                                                              "mem_info_vram_total")}
    return cards


def request(port, path, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return json.loads(response.read())


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return dict(head=git("rev-parse", "HEAD"), status=git("status", "--short", "--untracked-files=no"),
                diff_stat=git("diff", "HEAD", "--stat"))


def snapshot(folder):
    """Copy ``src/pyhip`` and the plugin into ``folder/source``; returns the PYTHONPATH entries."""
    source = folder / "source"
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(ROOT / "src/pyhip", source / "pyhip", ignore=ignore)
    shutil.copytree(PLUGIN, source / "plugin", ignore=ignore)
    return [str(source / "plugin"), str(source)]


def environment(folder, args):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYHIP_", "SGLANG_USE_PYHIP", "SGLANG_TEST_PYHIP", "FLYDSL_"))}
    visible = args.devices
    cache = folder / "cache"
    env.update(SGLANG_USE_AITER="1", AITER_MOE_PADDING_SIZE="64", HIP_VISIBLE_DEVICES=visible,
               CUDA_VISIBLE_DEVICES=visible, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
               PATH=os.pathsep.join((str(Path(sys.executable).parent), env["PATH"])),
               FLYDSL_RUNTIME_CACHE_DIR=str(cache / "flydsl"), SGLANG_PLUGINS="")
    if args.arm == "mono":
        env.update(PYTHONPATH=os.pathsep.join(item for item in (*snapshot(folder), env.get("PYTHONPATH")) if item),
                   SGLANG_PLUGINS="qwen38_monokernel", PYHIP_MONOKERNEL="1",
                   PYHIP_MONOKERNEL_VALIDATE="1" if args.mode == "validate" else "0")
    return env


def command(args):
    cmd = [str(Path(sys.executable).parent / "sglang"), "serve", "--model-path", str(MODEL),
           "--served-model-name", SERVED, "--host", "127.0.0.1", "--port", str(args.port),
           "--tp-size", str(args.tp), "--attention-backend", "aiter", "--moe-runner-backend", "aiter",
           "--kv-cache-dtype", "auto", "--chunked-prefill-size", "16384", "--watchdog-timeout", "1200",
           "--disable-radix-cache", "--mem-fraction-static", "0.95" if args.tp == 2 else "0.85",
           "--max-running-requests", "32", "--cuda-graph-max-bs-decode", "32", "--no-ple-offload-embedding",
           "--random-seed", "42"]
    if args.mode == "validate":
        cmd.append("--disable-cuda-graph")
    return cmd


def stop(server):
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=60)
    except ProcessLookupError:
        pass


def prompts(count, lengths):
    """Real-text prompts (the model's own README/config text repeated) of the given token lengths."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    text = ""
    for path in sorted(Path("/opt/sglang/docs").rglob("*.md"))[:40]:
        text += path.read_text(errors="ignore")
    ids = tokenizer(text)["input_ids"]
    out = []
    for i in range(count):
        n = lengths[i % len(lengths)]
        start = (i * 7919) % max(1, len(ids) - n)
        out.append(ids[start:start + n])
    return out


def validate(folder, port):
    batch = prompts(4, (900, 2100, 3600, 3000))
    sampling = dict(max_new_tokens=64, temperature=0, ignore_eos=True)
    outputs = {}
    outputs["single"] = request(port, "/generate", dict(input_ids=batch[0], sampling_params=sampling))["output_ids"]
    replies = request(port, "/generate", dict(input_ids=batch, sampling_params=sampling))
    outputs["batch4"] = [reply["output_ids"] for reply in replies]
    (folder / "outputs.json").write_text(json.dumps(outputs) + "\n")
    return outputs


def summarize_validation(log_text):
    """Distribution of the per-call metrics (all layers and ranks) and the worst call per layer."""
    import statistics

    values, worst, rows, mismatched = {}, {}, 0, 0
    for match in VALIDATE_LINE.finditer(log_text):
        report = ast.literal_eval(match.group(1))
        rows += report["rows"]
        mismatched += report.get("route_mismatch_rows", 0)
        entry = worst.setdefault(report["layer"], {})
        for key in ("out", "out_same_route", "normed", "conv", "ssm"):
            if key in report:
                values.setdefault(key, []).append(report[key])
                if report[key]["rel_l2"] > entry.get(key, {}).get("rel_l2", -1.0):
                    entry[key] = report[key]

    def spread(items, name):
        data = sorted(item[name] for item in items)
        return dict(median=statistics.median(data), p99=data[int(0.99 * (len(data) - 1))], worst=data[-1]
                    if name != "ulp1" else data[0])

    summary = {key: dict(calls=len(items), rel_l2=spread(items, "rel_l2"), ulp1=spread(items, "ulp1"))
               for key, items in values.items()}
    return dict(calls=len(values.get("out", [])), rows=rows, route_mismatch_rows=mismatched, metrics=summary,
                worst_per_layer=dict(sorted(worst.items())))


def bench(folder, port, env):
    results = {}
    for concurrency in CONCURRENCIES:
        out = folder / f"bench_c{concurrency}.jsonl"
        cmd = [sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang", "--model", str(MODEL),
               "--dataset-name", "random", "--host", "127.0.0.1", "--port", str(port), "--num-prompts", "32",
               "--random-input", "12000", "--random-output", "350", "--random-range-ratio", "1.0",
               "--max-concurrency", str(concurrency), "--output-file", str(out), "--seed", "42"]
        with (folder / f"bench_c{concurrency}.log").open("x") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        row = json.loads(out.read_text().splitlines()[-1])
        results[concurrency] = {key: row.get(key) for key in (
            "completed", "median_itl_ms", "mean_itl_ms", "median_tpot_ms", "mean_tpot_ms", "median_ttft_ms",
            "output_throughput", "duration")}
        print(f"C{concurrency}: {results[concurrency]}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("native", "mono"), required=True)
    parser.add_argument("--mode", choices=("validate", "bench"), required=True)
    parser.add_argument("--tp", type=int, choices=(2, 4), default=2)
    parser.add_argument("--devices", required=True, help="comma-separated visible GPUs, one per TP rank")
    parser.add_argument("--port", type=int, default=9132)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert len(args.devices.split(",")) == args.tp
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    folder = (args.output or DATA / f"qwen38_mono_p4_{stamp}" / f"{args.arm}_{args.mode}_tp{args.tp}").resolve()
    folder.mkdir(parents=True, exist_ok=False)
    env = environment(folder, args)
    cmd = command(args)
    manifest = dict(arm=args.arm, mode=args.mode, tp=args.tp, devices=args.devices, command=cmd,
                    env={k: env[k] for k in env if k.startswith(("SGLANG_", "AITER_", "PYHIP_", "HIP_", "CUDA_",
                                                                  "FLYDSL_", "PYTHONPATH"))},
                    git=git_state(), gpu_before=gpu_state(), started=datetime.now(timezone.utc).isoformat())
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    server = subprocess.Popen(cmd, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              bufsize=1, start_new_session=True)
    ready = threading.Event()

    def drain():
        with (folder / "server.log").open("x") as stream:
            for line in server.stdout:
                stream.write(line)
                stream.flush()
                if READY in line:
                    ready.set()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        deadline = time.time() + 3600
        while not ready.wait(timeout=5):
            if server.poll() is not None:
                raise RuntimeError(f"server exited before readiness; see {folder / 'server.log'}")
            if time.time() > deadline:
                raise RuntimeError("server not ready within an hour")
        manifest["ready"] = datetime.now(timezone.utc).isoformat()
        smoke = request(args.port, "/v1/chat/completions", dict(
            model=SERVED, messages=[dict(role="user", content="Reply with only the number: what is 2 + 3?")],
            max_tokens=32, temperature=0, chat_template_kwargs=dict(enable_thinking=False)))
        manifest["smoke"] = smoke["choices"][0]["message"]["content"]
        if args.mode == "validate":
            manifest["outputs"] = validate(folder, args.port)
            time.sleep(2)
            manifest["validation"] = summarize_validation((folder / "server.log").read_text())
        else:
            manifest["bench"] = bench(folder, args.port, env)
        manifest["complete"] = True
    finally:
        stop(server)
        reader.join(timeout=30)
        manifest["ended"] = datetime.now(timezone.utc).isoformat()
        manifest["gpu_after"] = gpu_state()
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("smoke", "validation", "bench") if k in manifest}, indent=1,
                     default=str))
    print(f"output: {folder}")


if __name__ == "__main__":
    main()
