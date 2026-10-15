"""SGLang decode capture for the Qwen3.8 MonoKernel golden check (integration correctness).

Builds a 4-layer view of the checkpoint (layers 0-2 GDN, layer 3 full attention with QSA; PLE and MTP
tensors dropped, shard files symlinked), then per TP size starts one eager SGLang server with AITER
attention/MoE plus --disable-cuda-graph/--skip-server-warmup and the capture plugin (capture_plugin/,
loaded through the sglang.srt.plugins entry point; SGLang source unchanged). Workload: one batched
/generate of four real-text prompts (two longer and two shorter than the 2048-token QSA budget) and
one single request, 3 greedy tokens each. The plugin records the first 4 decode steps on every rank:
3 at S=4 (the overlap scheduler runs one extra step) and 1 at S=1. At the first step it also writes
every layer's parameter and state-pool layouts (layout.json) and raw-bit samples of each parameter
(param_samples.pt), which check_reference compares with the checkpoint in SGLang's served format.
Each TP folder then gets check_reference.json; the exit status is nonzero unless every capture
completes and passes.

    .venv/bin/python benchmarks/monokernel/capture.py --tp 2 4 --devices 0,1,6,7

Output: <output>/{model_4layer, tp<N>/{capture/, server.log, manifest.json, check_reference.json}}.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "mytest/mydata"
PLUGIN = Path(__file__).resolve().parent / "capture_plugin"
MODEL_DIR = Path("/models/Qwen3.8-Flash-Next-PTPC-FP8")
NUM_LAYERS = 4
BATCH_LENGTHS = (3600, 2100, 900, 3000)
SINGLE_LENGTH = 2600
NEW_TOKENS = 3
STEPS = 4
READY = "The server is fired up and ready to roll!"
LAYER_NAME = re.compile(r"model\.language_model\.layers\.(\d+)\.")


def build_model(source: Path, folder: Path) -> None:
    """Symlink the shards that hold the first layers; drop PLE, MTP and later layers from config/index."""

    index = json.loads((source / "model.safetensors.index.json").read_text())

    def keep(name: str) -> bool:
        match = LAYER_NAME.match(name)
        in_range = match is None or int(match.group(1)) < NUM_LAYERS
        return in_range and not name.startswith("mtp.") and ".ple." not in name

    weight_map = {name: shard for name, shard in index["weight_map"].items() if keep(name)}
    folder.mkdir(parents=True)
    for item in source.iterdir():
        if item.name in ("config.json", "model.safetensors.index.json", ".cache"):
            continue
        if item.suffix == ".safetensors" and item.name not in weight_map.values():
            continue
        (folder / item.name).symlink_to(item)
    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    text["num_hidden_layers"] = NUM_LAYERS
    text["layer_types"] = text["layer_types"][:NUM_LAYERS]
    text["ple_layer_ids"] = []
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    index["weight_map"] = weight_map
    (folder / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def prompts(source: Path):
    """Consecutive token slices of the PyHIP docs: the batched prompts and the single prompt."""

    from transformers import AutoTokenizer

    texts = [path.read_text() for path in (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))]
    tokens = AutoTokenizer.from_pretrained(source).encode("\n\n".join(texts))
    lengths = (*BATCH_LENGTHS, SINGLE_LENGTH)
    if len(tokens) < sum(lengths):
        raise RuntimeError(f"need {sum(lengths)} prompt tokens, the docs give {len(tokens)}")
    slices, start = [], 0
    for length in lengths:
        slices.append(tokens[start : start + length])
        start += length
    return slices[: len(BATCH_LENGTHS)], slices[-1]


def physical_devices(visible):
    """Map this process's visible GPU numbers to physical ones (the server gets its own visibility)."""

    mask = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if not mask:
        return tuple(visible)
    ids = [int(item) for item in mask.split(",")]
    return tuple(ids[index] for index in visible)


def gpu_state(devices):
    states = {}
    for card in Path("/sys/class/drm").glob("card[0-9]*"):
        path = card / "device"
        if (path / "gpu_busy_percent").is_file():
            keys = ("gpu_busy_percent", "mem_info_vram_used", "mem_info_vram_total")
            states[path.resolve().name.lower()] = {key: int((path / key).read_text()) for key in keys}
    listing = json.loads(subprocess.check_output(["amd-smi", "list", "--json"], text=True))
    return {row["gpu"]: states[row["bdf"].lower()] for row in listing if row["gpu"] in devices}


def versions():
    import flydsl
    import torch

    heads = {}
    for name, path in (("pyhip", ROOT), ("sglang", Path("/opt/sglang")), ("aiter", Path("/opt/aiter"))):
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        heads[name] = result.stdout.strip() or None
    return dict(flydsl=flydsl.__version__, torch=torch.__version__, git_heads=heads)


def request(port, path, payload, timeout=1800):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def stop(server):
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=60)
    except ProcessLookupError:
        pass


def capture(folder, model, tp, devices, port, batch, single):
    """Run one owned server and write captures; returns the manifest."""

    capture_dir = folder / "capture"
    capture_dir.mkdir(parents=True)
    before = gpu_state(devices)
    # Another session's job on these GPUs would be clobbered (or OOM this server); refuse instead.
    if any(state["mem_info_vram_used"] > 0.1 * state["mem_info_vram_total"] for state in before.values()):
        raise RuntimeError(f"GPUs {devices} are in use: {before}")
    with socket.socket() as probe:
        # The previous TP's server leaves TIME_WAIT sockets; the server itself binds with SO_REUSEADDR.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    visible = ",".join(map(str, devices))
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYHIP_", "SGLANG_USE_PYHIP", "SGLANG_TEST_PYHIP"))
    }
    env.update(
        SGLANG_USE_AITER="1",
        AITER_MOE_PADDING_SIZE="64",
        HIP_VISIBLE_DEVICES=visible,
        CUDA_VISIBLE_DEVICES=visible,
        PYTHONPATH=os.pathsep.join(item for item in (str(PLUGIN), env.get("PYTHONPATH")) if item),
        SGLANG_PLUGINS="qwen38_capture",
        QWEN38_CAPTURE_DIR=str(capture_dir),
        QWEN38_CAPTURE_STEPS=str(STEPS),
        PATH=os.pathsep.join((str(Path(sys.executable).parent), env["PATH"])),
    )
    command = [
        str(Path(sys.executable).parent / "sglang"), "serve", "--model-path", str(model),
        "--served-model-name", "qwen38-4layer", "--host", "127.0.0.1", "--port", str(port),
        "--tp-size", str(tp), "--attention-backend", "aiter", "--moe-runner-backend", "aiter",
        "--kv-cache-dtype", "auto", "--chunked-prefill-size", "16384", "--watchdog-timeout", "1200",
        "--disable-radix-cache", "--mem-fraction-static", "0.5", "--max-running-requests", "32",
        "--no-ple-offload-embedding", "--disable-cuda-graph", "--skip-server-warmup", "--random-seed", "42",
    ]
    manifest = dict(
        tp=tp,
        devices=devices,
        command=command,
        gpu_before=before,
        env={key: env[key] for key in env if key.startswith(("SGLANG_", "AITER_", "QWEN38_", "HIP_", "CUDA_"))},
        versions=versions(),
        batch_lengths=[len(prompt) for prompt in batch],
        single_length=len(single),
        new_tokens=NEW_TOKENS,
        started=datetime.now(timezone.utc).isoformat(),
    )
    server = subprocess.Popen(
        command,
        cwd=folder,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
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
        sampling = dict(max_new_tokens=NEW_TOKENS, temperature=0, ignore_eos=True)
        replies = request(port, "/generate", dict(input_ids=batch, sampling_params=sampling))
        manifest["batch_outputs"] = [reply["output_ids"] for reply in replies]
        reply = request(port, "/generate", dict(input_ids=single, sampling_params=sampling))
        manifest["single_output"] = reply["output_ids"]
        files = sorted(path.name for path in capture_dir.iterdir())
        expected = sorted(
            f"tp{tp}_rank{rank}_{name}"
            for rank in range(tp)
            for name in (*(f"step{step:03d}.pt" for step in range(STEPS)), "layout.json", "param_samples.pt")
        )
        manifest["capture_files"] = files
        manifest["complete"] = files == expected
    finally:
        stop(server)
        reader.join(timeout=30)
        try:
            os.killpg(server.pid, 0)
            manifest["owned_group_gone"] = False
        except ProcessLookupError:
            manifest["owned_group_gone"] = True
        manifest["ended"] = datetime.now(timezone.utc).isoformat()
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp", type=int, nargs="+", default=[2, 4], choices=(2, 4, 8))
    parser.add_argument("--devices", help="comma-separated visible GPU numbers; TP N uses the first N")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, help="new directory (default: a new one under mytest/mydata)")
    parser.add_argument("--port", type=int, default=9131)
    parser.add_argument("--no-check", action="store_true", help="capture only, skip check_reference")
    args = parser.parse_args()
    visible = tuple(map(int, args.devices.split(","))) if args.devices else tuple(range(max(args.tp)))
    if len(visible) < max(args.tp):
        parser.error(f"--devices lists {len(visible)} GPUs, TP{max(args.tp)} needs {max(args.tp)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = (args.output or DATA / f"monokernel_qwen38_capture_{stamp}_{os.getpid()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    build_model(args.model_dir, output / "model_4layer")
    batch, single = prompts(args.model_dir)
    passed = True
    for tp in args.tp:
        folder = output / f"tp{tp}"
        manifest = capture(
            folder, output / "model_4layer", tp, physical_devices(visible[:tp]), args.port, batch, single
        )
        status = f"tp{tp} complete={manifest.get('complete')} files={len(manifest.get('capture_files', []))}"
        passed &= bool(manifest.get("complete"))
        if manifest.get("complete") and not args.no_check:
            import torch

            from benchmarks.monokernel.check_reference import run

            with torch.no_grad():
                report = run(folder / "capture", args.model_dir, torch.device("cuda", visible[0]))
            (folder / "check_reference.json").write_text(json.dumps(report, indent=1) + "\n")
            status += " check=" + ("PASS" if report["passed"] else "FAIL: " + "; ".join(report["failures"]))
            passed &= report["passed"]
        print(status, flush=True)
    print(f"output: {output}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
