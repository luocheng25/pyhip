"""逐 dispatch 记录 H3 attention 性能与 GPU 降频遥测。

默认在 ``HIP_VISIBLE_DEVICES`` 指定的卡上依次测试三个 dense 单段近似和
真实 4-wave varlen H3。每次 dispatch 都保留独立结果，不使用中值。
"""

import json
import hashlib
import importlib.metadata
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

HERE = Path(__file__).resolve().parent
PA4_DIR = HERE / "pa_4wave"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PA4_DIR))

H3_SEGMENTS = (63225, 7)
H3_HEADS = 14
H3_HEAD_DIM = 128
H3_SEQ_LEN = sum(H3_SEGMENTS)
H3_FLOPS = sum(4 * length * length * H3_HEAD_DIM * H3_HEADS for length in H3_SEGMENTS)
DENSE_FLOPS = 4 * H3_SEQ_LEN * H3_SEQ_LEN * H3_HEAD_DIM * H3_HEADS
THROTTLE_COUNTERS = (
    "accumulation_counter",
    "ppt_accumulated",
    "prochot_accumulated",
    "socket_thermal_accumulated",
    "vr_thermal_accumulated",
    "hbm_thermal_accumulated",
)
SOURCE_FILES = (
    HERE / "profile_h3_attention_throttle.py",
    HERE / "analyze_h3_attention_throttle.py",
    HERE / "test_attn_8wave_lkgv.py",
    HERE / "test_attn_8wave_32x32_lkgv.py",
    HERE / "test_attn_gemm.py",
    PA4_DIR / "test_pa_prefill.py",
    PA4_DIR / "pa_prefill_4wave.py",
    HERE.parents[1] / "src" / "contrib" / "flydsl" / "helpers.py",
)


def run_command(command):
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return {"command": command, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"command": command, "error": str(error)}


def run_json_command(command):
    result = run_command(command)
    if "stdout" in result:
        raw = result.pop("stdout")
        try:
            result["json"] = json.loads(raw)
        except json.JSONDecodeError as error:
            result["json_error"] = str(error)
            result["stdout"] = raw
    return result


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return f"unavailable: {error}"


def read_scaled_number(path, divisor):
    try:
        return int(Path(path).read_text()) / divisor
    except (OSError, ValueError):
        return "unavailable"


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def git_state(path):
    root_result = run_command(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    if "stdout" not in root_result:
        return root_result
    root = root_result["stdout"]
    commit = run_command(["git", "-C", root, "rev-parse", "HEAD"])
    status = run_command(
        ["git", "-C", root, "status", "--short", "--untracked-files=no"]
    )
    return {
        "root": root,
        "commit": commit.get("stdout", commit.get("error")),
        "status": status.get("stdout", status.get("error")),
    }


def source_hashes():
    return {
        str(path.relative_to(HERE.parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in SOURCE_FILES
    }


def physical_gpu_index():
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError("set HIP_VISIBLE_DEVICES to exactly one physical GPU")
    devices = visible.split(",")
    if len(devices) != 1:
        raise RuntimeError(f"expected one visible GPU, got {visible!r}")
    try:
        return int(devices[0])
    except ValueError as error:
        raise RuntimeError(f"expected a numeric physical GPU ID, got {visible!r}") from error


def gpu_bdf(physical_gpu):
    result = subprocess.run(
        ["amd-smi", "list", "--json"], check=True, capture_output=True, text=True
    )
    devices = json.loads(result.stdout)
    return next(device["bdf"] for device in devices if device["gpu"] == physical_gpu)


def gpu_runtime_state(physical_gpu, bdf):
    pci_path = Path("/sys/bus/pci/devices") / bdf
    process_result = run_json_command(
        ["amd-smi", "process", "-g", str(physical_gpu), "-G", "--json"]
    )
    try:
        process_list = process_result["json"][0]["process_list"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"cannot read GPU process list: {process_result}") from error
    running_processes = [
        entry["process_info"]
        for entry in process_list
        if isinstance(entry.get("process_info"), dict)
    ]
    return {
        "gpu_busy_percent": int(read_text(pci_path / "gpu_busy_percent")),
        "vram_used_bytes": int(read_text(pci_path / "mem_info_vram_used")),
        "vram_total_bytes": int(read_text(pci_path / "mem_info_vram_total")),
        "dpm_force_performance_level": read_text(
            pci_path / "power_dpm_force_performance_level"
        ),
        "running_processes": running_processes,
        "process_query": process_result,
    }


def require_idle_gpu(physical_gpu, bdf):
    state = gpu_runtime_state(physical_gpu, bdf)
    max_vram_mib = int(os.environ.get("ATTN_PROFILE_MAX_INITIAL_VRAM_MIB", "1024"))
    max_vram_bytes = max_vram_mib * 1024 * 1024
    allow_non_auto = os.environ.get("ATTN_PROFILE_ALLOW_NON_AUTO_DPM", "0") == "1"
    errors = []
    if state["running_processes"]:
        errors.append(f"running GPU processes: {state['running_processes']}")
    if state["gpu_busy_percent"] != 0:
        errors.append(f"gpu_busy_percent={state['gpu_busy_percent']}")
    if state["vram_used_bytes"] > max_vram_bytes:
        errors.append(
            f"initial VRAM={state['vram_used_bytes'] / 2**20:.1f} MiB exceeds "
            f"ATTN_PROFILE_MAX_INITIAL_VRAM_MIB={max_vram_mib}"
        )
    if state["dpm_force_performance_level"] != "auto" and not allow_non_auto:
        errors.append(
            f"DPM level={state['dpm_force_performance_level']!r}; reset to auto or set "
            "ATTN_PROFILE_ALLOW_NON_AUTO_DPM=1 for an explicit determinism experiment"
        )
    if errors:
        raise RuntimeError("GPU preflight failed before CUDA initialization: " + "; ".join(errors))
    state["max_initial_vram_mib"] = max_vram_mib
    state["allow_non_auto_dpm"] = allow_non_auto
    return state


def throttle_counters(physical_gpu):
    result = run_json_command(
        ["amd-smi", "metric", "-g", str(physical_gpu), "-v", "--json"]
    )
    try:
        return result["json"]["gpu_data"][0]["throttle"]
    except (KeyError, IndexError, TypeError):
        return {}


def labeled_sensor(hwmon, family, labels):
    labels = {label.lower() for label in labels}
    for label_path in sorted(hwmon.glob(f"{family}*_label")):
        label = label_path.read_text().strip()
        if label.lower() in labels:
            input_path = label_path.with_name(label_path.name.replace("_label", "_input"))
            if input_path.is_file():
                return input_path, label
    raise RuntimeError(f"cannot find {family} sensor with labels {sorted(labels)} under {hwmon}")


class SensorSampler:
    def __init__(self, bdf, interval=0.01):
        hwmon_dirs = list((Path("/sys/bus/pci/devices") / bdf / "hwmon").glob("hwmon*"))
        if len(hwmon_dirs) != 1:
            raise RuntimeError(f"expected one hwmon directory for {bdf}, got {hwmon_dirs}")
        hwmon = hwmon_dirs[0]
        sclk_path, sclk_label = labeled_sensor(hwmon, "freq", {"sclk", "gfxclk"})
        power_path, power_label = labeled_sensor(hwmon, "power", {"ppt", "socket power"})
        junction_path, junction_label = labeled_sensor(hwmon, "temp", {"junction", "hotspot"})
        mem_path, mem_label = labeled_sensor(hwmon, "temp", {"mem", "memory", "hbm"})
        self.paths = {
            "sclk_mhz": (sclk_path, 1e6),
            "power_w": (power_path, 1e6),
            "junction_c": (junction_path, 1e3),
            "mem_c": (mem_path, 1e3),
        }
        self.metadata = {
            "hwmon": str(hwmon),
            "interval_seconds": interval,
            "sensors": {
                "sclk_mhz": {"path": str(sclk_path), "label": sclk_label, "divisor": 1e6},
                "power_w": {"path": str(power_path), "label": power_label, "divisor": 1e6},
                "junction_c": {
                    "path": str(junction_path),
                    "label": junction_label,
                    "divisor": 1e3,
                },
                "mem_c": {"path": str(mem_path), "label": mem_label, "divisor": 1e3},
            },
            "power_cap_w": read_scaled_number(
                power_path.with_name(power_path.name.replace("_input", "_cap")), 1e6
            ),
        }
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

    def _sample_once(self):
        row = {"time": time.perf_counter()}
        for name, (path, divisor) in self.paths.items():
            row[name] = int(path.read_text()) / divisor
        with self.lock:
            self.samples.append(row)

    def _run(self):
        while not self.stop_event.is_set():
            self._sample_once()
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def between(self, begin, end):
        with self.lock:
            return [sample for sample in self.samples if begin <= sample["time"] <= end]


def summarize_sensors(samples):
    if not samples:
        return {
            "sensor_count": 0,
            "sclk_mean_mhz": None,
            "sclk_min_mhz": None,
            "sclk_max_mhz": None,
            "power_mean_w": None,
            "power_max_w": None,
            "junction_max_c": None,
            "mem_max_c": None,
        }
    return {
        "sensor_count": len(samples),
        "sclk_mean_mhz": statistics.mean(sample["sclk_mhz"] for sample in samples),
        "sclk_min_mhz": min(sample["sclk_mhz"] for sample in samples),
        "sclk_max_mhz": max(sample["sclk_mhz"] for sample in samples),
        "power_mean_w": statistics.mean(sample["power_w"] for sample in samples),
        "power_max_w": max(sample["power_w"] for sample in samples),
        "junction_max_c": max(sample["junction_c"] for sample in samples),
        "mem_max_c": max(sample["mem_c"] for sample in samples),
    }


def profile_dispatches(name, launch, native_flops, physical_gpu, bdf, warmup, iters):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    before = throttle_counters(physical_gpu)
    sensor_interval = float(os.environ.get("ATTN_PROFILE_SENSOR_INTERVAL_MS", "10")) / 1e3
    sampler = SensorSampler(bdf, sensor_interval)
    sampler.start()
    section_start = time.perf_counter()
    dispatches = []

    print(
        "sample,impl,index,elapsed_ms,native_tflops,h3_tflops,"
        "sclk_mean_mhz,sclk_min_mhz,sclk_max_mhz,power_mean_w,power_max_w,"
        "junction_max_c,mem_max_c,sensor_count",
        flush=True,
    )
    for index in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        wall_stop = time.perf_counter()
        elapsed_ms = start.elapsed_time(stop)
        sensors = summarize_sensors(sampler.between(wall_start, wall_stop))
        row = {
            "impl": name,
            "index": index,
            "wall_start_seconds": wall_start - section_start,
            "wall_elapsed_ms": (wall_stop - wall_start) * 1e3,
            "elapsed_ms": elapsed_ms,
            "native_tflops": native_flops / (elapsed_ms * 1e9),
            "h3_tflops": H3_FLOPS / (elapsed_ms * 1e9),
            **sensors,
        }
        dispatches.append(row)
        print(
            f"sample,{name},{index},{elapsed_ms:.3f},{row['native_tflops']:.3f},"
            f"{row['h3_tflops']:.3f},{row['sclk_mean_mhz']:.1f},"
            f"{row['sclk_min_mhz']:.1f},{row['sclk_max_mhz']:.1f},"
            f"{row['power_mean_w']:.1f},{row['power_max_w']:.1f},"
            f"{row['junction_max_c']:.1f},{row['mem_max_c']:.1f},"
            f"{row['sensor_count']}",
            flush=True,
        )

    sampler.stop()
    after = throttle_counters(physical_gpu)
    throttle_delta = {
        key: after.get(key) - before.get(key)
        if isinstance(after.get(key), (int, float)) and isinstance(before.get(key), (int, float))
        else None
        for key in THROTTLE_COUNTERS
    }
    print(f"throttle,{name},{json.dumps(throttle_delta, sort_keys=True)}", flush=True)
    return {
        "name": name,
        "dispatches": dispatches,
        "throttle_before": before,
        "throttle_after": after,
        "throttle_delta": throttle_delta,
        "sensor_metadata": sampler.metadata,
    }


def make_dense_inputs():
    generator = torch.Generator(device="cuda").manual_seed(1101)
    shape = (H3_HEADS, H3_SEQ_LEN, H3_HEAD_DIM)
    q, k, v = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    v_shuffled = (
        v.reshape(H3_HEADS, H3_SEQ_LEN // 8, 8, H3_HEAD_DIM)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    return q, k, v_shuffled


def bind_dense_launch(kernel, q, k, v, output, stream):
    return lambda: kernel(q, k, v, output, stream)


def bind_compiled_launch(kernel, args):
    return lambda: kernel(*args)


def prepare_launchers(selected):
    import test_attn_8wave_32x32_lkgv as attn_8wave_32x32
    import test_attn_8wave_lkgv as attn_8wave_lkgv
    import test_attn_gemm as attn_gemm
    from test_pa_prefill import make_h3_inputs

    stream = torch.cuda.current_stream()
    launchers = {}
    if any(name in selected for name in ("8wave_lkgv", "8wave_32x32", "4wave_dense")):
        q, k, v = make_dense_inputs()

    if "8wave_lkgv" in selected:
        output = torch.empty_like(q)
        kernel = attn_8wave_lkgv.MHA(H3_HEADS, H3_HEAD_DIM, 256, 32)
        kernel(q, k, v, output, stream)
        torch.cuda.synchronize()
        launchers["8wave_lkgv"] = (
            bind_dense_launch(kernel, q, k, v, output, stream),
            DENSE_FLOPS,
        )

    if "8wave_32x32" in selected:
        output = torch.empty_like(q)
        kernel = attn_8wave_32x32.MHA(H3_HEADS, H3_HEAD_DIM, 256, 32)
        kernel(q, k, v, output, stream)
        torch.cuda.synchronize()
        launchers["8wave_32x32"] = (
            bind_dense_launch(kernel, q, k, v, output, stream),
            DENSE_FLOPS,
        )

    if "4wave_dense" in selected:
        output = torch.empty_like(q)
        args = (q, k, v, output, stream)
        kernel = attn_gemm.fly_compiled(
            (H3_SEQ_LEN, H3_SEQ_LEN, H3_HEAD_DIM, 128, 32, H3_HEADS),
            lambda: attn_gemm.build(H3_SEQ_LEN, H3_SEQ_LEN, H3_HEAD_DIM, 128, 32, H=H3_HEADS),
            args,
        )
        torch.cuda.synchronize()
        launchers["4wave_dense"] = (bind_compiled_launch(kernel, args), DENSE_FLOPS)

    if "4wave_varlen" in selected:
        *_, launch = make_h3_inputs()
        launch()
        torch.cuda.synchronize()
        launchers["4wave_varlen"] = (launch, H3_FLOPS)

    return launchers


def collect_environment(physical_gpu, bdf, selected, warmup, iters):
    import aiter
    import flydsl
    import pyhip

    pci_path = Path("/sys/bus/pci/devices") / bdf
    sensor_metadata = SensorSampler(
        bdf, float(os.environ.get("ATTN_PROFILE_SENSOR_INTERVAL_MS", "10")) / 1e3
    ).metadata
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "packages": {
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "flydsl": package_version("flydsl"),
            "pyhip": package_version("pyhip"),
            "numpy": package_version("numpy"),
            "aiter": package_version("aiter"),
            "flydsl_file": str(Path(flydsl.__file__).resolve()),
            "pyhip_paths": [str(Path(path).resolve()) for path in pyhip.__path__],
            "aiter_file": str(Path(aiter.__file__).resolve()),
        },
        "repositories": {
            "pyhip": git_state(HERE.parents[1]),
            "flydsl": git_state(Path(flydsl.__file__).resolve().parents[2]),
            "aiter": git_state(Path(aiter.__file__).resolve().parent),
        },
        "source_sha256": source_hashes(),
        "gpu": {
            "physical_index": physical_gpu,
            "visible_index": torch.cuda.current_device(),
            "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES")
            or os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "arch": getattr(properties, "gcnArchName", "unavailable"),
            "compute_units": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "bdf": bdf,
            "numa_node": read_text(pci_path / "numa_node"),
            "dpm_force_performance_level": read_text(
                pci_path / "power_dpm_force_performance_level"
            ),
            "pp_dpm_sclk": read_text(pci_path / "pp_dpm_sclk"),
            "pp_dpm_mclk": read_text(pci_path / "pp_dpm_mclk"),
            "sensor_metadata": sensor_metadata,
            "amd_smi_list": run_json_command(["amd-smi", "list", "--json"]),
            "amd_smi_static": run_json_command(
                ["amd-smi", "static", "-g", str(physical_gpu), "-a", "-b", "-V", "--json"]
            ),
            "amd_smi_metric_before": run_json_command(
                [
                    "amd-smi",
                    "metric",
                    "-g",
                    str(physical_gpu),
                    "-p",
                    "-c",
                    "-t",
                    "-l",
                    "-v",
                    "--json",
                ]
            ),
        },
        "host": {
            "numa_balancing": read_text("/proc/sys/kernel/numa_balancing"),
            "amd_smi_version": run_command(["amd-smi", "version"]),
            "rocm_smi_version": run_command(["rocm-smi", "--version"]),
        },
        "profile": {
            "selected": selected,
            "warmup": warmup,
            "iters": iters,
            "sensor_interval_ms": float(
                os.environ.get("ATTN_PROFILE_SENSOR_INTERVAL_MS", "10")
            ),
            "h3_segments": H3_SEGMENTS,
            "h3_heads": H3_HEADS,
            "h3_head_dim": H3_HEAD_DIM,
            "h3_flops": H3_FLOPS,
            "dense_sequence_length": H3_SEQ_LEN,
            "dense_flops": DENSE_FLOPS,
        },
    }


def main():
    physical_gpu = physical_gpu_index()
    bdf = gpu_bdf(physical_gpu)
    preflight = require_idle_gpu(physical_gpu, bdf)
    print(
        f"preflight,physical_gpu={physical_gpu},bdf={bdf},busy=0,"
        f"vram_mib={preflight['vram_used_bytes'] / 2**20:.1f},"
        f"processes=0,dpm={preflight['dpm_force_performance_level']}",
        flush=True,
    )
    torch.set_default_device("cuda")
    warmup = int(os.environ.get("ATTN_PROFILE_WARMUP", "3"))
    iters = int(os.environ.get("ATTN_PROFILE_ITERS", "70"))
    selected = os.environ.get(
        "ATTN_PROFILE_IMPLS", "8wave_lkgv,8wave_32x32,4wave_dense,4wave_varlen"
    ).split(",")
    output_path = Path(
        os.environ.get("ATTN_PROFILE_OUTPUT", "/tmp/h3_attention_throttle_profile.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(physical_gpu, bdf, selected, warmup, iters)
    environment["gpu"]["preflight"] = preflight

    print(
        f"config,physical_gpu={physical_gpu},bdf={bdf},warmup={warmup},iters={iters},"
        f"impls={','.join(selected)},h3_flops={H3_FLOPS}",
        flush=True,
    )
    print(
        f"environment,gpu={environment['gpu']['name']},arch={environment['gpu']['arch']},"
        f"power_cap_w={environment['gpu']['sensor_metadata']['power_cap_w']},"
        f"numa_balancing={environment['host']['numa_balancing']}",
        flush=True,
    )
    launchers = prepare_launchers(selected)
    results = []
    for name in selected:
        launch, native_flops = launchers[name]
        results.append(
            profile_dispatches(name, launch, native_flops, physical_gpu, bdf, warmup, iters)
        )
    runtime_after_profile = gpu_runtime_state(physical_gpu, bdf)
    environment["gpu"]["runtime_after_profile"] = runtime_after_profile
    if len(runtime_after_profile["running_processes"]) > 1:
        print(
            f"warning,more than one KFD process visible after profile: "
            f"{runtime_after_profile['running_processes']}",
            flush=True,
        )

    output_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "environment": environment,
                "physical_gpu": physical_gpu,
                "bdf": bdf,
                "warmup": warmup,
                "iters": iters,
                "h3_flops": H3_FLOPS,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"output,{output_path}", flush=True)


if __name__ == "__main__":
    main()