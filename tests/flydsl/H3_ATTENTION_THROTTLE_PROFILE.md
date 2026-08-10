# H3 Attention 逐 Dispatch 降频记录

日期：2026-08-10

## 跨机器复现手册

### 1. 测试目标与不可改变项

该测试不是常规吞吐 benchmark，而是观察连续 dispatch 中的 DPM/PPT 循环。正式结果必须保留每次测量的
原始顺序，不得排序、取最小值、取中值或只报告一个汇总 TFLOPS。

默认实现顺序固定为：

1. `8wave_lkgv`
2. `8wave_32x32`
3. `4wave_dense`
4. `4wave_varlen`

采集器会先构造输入并编译所有选中实现，再对每个实现分别预热 3 次和连续采样 70 次。CUDA event 只覆盖
kernel dispatch；sysfs 采样线程不进入 event 时间。输入固定为 BF16、seed `1101`：

- dense：`H=14,D=128,M=N=63232`，V 使用 `[H,N/8,D,8]` preshuffle；
- varlen：真实 segments `(63225,7)`，`Hq=Hkv=14,Dq=Dv=128`；
- H3 FLOPs：`sum(4*S_i^2*D*H)=28,653,368,031,232`；
- dense 比真实 varlen 多算跨 segment attention，native FLOPs 高 0.022143%；JSON 同时保留
  `native_tflops` 和按真实 H3 FLOPs 归一的 `h3_tflops`。

### 2. 必须迁移的源码

先在目标机检出完整 PyHIP 仓库，再覆盖以下同版本文件。只复制 profiler 不够，因为本次三个 dense harness
和 H3 varlen 入口包含尚未进入基准提交的改动。

```text
src/contrib/flydsl/helpers.py
tests/flydsl/profile_h3_attention_throttle.py
tests/flydsl/analyze_h3_attention_throttle.py
tests/flydsl/test_attn_8wave_lkgv.py
tests/flydsl/test_attn_8wave_32x32_lkgv.py
tests/flydsl/test_attn_gemm.py
tests/flydsl/pa_4wave/test_pa_prefill.py
tests/flydsl/pa_4wave/pa_prefill_4wave.py
```

源机器可从 PyHIP 仓库根目录打包：

```bash
tar -czf h3-attention-repro-src.tgz \
  src/contrib/flydsl/helpers.py \
  tests/flydsl/profile_h3_attention_throttle.py \
  tests/flydsl/analyze_h3_attention_throttle.py \
  tests/flydsl/test_attn_8wave_lkgv.py \
  tests/flydsl/test_attn_8wave_32x32_lkgv.py \
  tests/flydsl/test_attn_gemm.py \
  tests/flydsl/pa_4wave/test_pa_prefill.py \
  tests/flydsl/pa_4wave/pa_prefill_4wave.py
sha256sum h3-attention-repro-src.tgz > h3-attention-repro-src.tgz.sha256
```

目标机在同一个 PyHIP 仓库根目录解包。正式 JSON 的 `environment.source_sha256` 会再次记录这 8 个文件
的内容 hash；跨机器比较前必须确认 hash 一致。

本次参考环境如下。迁移时不要求安装路径相同，但版本、提交和 GPU 架构差异必须保留在结果中：

| 项目 | 本次参考值 |
|---|---|
| PyHIP base commit | `974061a00c61bf8e39183a62f8b6c9dc20e11e91`，另加上表工作区文件 |
| FlyDSL | `0.3.0.dev765`，commit `950bed539f5225c2502eb6062bc9ce7cfcf7ccf5` |
| AITER | commit `d19f33251400bbc21a3a49cc8db421f918716b93` |
| Python | 3.10.12 |
| PyTorch | `2.13.0+rocm7.2`，HIP `7.2.53211` |
| ROCm / AMDSMI | ROCm 7.2.0 / AMDSMI 26.2.1 |
| GPU | AMD Instinct MI308X，`gfx942`，80 CU，约 192GiB，650W cap |
| 容器 / kernel | Ubuntu 22.04.5 / Linux `5.10.134-18.al8.x86_64` |
| NUMA | GPU 4 在 node 1；`kernel.numa_balancing=1`；本次未使用 `numactl` |

### 3. Python 与运行时准备

以下所有命令从 PyHIP 仓库根目录执行。用实际环境替换 `PYTHON`；不要依赖 shell 中偶然出现的
`python3`。PyHIP 建议 editable 安装，以保证导入刚迁移的源码：

```bash
export PYTHON=/path/to/rocm-python/bin/python
"$PYTHON" -m pip install -e . --no-deps
"$PYTHON" - <<'PY'
import torch, flydsl, pyhip, aiter
print("torch", torch.__version__, "HIP", torch.version.hip)
print("gpu", torch.cuda.get_device_name(0))
print("flydsl", flydsl.__file__)
print("pyhip", list(pyhip.__path__))
print("aiter", aiter.__file__)
PY
```

必须满足：

- `torch` 是 ROCm build，`torch.cuda.is_available()` 为真；
- FlyDSL C++/MLIR Python binding 已为目标 ROCm/GPU 构建；
- AITER 的 Python 包和 `module_aiter_core.so` 可导入；
- `amd-smi`、`rocm-smi` 在 `PATH` 中；
- 进程可读取 `/sys/bus/pci/devices/<BDF>/hwmon/hwmon*/`；
- hwmon 至少提供标签为 `sclk`/`gfxclk`、`PPT`/`socket power`、
  `junction`/`hotspot`、`mem`/`memory`/`hbm` 的传感器。

profiler 会通过 `amd-smi list --json` 自动把 `HIP_VISIBLE_DEVICES` 的物理 GPU 映射到 BDF，再按标签发现
sysfs 节点，因此迁移后不需要沿用本机的 GPU 4、`0001:0b:00.0` 或 `hwmon4`。

### 4. 选择真正空闲的 GPU

先检查所有卡，再设置 `GPU`。不能把瞬时 util=0 但有 KFD PID 或大量 VRAM 常驻的卡当作空闲卡。

```bash
amd-smi list
rocm-smi --showuse --showmemuse --showpids --showclocks
```

选择后再次检查该卡。下面以物理 GPU 4 为例：

```bash
export GPU=4
export BDF=$(GPU="$GPU" "$PYTHON" - <<'PY'
import json, os, subprocess
devices = json.loads(subprocess.check_output(["amd-smi", "list", "--json"], text=True))
print(next(device["bdf"] for device in devices if device["gpu"] == int(os.environ["GPU"])))
PY
)
echo "GPU=$GPU BDF=$BDF"
rocm-smi --showuse --showmemuse --showpids --showclocks
cat "/sys/bus/pci/devices/$BDF/gpu_busy_percent"
cat "/sys/bus/pci/devices/$BDF/mem_info_vram_used"
cat "/sys/bus/pci/devices/$BDF/power_dpm_force_performance_level"
cat /proc/sys/kernel/numa_balancing
cat "/sys/bus/pci/devices/$BDF/numa_node"
```

正式开始条件：目标卡 `GPU use=0%`、rocm-smi VRAM 显示 0%（约数百 MB driver reserve 可接受）、
`No KFD PIDs currently running`，且 `power_dpm_force_performance_level=auto`。测试期间若出现外部 KFD PID，
整轮作废并重跑。

profiler 会在导入 kernel 模块和建立 CUDA context 之前重复该检查，并在不满足以下条件时直接失败：

- `HIP_VISIBLE_DEVICES` 必须是单个数值型物理 GPU ID；
- AMD SMI 进程列表必须为空；
- `gpu_busy_percent` 必须为 0；
- 初始 VRAM 必须不超过 `ATTN_PROFILE_MAX_INITIAL_VRAM_MIB`，默认 1024MiB；
- DPM 必须为 `auto`，除非显式设置 `ATTN_PROFILE_ALLOW_NON_AUTO_DPM=1`。

正常日志应在任何 AITER/FlyDSL 编译输出之前出现：

```text
preflight,physical_gpu=4,bdf=0001:0b:00.0,busy=0,vram_mib=283.8,processes=0,dpm=auto
```

默认 1GiB 门槛用于容纳驱动保留显存，不代表 1GiB 外部任务可接受。若目标机空闲驱动基线明显更低，可收紧：

```bash
export ATTN_PROFILE_MAX_INITIAL_VRAM_MIB=512
```

基线必须使用默认自动 DPM。若上一轮启用了 performance determinism，先在有权限的终端恢复并重新确认
`auto`：

```bash
sudo amd-smi reset -g "$GPU" -d
```

不要在基线前改变 power cap、风扇、mclk/fclk 或 performance profile。

### 5. 两次 smoke test

先验证传感器与 JSON schema，避免跑完 280 次才发现 sysfs 不可读：

```bash
export OUT="artifacts/h3-throttle-$(hostname)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

HIP_VISIBLE_DEVICES="$GPU" \
ATTN_PROFILE_IMPLS=4wave_varlen \
ATTN_PROFILE_WARMUP=1 \
ATTN_PROFILE_ITERS=2 \
ATTN_PROFILE_OUTPUT="$OUT/smoke-varlen.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$OUT/pycache-smoke" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$PYTHON" -B tests/flydsl/profile_h3_attention_throttle.py \
  2>&1 | tee "$OUT/smoke-varlen.log"
```

再让四个实现各跑 2 次，验证所有 kernel 都能编译和独立派发：

```bash
HIP_VISIBLE_DEVICES="$GPU" \
ATTN_PROFILE_IMPLS=8wave_lkgv,8wave_32x32,4wave_dense,4wave_varlen \
ATTN_PROFILE_WARMUP=1 \
ATTN_PROFILE_ITERS=2 \
ATTN_PROFILE_OUTPUT="$OUT/smoke-all.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$OUT/pycache-smoke-all" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$PYTHON" -B tests/flydsl/profile_h3_attention_throttle.py \
  2>&1 | tee "$OUT/smoke-all.log"
```

每轮都应看到 `output,<JSON path>`。当前 AITER/PyTorch 组合可能在该行之后打印
`torch.library._clear_torch_ops_cache` weakref 清理异常；只要 JSON 已完整写盘、样本数正确且异常发生在
`output` 之后，它不影响已完成的 GPU event 测量。若异常发生在 `output` 之前，则该轮失败。

### 6. 正式 70 次逐 dispatch 采集

smoke 后重新确认 GPU 空闲，再执行固定顺序的正式测试：

```bash
rocm-smi --showuse --showmemuse --showpids --showclocks

set -o pipefail
HIP_VISIBLE_DEVICES="$GPU" \
ATTN_PROFILE_IMPLS=8wave_lkgv,8wave_32x32,4wave_dense,4wave_varlen \
ATTN_PROFILE_WARMUP=3 \
ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_OUTPUT="$OUT/profile-auto.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$OUT/pycache-formal" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$PYTHON" -B tests/flydsl/profile_h3_attention_throttle.py \
  2>&1 | tee "$OUT/profile-auto.log"
```

环境变量含义：

| 变量 | 正式值 | 含义 |
|---|---:|---|
| `HIP_VISIBLE_DEVICES` | 选中的物理 GPU | 进程内映射为 `cuda:0`，JSON 仍记录物理 ID/BDF |
| `ATTN_PROFILE_IMPLS` | 上述固定四项顺序 | 可选子集，但跨机器比较必须顺序一致 |
| `ATTN_PROFILE_WARMUP` | 3 | 每个实现单独预热次数 |
| `ATTN_PROFILE_ITERS` | 70 | 每实现连续 dispatch 数，不聚合 |
| `ATTN_PROFILE_SENSOR_INTERVAL_MS` | 10 | sysfs 轮询周期；约 120-183ms kernel 可取得 11-18 点 |
| `ATTN_PROFILE_MAX_INITIAL_VRAM_MIB` | 1024 | CUDA 初始化前允许的驱动保留显存上限 |
| `ATTN_PROFILE_ALLOW_NON_AUTO_DPM` | 0 | 基线拒绝非 `auto` DPM；仅固定时钟对照设为 1 |
| `ATTN_PROFILE_OUTPUT` | 独立 JSON 路径 | 自描述原始数据 |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | 0 | 避免迁移后误用旧 JIT disk cache |
| `PYTHONPYCACHEPREFIX` | 每轮独立目录 | 隔离 Python bytecode 缓存 |

`profile-auto.json` schema v2 自动保存：

- Python/Torch/HIP/FlyDSL/PyHIP/AITER/Numpy 版本和 import 路径；
- PyHIP、FlyDSL、AITER git commit 与 tracked dirty status；
- 8 个关键源码文件的 SHA256；
- GPU 型号、arch、CU、显存、物理/可见编号、BDF、NUMA node；
- DPM level、sclk/mclk 表、PPT cap、传感器路径与标签；
- CUDA 初始化前的 busy/VRAM/KFD process/DPM preflight，以及测试后的 runtime state；
- AMDSMI/ROCm-SMI 版本、GPU static/metric 快照、NUMA balancing；
- 每次 dispatch 的 event 延迟、wall time、native/H3 TFLOPS、sclk/power/温度；
- 每个实现测量前后的 PPT/PROCHOT/thermal throttle counter 及增量。

### 7. JSON 完整性验收

正式运行后立即验证，不能只看终端最后几行：

```bash
PROFILE="$OUT/profile-auto.json" "$PYTHON" - <<'PY'
import json, os
path = os.environ["PROFILE"]
data = json.load(open(path))
assert data["schema_version"] == 2
assert data["warmup"] == 3 and data["iters"] == 70
expected = ["8wave_lkgv", "8wave_32x32", "4wave_dense", "4wave_varlen"]
assert [result["name"] for result in data["results"]] == expected
for result in data["results"]:
    assert len(result["dispatches"]) == 70
    assert [row["index"] for row in result["dispatches"]] == list(range(70))
    assert all(row["sensor_count"] > 0 for row in result["dispatches"])
    assert all(row["elapsed_ms"] > 0 and row["h3_tflops"] > 0 for row in result["dispatches"])
print("validated", path, "samples", sum(len(r["dispatches"]) for r in data["results"]))
print("gpu", data["environment"]["gpu"]["name"], data["environment"]["gpu"]["arch"])
print("source_sha256", json.dumps(data["environment"]["source_sha256"], indent=2))
PY

sha256sum "$OUT/profile-auto.json" "$OUT/profile-auto.log" > "$OUT/SHA256SUMS"
rocm-smi --showuse --showmemuse --showpids --showclocks | tee "$OUT/gpu-after.txt"
```

### 8. 离线分析：不使用中值

分析器默认先逐条打印所有样本，再打印峰值、谷值、峰谷降幅、TFLOPS-sclk/power 相关性、高吞吐区间和
相邻 burst 的 dispatch/时间间隔。高吞吐阈值为
`min + 0.65 * (max - min)`；至少出现 4 个 burst 且相邻 dispatch 间隔跨度不超过 2 时，输出
`cycle_detected=True`。

```bash
"$PYTHON" tests/flydsl/analyze_h3_attention_throttle.py \
  "$OUT/profile-auto.json" \
  --analysis-json "$OUT/analysis-auto.json" \
  | tee "$OUT/analysis-auto.log"
```

需要短输出时可加 `--no-samples`，但原始 JSON 和正式归档日志仍必须保留全部逐次数据。分析器不计算中值。

### 9. 跨机器逐 dispatch 对比

把源机器 `profile-auto.json` 作为 baseline，目标机器同方法生成 candidate。先比较元数据：

```bash
BASE=/path/to/source-machine/profile-auto.json \
CAND=/path/to/target-machine/profile-auto.json \
"$PYTHON" - <<'PY'
import json, os
base = json.load(open(os.environ["BASE"]))
cand = json.load(open(os.environ["CAND"]))
for key in ("selected", "warmup", "iters", "sensor_interval_ms", "h3_segments",
            "h3_heads", "h3_head_dim", "h3_flops", "dense_sequence_length", "dense_flops"):
    assert base["environment"]["profile"][key] == cand["environment"]["profile"][key], key
assert base["environment"]["source_sha256"] == cand["environment"]["source_sha256"]
print("protocol and source hashes match")
print("baseline GPU", base["environment"]["gpu"]["name"], base["environment"]["gpu"]["arch"])
print("candidate GPU", cand["environment"]["gpu"]["name"], cand["environment"]["gpu"]["arch"])
PY
```

然后按实现和原始 index 一一比较 280 条记录：

```bash
"$PYTHON" tests/flydsl/analyze_h3_attention_throttle.py \
  "$BASE" --no-samples \
  --compare "$CAND" \
  --comparison-json "$OUT/source-vs-target-by-dispatch.json" \
  | tee "$OUT/source-vs-target-by-dispatch.log"
```

输出包含每个 dispatch 的 baseline/candidate 延迟、TFLOPS 和比值，不做平均或中值。自动 DPM 循环相位可能
在两机间偏移，因此 index-to-index 比值需结合各自的 sclk/power 序列解释；不能把某机快档和另一机慢档的
单个比值当作 kernel 固有差异。

### 10. 可选：performance determinism 对照

该步骤只用于验证循环是否由自动 DPM/PPT 引起，不能替代默认 auto-DPM 基线。需要管理员权限。两机必须用
相同的、各自都支持的 SCLK softmax limit；本机对照使用 1300MHz。

```bash
cleanup_dpm() { sudo amd-smi reset -g "$GPU" -d; }
trap cleanup_dpm EXIT
sudo amd-smi set -g "$GPU" -d 1300
amd-smi metric -g "$GPU" -l -c -p -t

HIP_VISIBLE_DEVICES="$GPU" \
ATTN_PROFILE_IMPLS=8wave_lkgv,8wave_32x32,4wave_dense,4wave_varlen \
ATTN_PROFILE_WARMUP=3 ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_ALLOW_NON_AUTO_DPM=1 \
ATTN_PROFILE_OUTPUT="$OUT/profile-determinism-1300.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$OUT/pycache-determinism-1300" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$PYTHON" -B tests/flydsl/profile_h3_attention_throttle.py \
  2>&1 | tee "$OUT/profile-determinism-1300.log"

cleanup_dpm
trap - EXIT
cat "/sys/bus/pci/devices/$BDF/power_dpm_force_performance_level"
```

最后一行必须恢复为 `auto`。如果固定 determinism 后 `cycle_detected=False`、sclk 方差显著下降且
PPT/thermal counter 不再随同样模式增长，可进一步支持 DPM/PPT 循环解释。固定 1300MHz 的绝对 TFLOPS
不能和 auto-DPM 峰值直接比较。

### 11. 常见失败与判定

- `ModuleNotFoundError: pyhip.contrib`：不要只设置 `PYTHONPATH=.../src`；按 `pyproject.toml` 做 editable
  install。
- `torch` 不可导入或不是 ROCm build：选错 Python 环境；停止测试。
- `cannot find ... sensor`：目标驱动的 hwmon 标签不同。先记录全部 `*_label`，再扩充 profiler 的标签集合；
  不要按 `freq1/temp2` 编号硬猜。
- `sensor_count=0`：10ms 轮询未覆盖 dispatch，降低 `ATTN_PROFILE_SENSOR_INTERVAL_MS` 后整轮重跑。
- JSON 少于 4 个结果或任一实现少于 70 条：整轮失败。
- 测试中出现外部 KFD PID、GPU util 或 VRAM 突然增加：整轮失败。
- `ppt_accumulated` 增长但 thermal counters 为 0：倾向 PPT/DPM；若 thermal counter 增长，则必须同时检查
  温度、风扇和机箱环境，不能沿用本机结论。
- `torch.library._clear_torch_ops_cache` weakref 异常：仅当发生在 `output,<path>` 之后且 JSON 验收通过时
  可忽略；否则视为失败。
- 不同架构、CU 数、power cap、固件、ROCm、Torch、FlyDSL/AITER commit 或源码 SHA：结果仍可观察，
  但必须标注为跨环境对比，不能声称严格复现。

## 测试口径

- 物理 GPU 4，BDF `0001:0b:00.0`，测试前后均为 `GPU use=0%`、`VRAM=0%`、`No KFD PIDs`。
- GPU 固定读回 `fclk=1300MHz`、`mclk=900MHz`；板卡功耗上限为 650W。
- 每个实现预热 3 次，随后连续测量 70 次；每次单独记录 CUDA-event 延迟和 TFLOPS，不使用中值。
- 每 10ms 从 sysfs 采样执行期 `sclk`、PPT 功耗、junction 温度和 HBM 温度。
- dense 实现使用 `H=14,D=128,M=N=63232` 单段近似；TFLOPS 按真实 H3 两段
  `(63225,7)` 的 `28.653368031232 TFLOP` 归一。
- 真实 4-wave paged varlen 直接执行 `(63225,7)`。
- 复现脚本：`profile_h3_attention_throttle.py`。
- 本次有效原始产物：`/tmp/h3_attention_throttle_profile.json` 和
  `/tmp/h3_attention_throttle_profile.log`。
- 第一轮 profiler 存在 Python lambda 晚绑定问题，dense 实现身份不可信，已作废；下表仅来自修复后重采数据。

运行命令：

```bash
HIP_VISIBLE_DEVICES=4 ATTN_PROFILE_ITERS=70 \
  FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  python3 -B tests/flydsl/profile_h3_attention_throttle.py
```

## 结论

存在稳定且严重的循环降频。四个实现都约每 6-7 个 dispatch 出现一次高吞吐 burst，随后立即跌入低档，
再逐步恢复；相邻 burst 的实际时间间隔约为 0.96-1.11 秒。峰谷吞吐下降 33%-37%。

吞吐与执行期 `sclk` 的相关系数为 0.76-0.80，与功耗的相关系数为 0.66-0.71。
所有实现的 PROCHOT、socket thermal、VR thermal 和 HBM thermal 累计值均为 0；junction 最高 77C，
不是温度保护。只有 PPT 累计值增长，其中 32x32 8-wave 和真实 varlen 4-wave 最明显，说明主要是功率/DPM
控制引起的频率振荡。

| 实现 | 峰值 | 谷值 | 峰谷降幅 | TFLOPS-sclk 相关性 | TFLOPS-power 相关性 | PPT 累计增量 |
|---|---:|---:|---:|---:|---:|---:|
| 8-wave LKGV | 239.910T @ 119.434ms | 156.420T @ 183.182ms | 34.80% | 0.7731 | 0.6764 | 3 |
| 8-wave 32x32 LKGV | 255.545T @ 112.127ms | 159.789T @ 179.320ms | 37.47% | 0.7636 | 0.6623 | 1291 |
| 4-wave dense GEMM | 240.467T @ 119.157ms | 156.814T @ 182.722ms | 34.79% | 0.7916 | 0.6859 | 22 |
| 4-wave paged varlen | 235.015T @ 121.921ms | 157.495T @ 181.931ms | 32.98% | 0.8011 | 0.7085 | 1809 |

高吞吐 burst 起点：

- 8-wave LKGV：`0,6,13,19,25,31,37,44,50,56,62,68`
- 8-wave 32x32 LKGV：`0,7,13,19,26,32,38,45,51,57,64`
- 4-wave dense GEMM：`0,6,13,19,25,31,38,44,50,56,62,69`
- 4-wave paged varlen：`0,6,13,19,25,31,38,44,50,56,63,69`

高低档遥测特征：

| 实现 | 高档平均 sclk / power | 低档平均 sclk / power | 峰值功耗 | 最高 junction |
|---|---:|---:|---:|---:|
| 8-wave LKGV | 1558MHz / 483W | 1252MHz / 389W | 564W | 68C |
| 8-wave 32x32 LKGV | 1488MHz / 515W | 1168MHz / 406W | 602W | 71C |
| 4-wave dense GEMM | 1559MHz / 496W | 1238MHz / 395W | 583W | 71C |
| 4-wave paged varlen | 1538MHz / 550W | 1220MHz / 433W | 624W | 77C |

## TODO

- [ ] 将`mha-fp8-d192`参考分支的BF16 D128 `TRANS/DS_WRITE`精确交织移植到
  `pa_4wave/pa_prefill_4wave.py`，同时保留当前raw-max优化。目标顺序为
  `TRANS(3) -> DS_WRITE(1) -> TRANS(4) -> DS_WRITE(1) -> TRANS(10)`，不改变完整K copy、
  8-wave源码或外部接口。
  - 真实H3近似shape `Hq=Hkv=14,D=128,M=N=63232`的定频基线：1100MHz下参考dense
    `183.889ms`、当前paged `185.236ms`，paged慢`0.73%`；1300MHz下慢约`0.76%`。
    Auto-DPM中观察到的约`4.64%`中值差主要来自PPT/DPM相位，不作为kernel验收依据。
  - 正确性要求：与当前paged输出逐元素对照保持`max_abs <= 2.44e-4`，并通过现有
    BF16 D128 ragged、batch和causal回归。
  - 性能要求：在1100MHz或1300MHz performance determinism下使用同进程
    `control -> candidate -> candidate -> control`夹心测试；候选相对当前paged必须稳定获胜，
    且资源保持2 waves/SIMD、无spill。
  - ATT要求：去除首尾10%后resident-wave反相率从当前约`88.8%`提升到至少`93%`，同时
    MFMA和barrier归一stall下降。现有H14 ATT有trace丢包，只能作方向性参考；正式验收需重采
    无截断trace，不能直接比较当前两份trace的绝对总stall。

## 全部逐次 TFLOPS

以下均为真实 H3 FLOPs 归一值，不作排序、截断或中值聚合。

| idx | 8w LKGV | 8w 32x32 | 4w dense | 4w varlen |
|---:|---:|---:|---:|---:|
| 0 | 239.9 | 255.5 | 237.7 | 235.0 |
| 1 | 233.8 | 238.6 | 237.9 | 234.0 |
| 2 | 158.0 | 160.7 | 158.1 | 164.1 |
| 3 | 157.2 | 160.5 | 157.0 | 157.5 |
| 4 | 156.4 | 160.4 | 156.9 | 157.8 |
| 5 | 157.2 | 160.5 | 157.1 | 157.8 |
| 6 | 220.6 | 215.1 | 216.7 | 211.1 |
| 7 | 238.1 | 255.3 | 240.5 | 233.3 |
| 8 | 163.9 | 173.5 | 166.2 | 176.5 |
| 9 | 156.7 | 160.5 | 157.6 | 159.2 |
| 10 | 158.3 | 160.5 | 157.0 | 158.0 |
| 11 | 156.9 | 160.8 | 157.5 | 158.0 |
| 12 | 204.8 | 185.3 | 199.7 | 192.6 |
| 13 | 238.1 | 255.3 | 232.0 | 232.5 |
| 14 | 172.3 | 198.7 | 183.3 | 192.9 |
| 15 | 158.4 | 160.5 | 157.1 | 158.6 |
| 16 | 157.9 | 160.2 | 157.2 | 158.2 |
| 17 | 157.7 | 161.3 | 157.4 | 158.0 |
| 18 | 190.0 | 166.3 | 186.0 | 177.6 |
| 19 | 238.1 | 248.2 | 231.5 | 232.2 |
| 20 | 183.4 | 230.0 | 197.2 | 211.5 |
| 21 | 158.4 | 160.5 | 157.0 | 158.3 |
| 22 | 158.7 | 162.1 | 157.6 | 158.4 |
| 23 | 157.2 | 160.8 | 157.3 | 158.0 |
| 24 | 178.3 | 165.9 | 174.3 | 165.3 |
| 25 | 238.3 | 210.7 | 230.2 | 230.3 |
| 26 | 196.8 | 255.1 | 213.4 | 230.8 |
| 27 | 158.0 | 169.2 | 157.0 | 160.1 |
| 28 | 158.2 | 161.4 | 157.6 | 158.1 |
| 29 | 157.4 | 161.3 | 157.0 | 157.7 |
| 30 | 169.9 | 165.2 | 167.5 | 162.4 |
| 31 | 234.0 | 184.0 | 222.6 | 214.5 |
| 32 | 211.6 | 252.5 | 232.3 | 231.5 |
| 33 | 157.9 | 193.3 | 157.7 | 171.9 |
| 34 | 158.5 | 161.2 | 157.1 | 158.0 |
| 35 | 157.4 | 159.8 | 156.9 | 157.8 |
| 36 | 167.6 | 167.4 | 167.4 | 162.2 |
| 37 | 218.9 | 167.3 | 206.5 | 195.9 |
| 38 | 228.1 | 240.7 | 238.2 | 231.4 |
| 39 | 159.0 | 224.5 | 164.5 | 185.4 |
| 40 | 158.5 | 161.4 | 157.0 | 158.6 |
| 41 | 157.1 | 160.5 | 157.1 | 157.7 |
| 42 | 168.1 | 166.7 | 166.7 | 162.1 |
| 43 | 202.4 | 165.9 | 194.9 | 180.7 |
| 44 | 237.8 | 207.8 | 233.5 | 231.0 |
| 45 | 163.7 | 253.3 | 175.3 | 200.7 |
| 46 | 158.1 | 166.8 | 156.9 | 159.4 |
| 47 | 157.8 | 160.6 | 157.7 | 157.8 |
| 48 | 167.5 | 166.5 | 164.6 | 161.7 |
| 49 | 188.9 | 166.0 | 185.4 | 168.8 |
| 50 | 238.1 | 182.7 | 230.3 | 229.1 |
| 51 | 173.8 | 249.1 | 188.0 | 219.8 |
| 52 | 158.8 | 189.9 | 156.8 | 159.5 |
| 53 | 157.4 | 160.4 | 158.2 | 158.1 |
| 54 | 166.8 | 166.1 | 164.6 | 161.6 |
| 55 | 179.1 | 166.4 | 175.2 | 163.5 |
| 56 | 235.2 | 168.0 | 226.2 | 217.0 |
| 57 | 185.8 | 235.0 | 202.4 | 231.2 |
| 58 | 158.5 | 220.1 | 157.0 | 165.0 |
| 59 | 157.2 | 160.8 | 158.3 | 158.5 |
| 60 | 166.1 | 165.6 | 164.8 | 161.2 |
| 61 | 173.1 | 166.7 | 169.8 | 162.9 |
| 62 | 228.0 | 165.9 | 216.4 | 199.0 |
| 63 | 198.4 | 205.5 | 219.0 | 231.1 |
| 64 | 159.0 | 249.9 | 157.5 | 177.9 |
| 65 | 157.8 | 163.7 | 158.0 | 158.5 |
| 66 | 165.4 | 165.8 | 165.0 | 161.2 |
| 67 | 170.8 | 166.5 | 168.4 | 162.6 |
| 68 | 214.5 | 166.4 | 203.1 | 183.6 |
| 69 | 214.2 | 182.6 | 232.9 | 230.8 |