#!/usr/bin/env python3
"""Benchmark staged VMEM and FP8 MFMA throughput on gfx94x.

Each wave uses one of four schedules. The default ``2stage_0`` schedule has:

* stage 0 issues V0 coalesced VMEM operations, then ``vmcnt(V0)`` waits for
  the batch issued by the previous stage 0;
* stage 1 executes C0 consecutive FP8 MFMA instructions.

The interleave schedule keeps one rolling stage: each next-batch VMEM operation is
immediately followed by ``vmcnt(V0)``, consumption of the matching previous-
batch result, and an even share of the C0 MFMAs.

All schedules use a finite epilogue: the final round drains and computes the
current batch without issuing another prefetch. Each wave therefore issues
exactly ``rounds`` VMEM batches for ``rounds`` compute rounds.

The prologue issues the first V0 operations. Loads use two register banks so
the next batch can remain outstanding while the previous batch is consumed.
``waves_per_simd`` selects the requested residency. The probe derives static
LDS per workgroup from that value and the workgroup wave count. If final ISA
vector-register usage permits fewer waves, the probe warns and validates the
lower achievable residency. Eight- and sixteen-wave workgroups use an extra
entry/drain barrier on the upper half.
"""

import argparse
import ctypes
import csv
import importlib.util
import json
import math
import os
import re
import statistics
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from pyhip.core.asmjit import JIT, jit
from pyhip.core.hiptools import get_lib, hip_check_error

VOID_POINTER = "void*"
WAVE_SIZE = 64
SIMDS_PER_CU = 4
LANES_PER_TRANSACTION = 8
BYTES_PER_LANE = 16
BYTES_PER_WAVE_OP = WAVE_SIZE * BYTES_PER_LANE
MFMA_OPCODE = "v_mfma_f32_16x16x32_fp8_fp8"
FLOPS_PER_MFMA = 2 * 16 * 16 * 32
MFMA_EXECUTION_CYCLES = 16
REALTIME_CLOCK_MHZ = 100.0
PERF_DETERMINISM_SCLK_MHZ = 1800
MIN_TIMED_DISPATCH_US = 100.0
METADATA_DWORDS = 12
WAVE_MODES = (4, 8, 16)
SCHEDULES = (
    "2stage_0",
    "2stage_prio",
    "2stage_barrier",
    "interleave",
)
SCHEDULE_CODES = {
    schedule: code for code, schedule in enumerate(SCHEDULES)
}
CACHE_POLICIES = ("non_temporal", "temporal")
VMEM_OPS = (
    "read",
    "write",
    "mixed",
    "read3_write2",
    "read12_write8",
)
VMEM_MIX_PATTERN = re.compile(r"read(\d+)_write(\d+)")
PATTERNS = ("private", "simd", "stage", "workgroup", "cu")
PHYSICAL_PATTERNS = ("simd", "stage", "cu")
XCC_COUNT = 4
SE_PER_XCC = 4
CU_ID_CAPACITY_PER_SE = 8
TOPOLOGY_CU_CAPACITY = XCC_COUNT * SE_PER_XCC * CU_ID_CAPACITY_PER_SE
LDS_BYTES_PER_CU = 64 * 1024
LDS_ALLOCATION_GRANULARITY = 256
MAX_WAVES_PER_SIMD = 8
VECTOR_REGISTERS_PER_SIMD = 512
VECTOR_REGISTER_ALLOCATION_GRANULARITY = 8
DEFAULT_CASES = ((64, 8), (64, 12), (32, 12))
EXPECTED_POWER_CAP_W = 650
DEFAULT_AMDSMI_ROOT = Path(
    os.environ.get(
        "AMDSMI_ROOT",
        "/tmp/amd-smi-lib-26.2.2-rocm-7.2.3/opt/rocm-7.2.3",
    )
)


def _resident_workgroups_per_cu(waves_per_block, waves_per_simd):
    if not 1 <= waves_per_simd <= MAX_WAVES_PER_SIMD:
        raise RuntimeError(
            f"waves-per-simd must be in [1, {MAX_WAVES_PER_SIMD}]"
        )
    resident_waves = SIMDS_PER_CU * waves_per_simd
    if resident_waves % waves_per_block:
        raise RuntimeError(
            f"{waves_per_block} waves/workgroup cannot produce exactly "
            f"{waves_per_simd} waves/SIMD"
        )
    return resident_waves // waves_per_block


def _lds_bytes(waves_per_block, waves_per_simd):
    resident_workgroups = _resident_workgroups_per_cu(
        waves_per_block, waves_per_simd
    )
    return (
        LDS_BYTES_PER_CU
        // resident_workgroups
        // LDS_ALLOCATION_GRANULARITY
        * LDS_ALLOCATION_GRANULARITY
    )


def _register_occupancy(artifact, waves_per_block):
    assembly_path = artifact.get("assembly_path")
    if assembly_path and Path(assembly_path).is_file():
        assembly = Path(assembly_path).read_text(encoding="utf-8")

        def metadata_count(name):
            matches = re.findall(
                rf"\.set\s+\S+\.{name},\s*(\d+)", assembly
            )
            if not matches:
                raise RuntimeError(
                    f"missing {name} in ISA metadata: {assembly_path}"
                )
            return int(matches[-1])

        vgprs = metadata_count("num_vgpr")
        agprs = metadata_count("num_agpr")
        sgprs = metadata_count("numbered_sgpr")
    else:
        used_gprs = artifact.get("used_gprs", ())

        def register_count(prefix):
            indices = [
                int(register[1:])
                for register in used_gprs
                if register.startswith(prefix) and register[1:].isdigit()
            ]
            return max(indices, default=-1) + 1

        vgprs = register_count("v")
        agprs = register_count("a")
        sgprs = register_count("s")
        if not vgprs and not agprs:
            raise RuntimeError("final ISA register usage is unavailable")

    vector_registers = vgprs + agprs
    allocated_vector_registers = (
        math.ceil(
            vector_registers / VECTOR_REGISTER_ALLOCATION_GRANULARITY
        )
        * VECTOR_REGISTER_ALLOCATION_GRANULARITY
    )
    register_waves = min(
        MAX_WAVES_PER_SIMD,
        VECTOR_REGISTERS_PER_SIMD // allocated_vector_registers,
    )
    waves_per_workgroup_per_simd = waves_per_block // SIMDS_PER_CU
    register_workgroups = (
        register_waves // waves_per_workgroup_per_simd
    )
    if register_workgroups < 1:
        raise RuntimeError(
            "final ISA register usage cannot support one workgroup"
        )
    achievable_waves = min(
        MAX_WAVES_PER_SIMD,
        register_workgroups * waves_per_workgroup_per_simd,
    )
    return {
        "sgpr_count": sgprs,
        "vgpr_count": vgprs,
        "agpr_count": agprs,
        "vector_registers_per_wave": vector_registers,
        "allocated_vector_registers_per_wave": (
            allocated_vector_registers
        ),
        "max_waves_per_simd_by_vector_registers": register_waves,
        "max_workgroups_per_cu_by_vector_registers": (
            register_workgroups
        ),
        "achievable_waves_per_simd": achievable_waves,
    }


def _driver_occupancy(artifact, waves_per_block):
    kernel = artifact.get("kernel")
    if kernel is None:
        raise RuntimeError("compiled kernel handle is unavailable")
    hip_kernel = kernel.build()
    hip_kernel.lazy_load_func()
    runtime = get_lib()
    occupancy = runtime.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    occupancy.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    occupancy.restype = ctypes.c_int32
    active_blocks = ctypes.c_int()
    hip_check_error(
        occupancy(
            ctypes.byref(active_blocks),
            hip_kernel.p_func,
            waves_per_block * WAVE_SIZE,
            0,
        )
    )
    waves_per_workgroup_per_simd = waves_per_block // SIMDS_PER_CU
    return {
        "max_active_workgroups_per_cu": active_blocks.value,
        "max_waves_per_simd": min(
            MAX_WAVES_PER_SIMD,
            active_blocks.value * waves_per_workgroup_per_simd,
        ),
    }


def _effective_waves_per_simd(
    register_occupancy, requested_waves_per_simd
):
    achievable = min(
        register_occupancy["achievable_waves_per_simd"],
        register_occupancy["driver"]["max_waves_per_simd"],
    )
    if achievable < requested_waves_per_simd:
        warnings.warn(
            "requested "
            f"{requested_waves_per_simd} waves/SIMD, but final ISA uses "
            f"{register_occupancy['vgpr_count']} VGPR + "
            f"{register_occupancy['agpr_count']} AGPR "
            f"({register_occupancy['allocated_vector_registers_per_wave']} "
            "allocated vector registers/wave); compiled-resource "
            "occupancy limits this workgroup "
            f"to {achievable} waves/SIMD",
            RuntimeWarning,
            stacklevel=2,
        )
    return min(requested_waves_per_simd, achievable)


def _read_realtime(builder):
    value = builder.gpr(2, "su32", align=2)
    builder.s_memrealtime(value)
    builder.s_waitcnt(mod="lgkmcnt(0)")
    return value


def _read_shader_cycles(builder):
    value = builder.gpr(2, "su32", align=2)
    builder.s_memtime(value)
    builder.s_waitcnt(mod="lgkmcnt(0)")
    return value


def _read_count(v0, vmem_op):
    if vmem_op == "read":
        return v0
    if vmem_op == "mixed":
        return (v0 + 1) // 2
    match = VMEM_MIX_PATTERN.fullmatch(vmem_op)
    if match is not None:
        reads, writes = (int(value) for value in match.groups())
        if reads + writes != v0:
            raise RuntimeError(
                f"{vmem_op} requires V0={reads + writes}, got {v0}"
            )
        return reads
    return 0


def _is_read(vmem_op, operation_index):
    if vmem_op == "read":
        return True
    if vmem_op == "mixed":
        return operation_index % 2 == 0
    match = VMEM_MIX_PATTERN.fullmatch(vmem_op)
    return match is not None and operation_index < int(match.group(1))


def _parse_vmem_op(value):
    if value in VMEM_OPS or VMEM_MIX_PATTERN.fullmatch(value):
        return value
    raise argparse.ArgumentTypeError(
        "vmem-op must be read, write, mixed, or readN_writeM"
    )


def _streams_per_epoch(waves_per_block, pattern, blocks_per_epoch):
    if pattern == "private":
        return blocks_per_epoch * waves_per_block
    if pattern == "workgroup":
        return blocks_per_epoch
    if pattern == "simd":
        return TOPOLOGY_CU_CAPACITY * SIMDS_PER_CU
    if pattern == "stage":
        return TOPOLOGY_CU_CAPACITY * 2
    if pattern == "cu":
        return TOPOLOGY_CU_CAPACITY
    raise RuntimeError(f"unknown pattern: {pattern}")


def _stream_count(waves_per_block, blocks, pattern, blocks_per_epoch):
    return math.ceil(blocks / blocks_per_epoch) * _streams_per_epoch(
        waves_per_block, pattern, blocks_per_epoch
    )


def _active_streams_per_epoch(
    waves_per_block, pattern, blocks_per_epoch, cu_count
):
    if pattern == "private":
        return blocks_per_epoch * waves_per_block
    if pattern == "workgroup":
        return blocks_per_epoch
    if pattern == "simd":
        return cu_count * SIMDS_PER_CU
    if pattern == "stage":
        return cu_count * 2
    if pattern == "cu":
        return cu_count
    raise RuntimeError(f"unknown pattern: {pattern}")


def _stream_id(
    builder,
    waves_per_block,
    pattern,
    hw_id=None,
    xcc_id=None,
):
    block = builder.blockIdx.x[0]
    wave = builder.warp_id[0]
    if pattern == "private":
        return block * waves_per_block + wave
    if pattern == "workgroup":
        return block
    simd = (hw_id >> 4) & 3
    slot = hw_id & 0xF
    cu = (hw_id >> 8) & 0xF
    se = (hw_id >> 13) & 7
    physical_cu = (
        xcc_id * (SE_PER_XCC * CU_ID_CAPACITY_PER_SE)
        + se * CU_ID_CAPACITY_PER_SE
        + cu
    )
    if pattern == "simd":
        return physical_cu * SIMDS_PER_CU + simd
    if pattern == "stage":
        phase = (
            slot
            if waves_per_block == 4
            else wave // (waves_per_block // 2)
        )
        return physical_cu * 2 + phase
    if pattern == "cu":
        return physical_cu
    raise RuntimeError(f"unknown pattern: {pattern}")


def _make_mfma_registers(builder):
    operands = builder.gpr(4, "vu32", 0x40404040, align=2)
    operand_a = operands[0:1]
    operand_b = operands[2:3]
    accumulators = builder.gpr(4, 4, "af32", align=4)
    accumulators[...] = 0.0
    return accumulators, operand_a, operand_b, operands


def _emit_mfma_stage(
    builder, accumulators, operand_a, operand_b, c0
):
    for index in range(c0):
        accumulator = accumulators[index & 3]
        builder.v_mfma_f32_16x16x32_fp8_fp8(
            accumulator,
            operand_a,
            operand_b,
            accumulator,
        )


def _mfma_partition(c0, v0):
    base = c0 // v0
    remainder = c0 % v0
    return tuple(
        base + int(operation_index < remainder)
        for operation_index in range(v0)
    )


@jit(no_pass=["pass_dse", "pass_dce"])
def wave_stage_tflops(
    builder: JIT,
    waves_per_block,
    waves_per_simd,
    rounds,
    c0,
    v0,
    s,
    vmem_op,
    pattern,
    n,
    epoch_bytes,
    i,
    t,
    data: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    metadata: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    schedule = SCHEDULES[s]
    assert waves_per_block in WAVE_MODES
    assert vmem_op in VMEM_OPS or VMEM_MIX_PATTERN.fullmatch(vmem_op)
    assert pattern in PATTERNS
    assert schedule != "2stage_barrier" or waves_per_block >= 8
    assert epoch_bytes > 0 and epoch_bytes & (epoch_bytes - 1) == 0
    nt = bool(n)
    interleave = schedule == "interleave"
    use_priority = schedule in ("2stage_prio", "2stage_barrier")
    use_barrier = schedule == "2stage_barrier"

    builder.alloc_lds(
        _lds_bytes(waves_per_block, waves_per_simd), align=16
    )
    data_buffer = builder.Buffer(data, epoch_bytes)
    output_buffer = builder.Buffer(output, epoch_bytes)
    read_count = _read_count(v0, vmem_op)
    load_values = (
        builder.gpr(2, read_count, 4, "vu32", align=4)
        if read_count
        else None
    )
    operation_offsets = builder.gpr(v0, "su32")
    lane_group = builder.lane_id[0] // LANES_PER_TRANSACTION
    lane_in_group = builder.lane_id[0] % LANES_PER_TRANSACTION
    vector_offset = builder.gpr(
        "vu32",
        lane_group * (LANES_PER_TRANSACTION * BYTES_PER_LANE)
        + lane_in_group * BYTES_PER_LANE,
    )
    accumulators, operand_a, operand_b, operands = (
        _make_mfma_registers(builder)
    )
    store_value = operands if vmem_op != "read" else None
    load_sink = builder.gpr("vu32", 0) if read_count else None

    if i or t or pattern in PHYSICAL_PATTERNS:
        hw_id = builder.gpr("su32")
        xcc_id = builder.gpr("su32")
        builder.s_getreg_b32(hw_id, mod="hwreg(HW_REG_HW_ID, 0, 20)")
        builder.s_getreg_b32(xcc_id, mod="hwreg(HW_REG_XCC_ID, 0, 4)")
    else:
        hw_id = None
        xcc_id = None
    stream_batches = rounds * v0
    stream_offset = builder.gpr(
        "su32",
        _stream_id(
            builder,
            waves_per_block,
            pattern,
            hw_id,
            xcc_id,
        )
        * (stream_batches * BYTES_PER_WAVE_OP),
    )
    if t:
        realtime_start = _read_realtime(builder)
        cycles_start = _read_shader_cycles(builder)

    def prepare_batch():
        for operation_index in range(v0):
            operation_offsets[operation_index] = (
                stream_offset[0]
                + operation_index * BYTES_PER_WAVE_OP
            )
        stream_offset[0] = (
            stream_offset[0] + v0 * BYTES_PER_WAVE_OP
        )

    def read_index(operation_index):
        return sum(
            _is_read(vmem_op, prior_index)
            for prior_index in range(operation_index)
        )

    def issue_operation(bank, operation_index):
        if _is_read(vmem_op, operation_index):
            data_buffer.load_dwordx4(
                load_values[bank, read_index(operation_index)],
                vector_offset,
                operation_offsets[operation_index],
                non_temporal=nt,
            )
        else:
            output_buffer.store_dwordx4(
                store_value,
                vector_offset,
                operation_offsets[operation_index],
                ext_mod="sc0 nt" if nt else "",
            )

    def issue_batch(bank):
        prepare_batch()
        for operation_index in range(v0):
            issue_operation(bank, operation_index)

    def consume_operation(bank, operation_index):
        if _is_read(vmem_op, operation_index):
            load_index = read_index(operation_index)
            builder.v_xor_b32(
                load_sink,
                load_sink,
                load_values[bank, load_index, 0],
            )

    def consume_batch(bank):
        for load_index in range(read_count):
            builder.v_xor_b32(
                load_sink,
                load_sink,
                load_values[bank, load_index, 0],
            )

    half_waves = waves_per_block // 2
    if use_barrier:
        with builder.If(builder.warp_id[0] >= half_waves):
            builder.s_barrier()
        builder.s_barrier()
    issue_batch(0)

    def emit_round(current_bank, next_bank, issue_next=True):

        if interleave:
            if issue_next:
                prepare_batch()
                mfma_counts = _mfma_partition(c0, v0)
                mfma_begin = 0
                for operation_index, mfma_count in enumerate(mfma_counts):
                    issue_operation(next_bank, operation_index)
                    # gfx942 tracks loads and stores in one ordered VM_CNT.
                    # V0 old requests plus one new request means vmcnt(V0)
                    # retires the oldest producer before consumption.
                    builder.s_waitcnt(mod=f"vmcnt({v0})")
                    consume_operation(current_bank, operation_index)
                    for mfma_index in range(
                        mfma_begin, mfma_begin + mfma_count
                    ):
                        accumulator = accumulators[mfma_index & 3]
                        builder.v_mfma_f32_16x16x32_fp8_fp8(
                            accumulator,
                            operand_a,
                            operand_b,
                            accumulator,
                        )
                    mfma_begin += mfma_count
            else:
                builder.s_waitcnt(mod="vmcnt(0)")
                consume_batch(current_bank)
                _emit_mfma_stage(
                    builder, accumulators, operand_a, operand_b, c0
                )
            return

        if i:
            builder.s_nop(11)
        if issue_next:
            issue_batch(next_bank)
            builder.s_waitcnt(mod=f"vmcnt({v0})")
        else:
            builder.s_waitcnt(mod="vmcnt(0)")
        consume_batch(current_bank)
        if use_barrier:
            builder.s_barrier()

        if i:
            builder.s_nop(12)
        if use_priority:
            builder.s_setprio(1)
        _emit_mfma_stage(
            builder, accumulators, operand_a, operand_b, c0
        )
        if use_priority:
            builder.s_setprio(0)
        if use_barrier:
            builder.s_barrier()

    if i:
        for round_index in range(rounds):
            emit_round(
                round_index & 1,
                (round_index + 1) & 1,
                issue_next=round_index + 1 < rounds,
            )
    else:
        pipelined_rounds = rounds - 1
        pair_index = builder.gpr("su32", 0)
        with builder.While(pair_index[0] < pipelined_rounds // 2):
            emit_round(0, 1)
            emit_round(1, 0)
            pair_index[0] += 1
        if pipelined_rounds % 2:
            emit_round(0, 1)
        emit_round(pipelined_rounds & 1, 0, issue_next=False)
    if use_barrier:
        with builder.If(builder.warp_id[0] < half_waves):
            builder.s_barrier()

    if t:
        cycles_stop = _read_shader_cycles(builder)
        realtime_stop = _read_realtime(builder)
        global_wave = (
            builder.blockIdx.x[0] * waves_per_block
            + builder.warp_id[0]
        )
        metadata_offset = builder.gpr(
            "su32", global_wave * METADATA_DWORDS * 4
        )
        record = builder.gpr(METADATA_DWORDS, "su32", align=4)
        record[0] = builder.blockIdx.x[0]
        record[1] = builder.warp_id[0]
        record[2] = hw_id[0]
        record[3] = xcc_id[0]
        record[4] = realtime_start[0]
        record[5] = realtime_start[1]
        record[6] = realtime_stop[0]
        record[7] = realtime_stop[1]
        record[8] = cycles_start[0]
        record[9] = cycles_start[1]
        record[10] = cycles_stop[0]
        record[11] = cycles_stop[1]
        builder.s_store_dwordx4(
            record[0:3], metadata, metadata_offset, mod="glc"
        )
        builder.s_store_dwordx4(
            record[4:7], metadata, metadata_offset + 16, mod="glc"
        )
        builder.s_store_dwordx4(
            record[8:11], metadata, metadata_offset + 32, mod="glc"
        )
        builder.s_waitcnt(mod="lgkmcnt(0)")


def _u32(value):
    return int(value) & 0xFFFFFFFF


def _decode_metadata(row):
    values = [_u32(value) for value in row]
    hw_id = values[2]
    return {
        "block": values[0],
        "wave": values[1],
        "slot": hw_id & 0xF,
        "simd": (hw_id >> 4) & 0x3,
        "cu": (hw_id >> 8) & 0xF,
        "se": (hw_id >> 13) & 0x7,
        "xcc": values[3],
        "start": values[4] | (values[5] << 32),
        "stop": values[6] | (values[7] << 32),
        "cycles_start": values[8] | (values[9] << 32),
        "cycles_stop": values[10] | (values[11] << 32),
    }


def _effective_sclk_mhz(metadata):
    frequencies = []
    for row in metadata.reshape(-1, METADATA_DWORDS).tolist():
        record = _decode_metadata(row)
        realtime_ticks = (record["stop"] - record["start"]) & (
            (1 << 64) - 1
        )
        shader_cycles = (
            record["cycles_stop"] - record["cycles_start"]
        ) & ((1 << 64) - 1)
        if realtime_ticks <= 0 or shader_cycles <= 0:
            raise RuntimeError(
                "invalid synchronized clock metadata: "
                f"realtime={realtime_ticks}, cycles={shader_cycles}"
            )
        frequencies.append(
            shader_cycles * REALTIME_CLOCK_MHZ / realtime_ticks
        )
    return _summary(frequencies)


def _dispatch_duration_us(metadata):
    records = [
        _decode_metadata(row)
        for row in metadata.reshape(-1, METADATA_DWORDS).tolist()
    ]
    if not records:
        raise RuntimeError("timed dispatch has no hardware metadata")
    duration_ticks = max(record["stop"] for record in records) - min(
        record["start"] for record in records
    )
    if duration_ticks <= 0:
        raise RuntimeError(
            f"invalid timed dispatch duration: {duration_ticks} ticks"
        )
    return duration_ticks / REALTIME_CLOCK_MHZ


def _theoretical_fp8_tflops(cu_count, effective_sclk_mhz):
    return (
        FLOPS_PER_MFMA
        / MFMA_EXECUTION_CYCLES
        * cu_count
        * SIMDS_PER_CU
        * effective_sclk_mhz
        / 1.0e6
    )


def _validate_residency(
    waves_per_block,
    requested_waves_per_simd,
    expected_waves_per_simd,
    metadata,
    cu_count,
    register_occupancy,
):
    groups = defaultdict(list)
    physical_cus = set()
    topology_failures = []
    for row in metadata.tolist():
        wave = _decode_metadata(row)
        if not (
            0 <= wave["xcc"] < XCC_COUNT
            and 0 <= wave["se"] < SE_PER_XCC
            and 0 <= wave["cu"] < CU_ID_CAPACITY_PER_SE
        ):
            topology_failures.append(
                {
                    "xcc": wave["xcc"],
                    "se": wave["se"],
                    "cu": wave["cu"],
                }
            )
        physical_cus.add((wave["xcc"], wave["se"], wave["cu"]))
        key = (
            wave["xcc"],
            wave["se"],
            wave["cu"],
            wave["simd"],
        )
        groups[key].append(wave)

    expected_blocks = _resident_workgroups_per_cu(
        waves_per_block, expected_waves_per_simd
    )
    failures = []
    for key, waves in sorted(groups.items()):
        slots = sorted(wave["slot"] for wave in waves)
        blocks = {wave["block"] for wave in waves}
        overlap = min(wave["stop"] for wave in waves) - max(
            wave["start"] for wave in waves
        )
        reasons = []
        if len(waves) != expected_waves_per_simd:
            reasons.append("wave_count")
        if slots != list(range(expected_waves_per_simd)):
            reasons.append("slots")
        if len(blocks) != expected_blocks:
            reasons.append("workgroups")
        if overlap <= 0:
            reasons.append("no_lifetime_overlap")
        if waves_per_block > 4 and expected_blocks == 1:
            low = sum(wave["wave"] < waves_per_block // 2 for wave in waves)
            if low * 2 != expected_waves_per_simd:
                reasons.append("phase_split")
        if reasons:
            failures.append(
                {
                    "key": key,
                    "reasons": reasons,
                    "waves": len(waves),
                    "slots": slots,
                    "blocks": sorted(blocks),
                    "overlap_ticks": overlap,
                }
            )

    return {
        "waves_per_workgroup": waves_per_block,
        "requested_waves_per_simd": requested_waves_per_simd,
        "actual_waves_per_simd": expected_waves_per_simd,
        "lds_bytes": _lds_bytes(
            waves_per_block, requested_waves_per_simd
        ),
        "expected_workgroups_per_cu": expected_blocks,
        "register_occupancy": register_occupancy,
        "physical_simd_groups": len(groups),
        "expected_physical_simd_groups": cu_count * SIMDS_PER_CU,
        "physical_cu_count": len(physical_cus),
        "topology_failures": topology_failures,
        "failures": failures,
        "valid": (
            not failures
            and not topology_failures
            and len(physical_cus) == cu_count
            and len(groups) == cu_count * SIMDS_PER_CU
        ),
    }


def _percentile(ordered, fraction):
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p25": _percentile(ordered, 0.25),
        "p75": _percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def _load_state_helper():
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "tests/contrib/moe/probe_control_k128_hardware.py"
    spec = importlib.util.spec_from_file_location(
        "wave_stage_tflops_state", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GPU state helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compact_state(state):
    keys = (
        "physical_device",
        "gpu_busy_percent",
        "vram_allocated_percent",
        "performance_level",
        "sclk",
        "mclk",
        "fclk",
        "power_cap_w",
        "ptl_state",
        "ptl_format",
        "numa_balancing",
    )
    return {key: state[key] for key in keys}


def _launch(
    blocks,
    waves_per_block,
    rounds,
    c0,
    v0,
    schedule,
    waves_per_simd,
    vmem_op,
    pattern,
    cache_policy,
    data_bytes,
    instrumented,
    record_timing,
    data,
    output,
    metadata,
):
    artifact = wave_stage_tflops(
        [blocks],
        [waves_per_block * WAVE_SIZE],
        waves_per_block,
        waves_per_simd,
        rounds,
        c0,
        v0,
        SCHEDULE_CODES[schedule],
        vmem_op,
        pattern,
        cache_policy == "non_temporal",
        data_bytes,
        instrumented,
        record_timing,
        data.data_ptr(),
        output.data_ptr(),
        metadata.data_ptr(),
    )
    if "assembly_path" not in artifact or "kernel" not in artifact:
        for kernel, cached_artifact in wave_stage_tflops.kernel_cache.values():
            if cached_artifact is artifact:
                artifact.setdefault(
                    "assembly_path",
                    str(Path(kernel.src_fpath).with_suffix(".s")),
                )
                artifact.setdefault("kernel", kernel)
                break
    return artifact


def _benchmark_case(args, case, data, output, properties):
    waves_per_block, c0, v0, pattern = case
    cu_count = properties.multi_processor_count
    requested_waves_per_simd = args.waves_per_simd
    residency_blocks = (
        cu_count
        * _resident_workgroups_per_cu(
            waves_per_block, requested_waves_per_simd
        )
    )
    blocks_per_epoch = residency_blocks
    residency_waves = residency_blocks * waves_per_block
    metadata = torch.zeros(
        (residency_waves, METADATA_DWORDS),
        dtype=torch.int32,
        device="cuda",
    )
    residency_rounds = min(args.rounds, args.residency_rounds)
    epoch_bytes = args.buffer_mib * 1024 * 1024
    artifact = _launch(
        residency_blocks,
        waves_per_block,
        residency_rounds,
        c0,
        v0,
        args.schedule,
        requested_waves_per_simd,
        args.vmem_op,
        pattern,
        args.cache_policy,
        epoch_bytes,
        True,
        True,
        data,
        output,
        metadata,
    )
    torch.cuda.synchronize()
    register_occupancy = _register_occupancy(
        artifact, waves_per_block
    )
    register_occupancy["driver"] = _driver_occupancy(
        artifact, waves_per_block
    )
    actual_waves_per_simd = _effective_waves_per_simd(
        register_occupancy, requested_waves_per_simd
    )
    if actual_waves_per_simd != requested_waves_per_simd:
        residency_blocks = (
            cu_count
            * _resident_workgroups_per_cu(
                waves_per_block, actual_waves_per_simd
            )
        )
        blocks_per_epoch = residency_blocks
        residency_waves = residency_blocks * waves_per_block
        metadata = torch.zeros(
            (residency_waves, METADATA_DWORDS),
            dtype=torch.int32,
            device="cuda",
        )
        _launch(
            residency_blocks,
            waves_per_block,
            residency_rounds,
            c0,
            v0,
            args.schedule,
            requested_waves_per_simd,
            args.vmem_op,
            pattern,
            args.cache_policy,
            epoch_bytes,
            True,
            True,
            data,
            output,
            metadata,
        )
        torch.cuda.synchronize()
    residency = _validate_residency(
        waves_per_block,
        requested_waves_per_simd,
        actual_waves_per_simd,
        metadata.cpu(),
        cu_count,
        register_occupancy,
    )
    if not residency["valid"]:
        raise RuntimeError(
            f"invalid {waves_per_block}-wave residency: {residency}"
        )

    resident_workgroups = _resident_workgroups_per_cu(
        waves_per_block, actual_waves_per_simd
    )
    if (
        not args.single_dispatch
        and args.workgroups_per_cu % resident_workgroups
    ):
        raise RuntimeError(
            f"workgroups-per-cu must be divisible by "
            f"{resident_workgroups} for {waves_per_block}-wave "
            f"{actual_waves_per_simd}-waves/SIMD residency"
        )
    if args.single_dispatch:
        epoch_count = 1
        blocks = (
            args.workgroups
            if args.workgroups is not None
            else cu_count * args.workgroups_per_cu
        )
        blocks_per_epoch = blocks
    else:
        epoch_count = args.workgroups_per_cu // resident_workgroups
        blocks = residency_blocks
        blocks_per_epoch = residency_blocks
    streams_per_epoch = _streams_per_epoch(
        waves_per_block, pattern, blocks_per_epoch
    )
    vmem_batches_per_wave = args.rounds
    vmem_instructions_per_wave = vmem_batches_per_wave * v0
    bytes_per_stream = (
        vmem_instructions_per_wave * BYTES_PER_WAVE_OP
    )
    if streams_per_epoch * bytes_per_stream > epoch_bytes:
        required_mib = math.ceil(
            streams_per_epoch * bytes_per_stream / (1024 * 1024)
        )
        raise RuntimeError(
            f"{waves_per_block}-wave {pattern} requires at least "
            f"{required_mib} MiB per address epoch"
        )
    address_stream_slots = epoch_count * _streams_per_epoch(
        waves_per_block, pattern, blocks_per_epoch
    )
    active_stream_count = epoch_count * _active_streams_per_epoch(
        waves_per_block, pattern, blocks_per_epoch, cu_count
    )
    timing_metadata = torch.zeros(
        (
            args.launches_per_sample,
            epoch_count,
            blocks * waves_per_block,
            METADATA_DWORDS,
        ),
        dtype=torch.int32,
        device="cuda",
    )

    def timed_launch(metadata):
        for epoch_index in range(epoch_count):
            epoch_begin = epoch_index * epoch_bytes // 4
            epoch_end = epoch_begin + epoch_bytes // 4
            _launch(
                blocks,
                waves_per_block,
                args.rounds,
                c0,
                v0,
                args.schedule,
                requested_waves_per_simd,
                args.vmem_op,
                pattern,
                args.cache_policy,
                epoch_bytes,
                False,
                True,
                data[epoch_begin:epoch_end],
                output[epoch_begin:epoch_end],
                metadata[epoch_index],
            )

    for _ in range(args.warmups):
        timed_launch(timing_metadata[0])
    torch.cuda.synchronize()

    flop_per_launch = (
        blocks
        * epoch_count
        * waves_per_block
        * args.rounds
        * c0
        * FLOPS_PER_MFMA
    )
    issued_bytes_per_launch = (
        blocks
        * epoch_count
        * waves_per_block
        * vmem_instructions_per_wave
        * BYTES_PER_WAVE_OP
    )
    elapsed_ms = []
    tflops = []
    issued_gbps = []
    effective_sclk_mhz = []
    theoretical_fp8_tflops = []
    theoretical_attainment = []
    dispatch_duration_samples = []
    for _ in range(args.samples):
        timing_metadata.zero_()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for launch_index in range(args.launches_per_sample):
            timed_launch(timing_metadata[launch_index])
        stop.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(stop)
        per_launch_ms = total_ms / args.launches_per_sample
        sample_tflops = (
            flop_per_launch
            * args.launches_per_sample
            / (total_ms * 1.0e9)
        )
        host_timing_metadata = timing_metadata.cpu()
        dispatch_durations = [
            _dispatch_duration_us(
                host_timing_metadata[launch_index, epoch_index]
            )
            for launch_index in range(args.launches_per_sample)
            for epoch_index in range(epoch_count)
        ]
        if min(dispatch_durations) < MIN_TIMED_DISPATCH_US:
            raise RuntimeError(
                f"timed dispatch is only {min(dispatch_durations):.3f} us; "
                f"increase --rounds so every dispatch is at least "
                f"{MIN_TIMED_DISPATCH_US:.0f} us"
            )
        clock = _effective_sclk_mhz(host_timing_metadata)
        peak_tflops = _theoretical_fp8_tflops(
            cu_count, clock["median"]
        )
        elapsed_ms.append(per_launch_ms)
        tflops.append(sample_tflops)
        issued_gbps.append(
            issued_bytes_per_launch * args.launches_per_sample
            / (total_ms * 1.0e6)
        )
        effective_sclk_mhz.append(clock["median"])
        theoretical_fp8_tflops.append(peak_tflops)
        theoretical_attainment.append(sample_tflops / peak_tflops)
        dispatch_duration_samples.append(dispatch_durations)

    logical_unique_bytes = (
        active_stream_count * bytes_per_stream
    )
    result = {
        "config": {
            "waves_per_workgroup": waves_per_block,
            "threads_per_workgroup": waves_per_block * WAVE_SIZE,
            "c0": c0,
            "v0": v0,
            "schedule": args.schedule,
            "requested_waves_per_simd": requested_waves_per_simd,
            "actual_waves_per_simd": actual_waves_per_simd,
            "vmem_op": args.vmem_op,
            "pattern": pattern,
            "cache_policy": args.cache_policy,
            "rounds": args.rounds,
            "workgroups_per_cu": args.workgroups_per_cu,
            "requested_workgroups": args.workgroups,
            "single_dispatch": args.single_dispatch,
            "workgroups_per_epoch": blocks,
            "address_epoch_count": epoch_count,
            "total_workgroups": blocks * epoch_count,
            "waves_per_epoch": blocks * waves_per_block,
            "active_stream_count": active_stream_count,
            "address_stream_slots": address_stream_slots,
            "blocks_per_address_epoch": blocks_per_epoch,
            "buffer_mib": args.buffer_mib,
            "total_buffer_mib": args.buffer_mib
            * epoch_count,
            "allocated_workspace_mib": args.buffer_mib
            * epoch_count
            * (2 if _read_count(v0, args.vmem_op) < v0 else 1),
            "launches_per_sample": args.launches_per_sample,
            "minimum_timed_dispatch_us": MIN_TIMED_DISPATCH_US,
        },
        "residency": residency,
        "residency_active_stream_count": _active_streams_per_epoch(
            waves_per_block, pattern, blocks_per_epoch, cu_count
        ),
        "residency_address_stream_slots": streams_per_epoch,
        "work": {
            "flops_per_mfma": FLOPS_PER_MFMA,
            "round_definition": (
                "V0 repetitions of issue one VMEM, immediate vmcnt(V0), "
                "consume matching previous-bank result, and an even "
                "share of C0 MFMA instructions, per wave; the final "
                "round drains and computes without another prefetch"
                if args.schedule == "interleave"
                else "one stage0 issue/wait/consume followed by one "
                "stage1 segment of C0 MFMA instructions, per wave; "
                "the final round drains and computes without another "
                "prefetch"
            ),
            "workgroup_definition": "one launched workgroup",
            "mfma_instructions_per_wave_per_workgroup": args.rounds * c0,
            "mfma_instructions_per_workgroup": (
                waves_per_block * args.rounds * c0
            ),
            "flops_per_workgroup": (
                waves_per_block * args.rounds * c0 * FLOPS_PER_MFMA
            ),
            "vmem_batches_per_wave_per_workgroup": vmem_batches_per_wave,
            "vmem_instructions_per_wave_per_workgroup": (
                vmem_instructions_per_wave
            ),
            "vmem_instructions_per_workgroup": (
                waves_per_block * vmem_instructions_per_wave
            ),
            "logical_vmem_bytes_per_workgroup": (
                waves_per_block
                * vmem_instructions_per_wave
                * BYTES_PER_WAVE_OP
            ),
            "flops_per_launch": flop_per_launch,
            "vmem_instructions_per_wave": vmem_instructions_per_wave,
            "issued_vmem_bytes_per_launch": issued_bytes_per_launch,
            "logical_unique_bytes_per_launch": logical_unique_bytes,
            "epoch_utilization": (
                streams_per_epoch * bytes_per_stream / epoch_bytes
            ),
            "bytes_per_wave_vmem_instruction": BYTES_PER_WAVE_OP,
            "lanes_per_transaction": LANES_PER_TRANSACTION,
            "bytes_per_lane": BYTES_PER_LANE,
            "nominal_sclk_mhz": PERF_DETERMINISM_SCLK_MHZ,
            "nominal_theoretical_fp8_tflops": _theoretical_fp8_tflops(
                cu_count, PERF_DETERMINISM_SCLK_MHZ
            ),
        },
        "kernel_ms": _summary(elapsed_ms),
        "tflops": _summary(tflops),
        "issued_gbps": _summary(issued_gbps),
        "effective_sclk_mhz": _summary(effective_sclk_mhz),
        "theoretical_fp8_tflops": _summary(theoretical_fp8_tflops),
        "end_to_end_theoretical_attainment": _summary(
            theoretical_attainment
        ),
        "timed_dispatch_us": _summary(
            [
                duration
                for sample in dispatch_duration_samples
                for duration in sample
            ]
        ),
        "samples": [
            {
                "kernel_ms": elapsed_ms[index],
                "tflops": tflops[index],
                "issued_gbps": issued_gbps[index],
                "effective_sclk_mhz": effective_sclk_mhz[index],
                "theoretical_fp8_tflops": theoretical_fp8_tflops[index],
                "end_to_end_theoretical_attainment": (
                    theoretical_attainment[index]
                ),
                "timed_dispatch_us": _summary(
                    dispatch_duration_samples[index]
                ),
            }
            for index in range(args.samples)
        ],
    }
    print(
        f"waves={waves_per_block:2d} C0={c0:3d} V0={v0:2d} "
        f"schedule={args.schedule:16s} "
        f"waves/SIMD={actual_waves_per_simd} "
        f"pattern={pattern:9s} "
        f"op={args.vmem_op:5s} "
        f"{result['kernel_ms']['median']:.4f} ms "
        f"{result['tflops']['median']:.2f} TFLOPS "
        f"{result['issued_gbps']['median']:.1f} GB/s "
        f"SCLK={result['effective_sclk_mhz']['median']:.1f} MHz "
        f"peak={result['theoretical_fp8_tflops']['median']:.2f} "
        f"attain={result['end_to_end_theoretical_attainment']['median']:.2%} "
        f"dispatch_min={result['timed_dispatch_us']['min']:.2f} us"
    )
    return result


def _parse_case(value):
    try:
        c0, v0 = (int(item) for item in value.split(":", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "case must have the form C0:V0"
        ) from error
    if c0 < 1 or v0 < 1:
        raise argparse.ArgumentTypeError("C0 and V0 must be positive")
    return c0, v0


def _validate_args(args):
    if args.rounds < 1:
        raise RuntimeError("rounds must be positive")
    if args.workgroups_per_cu < 2:
        raise RuntimeError("workgroups-per-cu must be at least 2")
    if args.samples < 1 or args.warmups < 1:
        raise RuntimeError("samples and warmups must be positive")
    if args.launches_per_sample < 1:
        raise RuntimeError("launches-per-sample must be positive")
    if args.residency_rounds < 2:
        raise RuntimeError("residency-rounds must be at least 2")
    if not 1 <= args.waves_per_simd <= MAX_WAVES_PER_SIMD:
        raise RuntimeError(
            f"waves-per-simd must be in [1, {MAX_WAVES_PER_SIMD}]"
        )
    if args.schedule == "2stage_barrier":
        invalid_waves = [
            waves
            for waves in getattr(args, "waves", ())
            if waves < 8
        ]
        if getattr(args, "waves_per_workgroup", 8) < 8 or invalid_waves:
            raise RuntimeError(
                "2stage_barrier requires at least 8 waves/workgroup"
            )
    if args.workgroups is not None:
        if args.workgroups < 1:
            raise RuntimeError("workgroups must be positive")
        if not args.single_dispatch:
            raise RuntimeError("workgroups requires single-dispatch")
    if not 0 <= args.max_initial_vram_percent <= 100:
        raise RuntimeError("max-initial-vram-percent must be in [0, 100]")
    buffer_bytes = args.buffer_mib * 1024 * 1024
    if (
        not 256 <= args.buffer_mib <= 2048
        or buffer_bytes & (buffer_bytes - 1)
    ):
        raise RuntimeError(
            "buffer-mib must be a power of two in [256, 2048]"
        )


def _run_benchmarks(args, cases):
    _validate_args(args)
    for waves_per_block, _, _, _ in cases:
        _resident_workgroups_per_cu(
            waves_per_block, args.waves_per_simd
        )
    visible_device = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible_device != str(args.physical_device):
        raise RuntimeError(
            "HIP_VISIBLE_DEVICES must equal --physical-device"
        )

    state_helper = _load_state_helper()
    original = state_helper.read_gpu_state(
        args.physical_device, args.amdsmi_root
    )
    if original["performance_level"] != "auto":
        raise RuntimeError(f"GPU must start in auto: {original}")
    if (
        original["gpu_busy_percent"] > 5
        or original["vram_allocated_percent"]
        > args.max_initial_vram_percent
    ):
        raise RuntimeError(f"GPU is not idle: {original}")
    if abs(original["power_cap_w"] - EXPECTED_POWER_CAP_W) > 0.5:
        raise RuntimeError(f"expected 650 W power cap: {original}")

    payload = None
    restored = None
    state_change_attempted = False
    try:
        state_change_attempted = True
        state_helper.set_experiment_state(
            args.physical_device, args.amdsmi_root
        )
        managed = state_helper.read_gpu_state(
            args.physical_device, args.amdsmi_root
        )
        if (
            managed["performance_level"] != "perf_determinism"
            or managed["ptl_state"] != "Enabled"
            or managed["ptl_format"] != "VECTOR,F8"
        ):
            raise RuntimeError(f"failed to set experiment state: {managed}")
        torch.cuda.set_device(args.device)
        properties = torch.cuda.get_device_properties(args.device)
        if not properties.gcnArchName.startswith("gfx94"):
            raise RuntimeError(
                f"gfx94x required, got {properties.gcnArchName}"
            )

        max_epochs = (
            1
            if args.single_dispatch
            else args.workgroups_per_cu
        )
        data_bytes = args.buffer_mib * 1024 * 1024 * max_epochs
        workspace_copies = 2 if _read_count(
            max(case[2] for case in cases), args.vmem_op
        ) < max(case[2] for case in cases) else 1
        free_bytes, _ = torch.cuda.mem_get_info(args.device)
        required_bytes = data_bytes * workspace_copies
        if required_bytes > int(free_bytes * 0.9):
            raise RuntimeError(
                f"benchmark needs {required_bytes / 2**30:.1f} GiB, "
                f"but only {free_bytes / 2**30:.1f} GiB is free"
            )
        primary = torch.empty(
            data_bytes // 4, dtype=torch.int32, device="cuda"
        )
        if workspace_copies == 2:
            data = primary
            output = torch.empty_like(primary)
        else:
            data = primary
            output = primary
        results = [
            _benchmark_case(args, case, data, output, properties)
            for case in cases
        ]
        payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "device": {
                "name": properties.name,
                "arch": properties.gcnArchName,
                "cu_count": properties.multi_processor_count,
                "physical_device": args.physical_device,
            },
            "method": {
                "schedule": args.schedule,
                "cache_policy": args.cache_policy,
                "requested_waves_per_simd": args.waves_per_simd,
                "mfma_opcode": MFMA_OPCODE,
                "mfma_flops": FLOPS_PER_MFMA,
                "mfma_execution_cycles": MFMA_EXECUTION_CYCLES,
                "realtime_clock_mhz": REALTIME_CLOCK_MHZ,
                "frequency_measurement": (
                    "effective_sclk_mhz = delta(s_memtime) * 100 / "
                    "delta(s_memrealtime), recorded inside every timed dispatch"
                ),
                "theoretical_fp8_tflops": (
                    "mfma_flops / mfma_execution_cycles * CU_count * "
                    "4_SIMD_per_CU * effective_sclk_mhz / 1e6"
                ),
                "end_to_end_attainment": (
                    "event-based measured_tflops / in-kernel "
                    "frequency-scaled theoretical_fp8_tflops"
                ),
                "stage0": (
                    "rolling issue-one/immediate-vmcnt(V0)/consume/MFMA "
                    "groups"
                    if args.schedule == "interleave"
                    else f"issue V0 {args.cache_policy} VMEM operations, "
                    "then wait for the previous batch; the final round "
                    "drains with vmcnt(0) and does not issue another batch"
                ),
                "stage1": (
                    None
                    if args.schedule == "interleave"
                    else "issue C0 consecutive FP8 MFMA instructions"
                ),
                "stage_boundary": args.schedule,
            },
            "initial_state": _compact_state(original),
            "managed_state": _compact_state(managed),
            "results": results,
        }
    finally:
        if state_change_attempted:
            state_helper.restore_experiment_state(
                args.physical_device, args.amdsmi_root, original
            )
        restored = state_helper.read_gpu_state(
            args.physical_device, args.amdsmi_root
        )
        for key in (
            "performance_level",
            "ptl_state",
            "ptl_format",
            "numa_balancing",
        ):
            if restored[key] != original[key]:
                raise RuntimeError(
                    f"GPU state restoration mismatch for {key}: "
                    f"{original[key]!r} -> {restored[key]!r}"
                )
    payload["restored_state"] = _compact_state(restored)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON: {args.json}")


def _run_pmc(args):
    if args.rounds < 1 or args.c0 < 1 or args.v0 < 1:
        raise RuntimeError("rounds, C0, and V0 must be positive")
    if args.workgroups_per_cu < 1:
        raise RuntimeError("workgroups-per-cu must be positive")
    if (
        args.schedule == "2stage_barrier"
        and args.waves_per_workgroup < 8
    ):
        raise RuntimeError(
            "2stage_barrier requires at least 8 waves/workgroup"
        )
    _resident_workgroups_per_cu(
        args.waves_per_workgroup, args.waves_per_simd
    )
    buffer_bytes = args.buffer_mib * 1024 * 1024
    if (
        not 256 <= args.buffer_mib <= 2048
        or buffer_bytes & (buffer_bytes - 1)
    ):
        raise RuntimeError(
            "buffer-mib must be a power of two in [256, 2048]"
        )
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    if not properties.gcnArchName.startswith("gfx94"):
        raise RuntimeError(f"gfx94x required, got {properties.gcnArchName}")

    blocks = properties.multi_processor_count * args.workgroups_per_cu
    epoch_bytes = args.buffer_mib * 1024 * 1024
    stream_count = _streams_per_epoch(
        args.waves_per_workgroup, args.pattern, blocks
    )
    vmem_batches = args.rounds
    bytes_per_stream = vmem_batches * args.v0 * BYTES_PER_WAVE_OP
    if stream_count * bytes_per_stream > epoch_bytes:
        raise RuntimeError("PMC private footprint exceeds the address epoch")
    data = torch.empty(epoch_bytes // 4, dtype=torch.int32, device="cuda")
    output = (
        torch.empty_like(data)
        if _read_count(args.v0, args.vmem_op) < args.v0
        else data
    )
    metadata = torch.zeros(
        (blocks * args.waves_per_workgroup, METADATA_DWORDS),
        dtype=torch.int32,
        device="cuda",
    )
    artifact = _launch(
        blocks,
        args.waves_per_workgroup,
        args.rounds,
        args.c0,
        args.v0,
        args.schedule,
        args.waves_per_simd,
        args.vmem_op,
        args.pattern,
        args.cache_policy,
        epoch_bytes,
        False,
        True,
        data,
        output,
        metadata,
    )
    torch.cuda.synchronize()
    register_occupancy = _register_occupancy(
        artifact, args.waves_per_workgroup
    )
    register_occupancy["driver"] = _driver_occupancy(
        artifact, args.waves_per_workgroup
    )
    actual_waves_per_simd = _effective_waves_per_simd(
        register_occupancy, args.waves_per_simd
    )

    read_operations = sum(
        _is_read(args.vmem_op, operation_index)
        for operation_index in range(args.v0)
    )
    write_operations = args.v0 - read_operations
    expected = {
        "workgroups": blocks,
        "waves": blocks * args.waves_per_workgroup,
        "vmem_batches": vmem_batches,
        "mfma_instructions": (
            blocks
            * args.waves_per_workgroup
            * args.rounds
            * args.c0
        ),
        "read_vmem_instructions": (
            blocks
            * args.waves_per_workgroup
            * vmem_batches
            * read_operations
        ),
        "write_vmem_instructions": (
            blocks
            * args.waves_per_workgroup
            * vmem_batches
            * write_operations
        ),
        "logical_read_bytes": (
            blocks
            * args.waves_per_workgroup
            * vmem_batches
            * read_operations
            * BYTES_PER_WAVE_OP
        ),
        "logical_write_bytes": (
            blocks
            * args.waves_per_workgroup
            * vmem_batches
            * write_operations
            * BYTES_PER_WAVE_OP
        ),
    }
    payload = {
        "config": {
            "waves_per_workgroup": args.waves_per_workgroup,
            "c0": args.c0,
            "v0": args.v0,
            "schedule": args.schedule,
            "requested_waves_per_simd": args.waves_per_simd,
            "actual_waves_per_simd": actual_waves_per_simd,
            "lds_bytes": _lds_bytes(
                args.waves_per_workgroup, args.waves_per_simd
            ),
            "register_occupancy": register_occupancy,
            "vmem_op": args.vmem_op,
            "pattern": args.pattern,
            "cache_policy": args.cache_policy,
            "rounds": args.rounds,
            "workgroups_per_cu": args.workgroups_per_cu,
            "workgroups": blocks,
            "buffer_mib": args.buffer_mib,
        },
        "expected": expected,
        "effective_sclk_mhz": _effective_sclk_mhz(metadata.cpu()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _read_pmc_csv(path):
    values = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if "wave_stage_tflops" not in row["Kernel_Name"]:
                continue
            values[row["Counter_Name"]] = float(row["Counter_Value"])
    if not values:
        raise RuntimeError(f"no wave_stage_tflops counters in {path}")
    return values


def _run_pmc_analyze(args):
    expected = json.loads(args.expected.read_text(encoding="utf-8"))[
        "expected"
    ]
    sq = _read_pmc_csv(args.sq_csv)
    l2 = _read_pmc_csv(args.l2_csv)
    hbm = _read_pmc_csv(args.hbm_csv)

    logical_read_bytes = expected["logical_read_bytes"]
    hbm_bytes = (
        hbm["TCC_BUBBLE_sum"] * 128
        + (
            hbm["TCC_EA0_RDREQ_sum"]
            - hbm["TCC_BUBBLE_sum"]
            - hbm["TCC_EA0_RDREQ_32B_sum"]
        )
        * 64
        + hbm["TCC_EA0_RDREQ_32B_sum"] * 32
    )
    closure = {
        "sq_mfma_instruction_ratio": (
            sq["SQ_INSTS_MFMA"] / expected["mfma_instructions"]
        ),
        "sq_read_instruction_ratio": (
            sq["SQ_INSTS_VMEM_RD"]
            / expected["read_vmem_instructions"]
        ),
        "sq_wave_ratio": sq["SQ_WAVES"] / expected["waves"],
        "tcp_requests_per_vmem": (
            l2["TCP_TCC_READ_REQ_sum"]
            / expected["read_vmem_instructions"]
        ),
        "l2_hit_rate": (
            l2["TCC_HIT_sum"]
            / (l2["TCC_HIT_sum"] + l2["TCC_MISS_sum"])
        ),
        "hbm_read_bytes": hbm_bytes,
        "hbm_over_logical_ratio": hbm_bytes / logical_read_bytes,
        "hbm_relative_error": (
            hbm_bytes - logical_read_bytes
        )
        / logical_read_bytes,
    }
    checks = {
        "sq_mfma_instructions_exact": math.isclose(
            closure["sq_mfma_instruction_ratio"],
            1.0,
            abs_tol=1e-12,
        ),
        "sq_read_instructions_exact": math.isclose(
            closure["sq_read_instruction_ratio"], 1.0, abs_tol=1e-12
        ),
        "sq_waves_exact": math.isclose(
            closure["sq_wave_ratio"], 1.0, abs_tol=1e-12
        ),
        "tcp_requests_per_vmem_is_8": math.isclose(
            closure["tcp_requests_per_vmem"], 8.0, abs_tol=1e-12
        ),
        "hbm_bytes_within_0_1_percent": abs(
            closure["hbm_relative_error"]
        ) <= 0.001,
    }
    result = {
        "expected": expected,
        "pmc": {"sq": sq, "l2": l2, "hbm": hbm},
        "closure": closure,
        "checks": checks,
        "valid": all(checks.values()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not result["valid"]:
        raise RuntimeError(f"private PMC closure failed: {checks}")


def _self_test():
    assert BYTES_PER_WAVE_OP == 1024
    assert FLOPS_PER_MFMA == 16384
    assert _theoretical_fp8_tflops(80, 1800) == 589.824
    assert _read_count(8, "read") == 8
    assert _read_count(8, "write") == 0
    assert _read_count(9, "mixed") == 5
    assert _read_count(5, "read3_write2") == 3
    assert _read_count(20, "read12_write8") == 12
    assert _parse_vmem_op("read29_write8") == "read29_write8"
    assert _read_count(37, "read29_write8") == 29
    assert _is_read("read29_write8", 28)
    assert not _is_read("read29_write8", 29)
    assert _mfma_partition(24, 5) == (5, 5, 5, 5, 4)
    assert _mfma_partition(96, 20) == (5,) * 16 + (4,) * 4
    assert [_is_read("read3_write2", index) for index in range(5)] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert [_is_read("read12_write8", index) for index in range(20)] == [
        True,
    ] * 12 + [False] * 8
    assert _lds_bytes(4, 1) == 64 * 1024
    assert _lds_bytes(4, 2) == 32 * 1024
    assert _lds_bytes(4, 4) == 16 * 1024
    assert _resident_workgroups_per_cu(4, 4) == 4
    assert _lds_bytes(8, 2) == 64 * 1024
    assert _lds_bytes(8, 4) == 32 * 1024
    assert _resident_workgroups_per_cu(8, 4) == 2
    assert _lds_bytes(16, 4) == 64 * 1024
    assert _resident_workgroups_per_cu(16, 4) == 1
    try:
        _resident_workgroups_per_cu(8, 1)
    except RuntimeError as error:
        assert "cannot produce exactly" in str(error)
    else:
        raise AssertionError("8-wave workgroup unexpectedly accepted Q=1")
    register_occupancy = _register_occupancy(
        {
            "used_gprs": [
                *(f"v{index}" for index in range(108)),
                *(f"a{index}" for index in range(16)),
                *(f"s{index}" for index in range(44)),
            ]
        },
        8,
    )
    assert register_occupancy["vgpr_count"] == 108
    assert register_occupancy["agpr_count"] == 16
    assert register_occupancy["achievable_waves_per_simd"] == 4
    register_occupancy["driver"] = {
        "max_active_workgroups_per_cu": 2,
        "max_waves_per_simd": 4,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _effective_waves_per_simd(register_occupancy, 8) == 4
    assert len(caught) == 1
    warning = str(caught[0].message)
    assert "requested 8 waves/SIMD" in warning
    assert "108 VGPR + 16 AGPR" in warning
    assert "to 4 waves/SIMD" in warning
    assert _stream_count(8, 80, "private", 80) == 640
    assert _stream_count(8, 80, "simd", 80) == 512
    assert _stream_count(16, 80, "stage", 80) == 256
    assert _stream_count(4, 160, "simd", 160) == 512
    assert _stream_count(4, 160, "workgroup", 160) == 160
    assert _stream_count(4, 320, "cu", 160) == 256
    assert _active_streams_per_epoch(16, "simd", 80, 80) == 320
    assert _active_streams_per_epoch(16, "stage", 80, 80) == 160
    assert _active_streams_per_epoch(16, "cu", 80, 80) == 80
    exact_grid_args = _parser().parse_args(
        [
            "bench",
            "--waves-per-workgroup",
            "8",
            "--c0",
            "192",
            "--v0",
            "37",
            "--vmem-op",
            "read29_write8",
            "--workgroups",
            "1024",
            "--single-dispatch",
        ]
    )
    _validate_args(exact_grid_args)
    exact_grid_args.single_dispatch = False
    try:
        _validate_args(exact_grid_args)
    except RuntimeError as error:
        assert "workgroups requires single-dispatch" in str(error)
    else:
        raise AssertionError("exact workgroups accepted without one dispatch")
    for waves in WAVE_MODES:
        assert waves * WAVE_SIZE <= 1024
    print("self-test passed: stream patterns, FLOPs, and LDS modes")


def _add_common_arguments(parser):
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-device", type=int, default=4)
    parser.add_argument("--vmem-op", type=_parse_vmem_op, default="read")
    parser.add_argument("--schedule", choices=SCHEDULES, default="2stage_0")
    parser.add_argument("--waves-per-simd", type=int, default=4)
    parser.add_argument(
        "--cache-policy",
        choices=CACHE_POLICIES,
        default="temporal",
    )
    parser.add_argument("--rounds", type=int, default=128)
    parser.add_argument("--workgroups-per-cu", type=int, default=8)
    parser.add_argument("--workgroups", type=int)
    parser.add_argument("--buffer-mib", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--launches-per-sample", type=int, default=1)
    parser.add_argument("--residency-rounds", type=int, default=8)
    parser.add_argument("--max-initial-vram-percent", type=int, default=20)
    parser.add_argument("--single-dispatch", action="store_true")
    parser.add_argument(
        "--amdsmi-root", type=Path, default=DEFAULT_AMDSMI_ROOT
    )
    parser.add_argument("--json", type=Path)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("bench")
    _add_common_arguments(bench)
    bench.add_argument(
        "--waves-per-workgroup", type=int, choices=WAVE_MODES, required=True
    )
    bench.add_argument("--c0", type=int, required=True)
    bench.add_argument("--v0", type=int, required=True)
    bench.add_argument("--pattern", choices=PATTERNS, default="private")

    sweep = subparsers.add_parser("sweep")
    _add_common_arguments(sweep)
    sweep.add_argument(
        "--waves", nargs="+", type=int, choices=WAVE_MODES,
        default=list(WAVE_MODES),
    )
    sweep.add_argument(
        "--cases", nargs="+", type=_parse_case,
        default=list(DEFAULT_CASES), metavar="C0:V0",
    )
    sweep.add_argument(
        "--patterns", nargs="+", choices=PATTERNS,
        default=["private", "simd"],
    )
    pmc = subparsers.add_parser("pmc-run")
    pmc.add_argument("--device", type=int, default=0)
    pmc.add_argument(
        "--waves-per-workgroup", type=int, choices=WAVE_MODES, required=True
    )
    pmc.add_argument("--c0", type=int, required=True)
    pmc.add_argument("--v0", type=int, required=True)
    pmc.add_argument("--vmem-op", type=_parse_vmem_op, default="read")
    pmc.add_argument("--schedule", choices=SCHEDULES, default="2stage_0")
    pmc.add_argument("--waves-per-simd", type=int, default=4)
    pmc.add_argument(
        "--cache-policy",
        choices=CACHE_POLICIES,
        default="temporal",
    )
    pmc.add_argument("--pattern", choices=PATTERNS, default="private")
    pmc.add_argument("--rounds", type=int, default=128)
    pmc.add_argument("--workgroups-per-cu", type=int, default=1)
    pmc.add_argument("--buffer-mib", type=int, default=2048)
    pmc.add_argument("--warmups", type=int, default=1)
    pmc.add_argument("--samples", type=int, default=1)
    pmc.add_argument("--launches-per-sample", type=int, default=1)
    pmc.add_argument("--residency-rounds", type=int, default=2)
    pmc.add_argument("--json", type=Path)
    pmc_analyze = subparsers.add_parser("pmc-analyze")
    pmc_analyze.add_argument("--expected", type=Path, required=True)
    pmc_analyze.add_argument("--sq-csv", type=Path, required=True)
    pmc_analyze.add_argument("--l2-csv", type=Path, required=True)
    pmc_analyze.add_argument("--hbm-csv", type=Path, required=True)
    pmc_analyze.add_argument("--json", type=Path)
    subparsers.add_parser("self-test")
    return parser


def main():
    args = _parser().parse_args()
    if args.command == "self-test":
        _self_test()
        return
    if args.command == "pmc-run":
        if args.c0 < 1 or args.v0 < 1:
            raise RuntimeError("C0 and V0 must be positive")
        _run_pmc(args)
        return
    if args.command == "pmc-analyze":
        _run_pmc_analyze(args)
        return
    if args.command == "bench":
        if args.c0 < 1 or args.v0 < 1:
            raise RuntimeError("C0 and V0 must be positive")
        cases = [
            (
                args.waves_per_workgroup,
                args.c0,
                args.v0,
                args.pattern,
            )
        ]
    else:
        cases = [
            (waves, c0, v0, pattern)
            for waves in args.waves
            for c0, v0 in args.cases
            for pattern in args.patterns
        ]
    _run_benchmarks(args, cases)


if __name__ == "__main__":
    main()