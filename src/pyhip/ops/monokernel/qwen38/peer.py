"""Uncached device memory shared by the tensor-parallel ranks over XGMI, for the in-launch all-reduce of
the fused layer launches (``flydsl/xgmi.py``).

Every rank allocates one zeroed region with ``hipExtMallocWithFlags(hipDeviceMallocUncached)`` (as AITER's
all-reduce signals), exports it with ``hipIpcGetMemHandle`` and maps the other ranks' regions with
``hipIpcOpenMemHandle``; the handles travel over a CPU process group. ``table`` is the device table the
kernels read: int64 region base of every rank (as mapped in this process), then this rank's index. The
regions live as long as the process.
"""

from __future__ import annotations

import ctypes
from functools import cache

import torch
import torch.distributed as dist

UNCACHED = 0x3  # hipDeviceMallocUncached
LAZY_PEER_ACCESS = 0x1  # hipIpcMemLazyEnablePeerAccess
MAX_RANKS = 8
TABLE_WORDS = 16  # int64: bases[MAX_RANKS], rank, padding


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


def _loaded_hip() -> str:
    """Path of the HIP runtime this process already uses (torch's), so the calls share its context."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = line.split()[-1]
            if "libamdhip64.so" in path:
                return path
    raise RuntimeError("libamdhip64 is not loaded (import torch and initialize the GPU first)")


@cache
def _hip():
    lib = ctypes.CDLL(_loaded_hip())
    lib.hipExtMallocWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
    lib.hipDeviceSynchronize.argtypes = []
    for name in ("hipExtMallocWithFlags", "hipMemset", "hipIpcGetMemHandle", "hipIpcOpenMemHandle",
                 "hipDeviceSynchronize"):
        getattr(lib, name).restype = ctypes.c_int
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    return lib


def _check(error: int, what: str):
    if error != 0:
        raise RuntimeError(f"{what}: {_hip().hipGetErrorString(error).decode()}")


class PeerMemory:
    """``size`` bytes of zeroed uncached memory on every rank of ``group`` (a CPU group, e.g. SGLang's
    ``get_tp_group().cpu_group``; ``None``: this process alone), mapped into every rank; ``table`` [16] int64
    on ``device``."""

    def __init__(self, size: int, group, device: torch.device):
        lib = _hip()
        self.size, self.device = size, torch.device(device)
        alone = group is None
        self.rank = 0 if alone else dist.get_rank(group=group)
        self.world = 1 if alone else dist.get_world_size(group=group)
        assert self.world <= MAX_RANKS
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            local = ctypes.c_void_p()
            _check(lib.hipExtMallocWithFlags(ctypes.byref(local), size, UNCACHED), "hipExtMallocWithFlags")
            _check(lib.hipMemset(local, 0, size), "hipMemset")
            _check(lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
            handle = _IpcHandle()
            handles = [None]
            if not alone:
                _check(lib.hipIpcGetMemHandle(ctypes.byref(handle), local), "hipIpcGetMemHandle")
                handles = [None] * self.world
                dist.all_gather_object(handles, bytes(handle), group=group)
            bases = []
            for r, raw in enumerate(handles):
                if r == self.rank:
                    bases.append(local.value)
                    continue
                mapped = ctypes.c_void_p()
                _check(lib.hipIpcOpenMemHandle(ctypes.byref(mapped), _IpcHandle.from_buffer_copy(raw),
                                               LAZY_PEER_ACCESS), f"hipIpcOpenMemHandle(rank {r})")
                bases.append(mapped.value)
        self.bases = bases
        words = bases + [0] * (MAX_RANKS - self.world) + [self.rank] + [0] * (TABLE_WORDS - MAX_RANKS - 1)
        self.table = torch.tensor(words, dtype=torch.int64, device=self.device)
        # every rank's region is zeroed and mapped before any kernel signals into it
        if not alone:
            dist.barrier(group=group)
