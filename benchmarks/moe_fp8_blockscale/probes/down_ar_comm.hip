// Device functions of the fused MoE down GEMM + top-k sum + two-shot all-reduce kernel (down_ar_fused.py, TP8,
// HIP IPC), linked into the FlyDSL kernel as LLVM bitcode and inlined there.
//
// One kernel, 256-thread WGs, two per CU. dar_role: the first WG on a CU computes; a second WG becomes a comm WG
// while its XCD's comm quota lasts (NCOMM / 8 per XCD), else it computes too. Compute WGs (4 waves, FlyDSL down
// body) take (column group, M-block) tasks column-group-major from a per-XCD queue (dar_claim); the XCD's last task
// of a group writes back the XCD's L2 (one agent-scope release covers every task of the XCD) and counts the XCD in
// ready[cg] (dar_publish). Comm WGs (dar_comm) run, per column group cg once every XCD has finished it, the
// sum_ar_cols.hip scheme on its CW columns:
//   RS: wave pair p (sum wave p, push wave p + 2) takes rows seq = p, p + 2, ... of the WG's (unit, shard)
//       sequence; the sum wave adds the TOPK route rows in fp32 (k order), rounds to bf16 and fills an LDS ring
//       slot, the push wave writes the slot to the owner's inbox (sc0|sc1) and, one row later, its rs flag.
//   AG: waves independently: wait rs[cg][*][u], fp32 rank-order sum of the inbox pieces, bf16, push to every
//       rank's gather row, then the ag flags there.
// End: wait ag[cg][*][u] for the WG's units of every group, a barrier over the comm WGs, a system acquire.
// Flags hold the launch epoch (monotonic per call on every rank); ctl (local counters) is zeroed by the host.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define H 4096
#ifndef CW
#define CW 512  // column group width; HP 16-byte pieces per lane and row
#endif
#define HP (CW / 512)
#define NCG (H / CW)
#define NPES 8
#define TOPK 8
#define NSLOT 4
#define NPAIR 2
#define SYS 17  // sc0|sc1
#define NT 2
#define BM 256

#define C_CU 0        // [8][256] WG arrivals per CU, key = XCC_ID * 256 + HW_ID[15:8]
#define C_COMM 2048   // [8] comm WGs per XCD
#define C_Q 2056      // [8] task queue head per XCD
#define C_READY 2064  // [NCG] XCDs that finished the group
#define C_DONE 2072   // comm WGs that passed the end
#define C_ERR 2073    // sticky code of a wait that hit the 20 s guard: 1 rs, 2 ag, 3 barrier, 4/5 ring, 6 ready
#define C_CIDX 2216   // [8] compute WGs per XCD (after the C_DBG timeline)
#define C_XR 2224     // [8][NCG] finished compute tasks per XCD and column group
#define C_DBG 2080    // u64 timeline (s_memrealtime): ~start, ready, ~seen, rs, ag [NCG each], fin, compute end
#define D_START 0
#define D_READY 1
#define D_SEEN (1 + NCG)
#define D_RS (1 + 2 * NCG)
#define D_AG (1 + 3 * NCG)
#define D_FIN (1 + 4 * NCG)
#define D_CEND (2 + 4 * NCG)
#define C_TASK 2560   // [8 XCD][640 tasks][4] u32: claim time, done time, CU key, 0 (low 32 bits of s_memrealtime)
#define MAXT 640

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) int lint;
typedef __attribute__((address_space(3))) u32x4 lv4;
typedef const __attribute__((address_space(4))) int cint;  // scalar loads: lgkmcnt, not the in-order vmcnt
typedef const __attribute__((address_space(4))) uint64_t cu64;

struct Layout {  // byte offsets in every rank's symmetric buffer (sized for tmax tokens)
    uint64_t inbox, gather, rs, ag, total;
};

__device__ __forceinline__ Layout layout(int tmax) {
    Layout L;
    const uint64_t rmax = tmax / NPES;
    L.inbox = 0;                                       // [NPES src][rmax][H] bf16
    L.gather = (uint64_t)tmax * H * 2;                 // [tmax][H] bf16: the all-reduced output
    L.rs = L.gather + (uint64_t)tmax * H * 2;          // [NCG][NPES src][rmax] i32
    L.ag = L.rs + (uint64_t)NCG * NPES * rmax * 4;     // [NCG][tmax] i32
    L.total = L.ag + (uint64_t)NCG * tmax * 4;
    return L;
}

__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(uint64_t base, int n = 0x7FFFFFFF) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, (short)0, n, 0x00020000);
}
__device__ __forceinline__ int tid() { return __builtin_amdgcn_workitem_id_x(); }
__device__ __forceinline__ unsigned xcc_id() {
    unsigned x;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(x));
    return x & 7;
}
__device__ __forceinline__ uint64_t now() { return __builtin_amdgcn_s_memrealtime(); }
__device__ __forceinline__ void fail(int* ctl, int code) {
    __hip_atomic_store(&ctl[C_ERR], code, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void flag_put(uint64_t addr, int epoch) {
    __hip_atomic_exchange((int*)addr, epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// 20 s at 100 MHz: hang guard only
__device__ __forceinline__ void flag_wait(uint64_t addr, int target, int* ctl, int code) {
    const uint64_t t0 = now();
    while (__hip_atomic_load((int*)addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < target) {
        __builtin_amdgcn_s_sleep(1);
        if (now() - t0 > 2000000000ull) {
            fail(ctl, code);
            return;
        }
    }
}
__device__ __forceinline__ void count_wait(int* p, int target, int* ctl, int code) {
    const uint64_t t0 = now();
    while (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
        __builtin_amdgcn_s_sleep(2);
        if (now() - t0 > 2000000000ull) {
            fail(ctl, code);
            return;
        }
    }
}
__device__ __forceinline__ void lds_wait(lint* p, int v, int* ctl, int code) {
    const uint64_t t0 = now();
    while (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) < v) {
        __builtin_amdgcn_s_sleep(0);
        if (now() - t0 > 2000000000ull) {
            fail(ctl, code);
            return;
        }
    }
}
__device__ __forceinline__ void stamp(int* ctl, int slot, uint64_t v) {  // max; min slots store ~t
    __hip_atomic_fetch_max((unsigned long long*)(ctl + C_DBG) + slot, (unsigned long long)v, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void add8(float* acc, u32x4 w) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc[2 * j] += __uint_as_float(w[j] << 16);
        acc[2 * j + 1] += __uint_as_float(w[j] & 0xFFFF0000u);
    }
}
__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {  // RNE, finite inputs
    uint32_t a = __float_as_uint(lo), b = __float_as_uint(hi);
    a = (a + 0x7FFFu + ((a >> 16) & 1)) >> 16;
    b = (b + 0x7FFFu + ((b >> 16) & 1)) & 0xFFFF0000u;
    return a | b;
}
__device__ __forceinline__ u32x4 pack(const float* acc) {
    u32x4 r = {bf16x2(acc[0], acc[1]), bf16x2(acc[2], acc[3]), bf16x2(acc[4], acc[5]), bf16x2(acc[6], acc[7])};
    return r;
}
__device__ __forceinline__ int bcast(lint* s, int v) {  // thread 0's v to the whole WG
    if (tid() == 0) s[0] = v;
    __syncthreads();
    const int r = s[0];
    __syncthreads();
    return r;
}

// -1: compute WG, -2: idle (beyond cpx compute WGs on this XCD); else the comm WG index (XCD-major, ncomm / 8 per
// XCD).
extern "C" __device__ __attribute__((always_inline)) int dar_role(long ctl_, int ncomm, int cpx, int lds) {
    int* ctl = (int*)ctl_;
    int role = -1;
    if (tid() == 0) {
        unsigned hw;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw));
        const unsigned x = xcc_id();
        const int per = ncomm / 8;
        stamp(ctl, D_START, ~now());
        if (__hip_atomic_fetch_add(&ctl[C_CU + x * 256 + ((hw >> 8) & 0xFF)], 1, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT) > 0) {
            const int k = __hip_atomic_fetch_add(&ctl[C_COMM + x], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if (k < per) role = x * per + k;
        }
        if (role == -1 &&
            __hip_atomic_fetch_add(&ctl[C_CIDX + x], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= cpx)
            role = -2;
    }
    return bcast((lint*)(size_t)lds, role);
}

extern "C" __device__ __attribute__((always_inline)) int dar_xcc() { return xcc_id(); }

// Next task index of this XCD's queue (uniform over the WG).
extern "C" __device__ __attribute__((always_inline)) int dar_claim(long ctl_, int lds) {
    int* ctl = (int*)ctl_;
    int i = 0;
    if (tid() == 0) {
        const unsigned x = xcc_id();
        i = __hip_atomic_fetch_add(&ctl[C_Q + x], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (i < MAXT) {
            unsigned hw;
            asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw));
            ctl[C_TASK + (x * MAXT + i) * 4] = (int)now();
            ctl[C_TASK + (x * MAXT + i) * 4 + 2] = (int)((hw >> 8) & 0xFF);
        }
    }
    return bcast((lint*)(size_t)lds, i);
}

// After a task's Y stores: every wave's stores reached L2; the XCD's last task of the group releases (agent scope:
// L2 write-back) and counts the XCD in ready[cg]. wbl2: 0 none (stores were write-through), 1 every task, 2 once
// per XCD and group (one write-back covers the whole XCD's L2).
extern "C" __device__ __attribute__((always_inline)) void dar_publish(long ctl_, int cg, int wbl2, int nb, int nx,
                                                                    int i) {
    int* ctl = (int*)ctl_;
    __builtin_amdgcn_s_waitcnt(0x0f70);  // vmcnt(0)
    __syncthreads();
    if (tid() == 0) {
        const unsigned x = xcc_id();
        if (wbl2 == 1) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        if (__hip_atomic_fetch_add(&ctl[C_XR + x * NCG + cg], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1 ==
            nb) {
            if (wbl2 == 2) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            if (__hip_atomic_fetch_add(&ctl[C_READY + cg], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1 == nx)
                stamp(ctl, D_READY + cg, now());
        }
        if (i < MAXT) ctl[C_TASK + (x * MAXT + i) * 4 + 1] = (int)now();
    }
}

extern "C" __device__ __attribute__((always_inline)) void dar_cend(long ctl_) {
    if (tid() == 0) stamp((int*)ctl_, D_CEND, now());
}

// mode bits: 4 = all-reduce a ready partial y_ = P [T][H] (one row per token, no top-k sum), 8 = do not wait for
// the down groups (probes: how fast is the exchange alone / concurrent with the down GEMM).
template <bool PARTIAL>
__device__ __forceinline__ void comm_body(long y_, long inv_, long peers_, long ctl_, long nv_, int T, int tmax,
                                          int rank, int epoch, int lds, int ncomm, int c, int mode) {
    int* ctl = (int*)ctl_;
    cu64* peers = (cu64*)peers_;
    cint* inv = (cint*)inv_;
    lv4* ring = (lv4*)(size_t)lds;                        // [NPAIR][NSLOT][CW / 8]
    lint* prod = (lint*)(size_t)(lds + NPAIR * NSLOT * CW * 2);
    lint* cons = prod + NPAIR;
    const Layout L = layout(tmax);
    const int R = T / NPES, rmax = tmax / NPES, tx = tid(), wave = tx >> 6, lane = tx & 63, p = wave & 1;
    const uint64_t own = peers[rank];
    const int ntile = *(cint*)nv_, nx = ntile < 8 ? ntile : 8;  // nv_: the compact tile count
    constexpr bool partial = PARTIAL;
    const bool gated = !(mode & 8);
#ifdef PRIO
    __builtin_amdgcn_s_setprio(PRIO);
#endif
    const int rows = ((R - c + ncomm - 1) / ncomm) * NPES;  // (unit, shard) rows of this WG per group
    const int npair = (rows - p + 1) / 2;                    // of them for pair p
    const __amdgpu_buffer_rsrc_t yr = rsrc((uint64_t)y_, -1);  // Y is 2.4 GB at 32K tokens: full 4 GB range
    const __amdgpu_buffer_rsrc_t inbox = rsrc(own + L.inbox);
    const int gw = c * 4 + wave, nw = ncomm * 4;
    if (tx < NPAIR) prod[tx] = cons[tx] = 0;
    __syncthreads();

    for (int cg = 0; cg < NCG; ++cg) {
        const uint32_t col = (cg * (CW / 8) + lane) * 16;  // this lane's first 16 B of the group; second at +1 KiB
        const int base = cg * npair;                        // ring slots used by pair p before this group
        if (tx == 0) {
            if (gated) count_wait(&ctl[C_READY + cg], nx, ctl, 6);
            stamp(ctl, D_SEEN + cg, ~now());
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

        // ---- RS ----------------------------------------------------------------------------------------------
        if (wave < 2) {
            u32x4 buf[2][TOPK][HP];
            auto fetch = [&](int seq, u32x4 (&dst)[TOPK][HP]) {
                const int u = c + (seq / NPES) * ncomm, t = ((rank + u + seq % NPES) % NPES) * R + u;
                if constexpr (partial) {
#pragma unroll
                    for (int h = 0; h < HP; ++h)
                        dst[0][h] = __builtin_amdgcn_raw_buffer_load_b128(yr, (uint32_t)t * (H * 2) + col + h * 1024,
                                                                          0, NT);
                    return;
                }
#pragma unroll
                for (int k = 0; k < TOPK; ++k) {
                    const uint32_t row = (uint32_t)inv[t * TOPK + k] * (H * 2);
#pragma unroll
                    for (int h = 0; h < HP; ++h)
                        dst[k][h] = __builtin_amdgcn_raw_buffer_load_b128(yr, row + col + h * 1024, 0, NT);
                }
            };
            if (p < rows) fetch(p, buf[0]);
            for (int seq = p, n = 0; seq < rows; seq += 4, n += 2) {
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const int cur = seq + 2 * b;
                    if (cur >= rows) break;
                    if (cur + 2 < rows) fetch(cur + 2, buf[b ^ 1]);
                    u32x4 out[HP];
#pragma unroll
                    for (int h = 0; h < HP; ++h) {
                        out[h] = buf[b][0][h];
                        if constexpr (!partial) {
                            float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                            for (int k = 0; k < TOPK; ++k) add8(acc, buf[b][k][h]);
                            out[h] = pack(acc);
                        }
                    }
                    const int slot = base + n + b;
                    lds_wait(&cons[p], slot - NSLOT + 1, ctl, 4);
#pragma unroll
                    for (int h = 0; h < HP; ++h) ring[(p * NSLOT + slot % NSLOT) * (CW / 8) + lane + h * 64] = out[h];
                    __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0)
                    if (lane == 0)
                        __hip_atomic_store(&prod[p], slot + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
                }
            }
        } else {
            uint64_t pending = 0;  // rs flag of the previous row, set once its stores are acknowledged
            for (int seq = p, n = 0; seq < rows; seq += 2, ++n) {
                const int u = c + (seq / NPES) * ncomm, s = (rank + u + seq % NPES) % NPES, slot = base + n;
                lds_wait(&prod[p], slot + 1, ctl, 5);
                u32x4 v[HP];
#pragma unroll
                for (int h = 0; h < HP; ++h) v[h] = ring[(p * NSLOT + slot % NSLOT) * (CW / 8) + lane + h * 64];
                __builtin_amdgcn_s_waitcnt(0xc07f);
                if (lane == 0) __hip_atomic_store(&cons[p], slot + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
                const __amdgpu_buffer_rsrc_t d = rsrc(peers[s] + L.inbox + ((uint64_t)rank * rmax + u) * H * 2);
#pragma unroll
                for (int h = 0; h < HP; ++h) __builtin_amdgcn_raw_buffer_store_b128(v[h], d, col + h * 1024, 0, SYS);
                if (pending) {
                    __builtin_amdgcn_s_waitcnt(0x0f70 | HP);  // vmcnt(HP): the previous row's stores are acknowledged
                    if (lane == 0) flag_put(pending, epoch);
                }
                pending = peers[s] + L.rs + (((uint64_t)cg * NPES + rank) * rmax + u) * 4;
            }
            if (pending) {
                __builtin_amdgcn_s_waitcnt(0x0f70);
                if (lane == 0) flag_put(pending, epoch);
            }
            if (lane == 0) stamp(ctl, D_RS + cg, now());
        }

        // ---- AG (waves independent) ------------------------------------------------------------------------
        for (int u = gw; u < R; u += nw) {
            if (lane < NPES) flag_wait(own + L.rs + (((uint64_t)cg * NPES + lane) * rmax + u) * 4, epoch, ctl, 1);
            u32x4 w[NPES][HP];
#pragma unroll
            for (int src = 0; src < NPES; ++src)
#pragma unroll
                for (int h = 0; h < HP; ++h)
                    w[src][h] = __builtin_amdgcn_raw_buffer_load_b128(
                        inbox, (uint32_t)(((uint64_t)src * rmax + u) * H * 2) + col + h * 1024, 0, SYS);
            u32x4 red[HP];
#pragma unroll
            for (int h = 0; h < HP; ++h) {
                float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int src = 0; src < NPES; ++src) add8(acc, w[src][h]);
                red[h] = pack(acc);
            }
            const uint64_t row = (uint64_t)rank * R + u;
#pragma unroll
            for (int i = 0; i < NPES; ++i) {
                const __amdgpu_buffer_rsrc_t d = rsrc(peers[(rank + u + i) % NPES] + L.gather + row * H * 2);
#pragma unroll
                for (int h = 0; h < HP; ++h) __builtin_amdgcn_raw_buffer_store_b128(red[h], d, col + h * 1024, 0, SYS);
            }
            __builtin_amdgcn_s_waitcnt(0x0f70);
            if (lane < NPES) flag_put(peers[(rank + lane) % NPES] + L.ag + ((uint64_t)cg * tmax + row) * 4, epoch);
        }
        if (lane == 0) stamp(ctl, D_AG + cg, now());
    }

    // ---- every group landed everywhere --------------------------------------------------------------------------
    for (int cg = 0; cg < NCG; ++cg)
        for (int u = gw; u < R; u += nw)
            if (lane < NPES) flag_wait(own + L.ag + ((uint64_t)cg * tmax + (uint64_t)lane * R + u) * 4, epoch, ctl, 2);
    __syncthreads();
    if (tx == 0) {
        __hip_atomic_fetch_add(&ctl[C_DONE], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        count_wait(&ctl[C_DONE], ncomm, ctl, 3);
        stamp(ctl, D_FIN, now());
    }
    __syncthreads();
    if (tx < 64) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

extern "C" __device__ __attribute__((always_inline)) void dar_comm(long y_, long inv_, long peers_, long ctl_, long nv_,
                                                                 int T, int tmax, int rank, int epoch, int lds,
                                                                 int ncomm, int c, int mode) {
    if (mode & 4)
        comm_body<true>(y_, inv_, peers_, ctl_, nv_, T, tmax, rank, epoch, lds, ncomm, c, mode);
    else
        comm_body<false>(y_, inv_, peers_, ctl_, nv_, T, tmax, rank, epoch, lds, ncomm, c, mode);
}
