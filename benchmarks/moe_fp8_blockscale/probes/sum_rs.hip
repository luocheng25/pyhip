// MoE top-k sum fused with a reduce-scatter over 8 GPUs (MI300X, HIP IPC symmetric buffers), for the
// sequence-parallel MoE layout: rank s owns token rows [off[s], off[s + 1]) of the gathered batch (shard
// sizes may differ, e.g. two DP groups with different token counts).
//
// Every rank holds Y [rows, H] bf16 (its I-slice route rows, route weight folded in) and INV [T, TOPK]
// (sorted row of route (t, k)). out[u] (rank s, u < size[s]) = bf16(sum over ranks in rank order of
// bf16(sum_k Y[INV[off[s] + u, k]])). Unit u = row u of every shard. Persistent grid of 512-thread WGs (every
// WG resident), units u = blockIdx.x + j * gridDim.x < max size; (u, s) pairs with u >= size[s] are skipped.
//   RS: waves 0-3 sum the TOPK rows of token (unit u, destination s rotated by u) in fp32, round to bf16 and
//      write a quarter row into an LDS ring; waves 4-7 copy it to rank s's inbox[parity][rank][u] with
//      write-through stores and, after a unit, set rs[rank][u][q] = epoch on its destinations (monotonic
//      flags, never reset).
//   reduce: wait rs[*][u][*] for this rank's units, sum the NPES inbox rows in fp32 in rank order
//      (L2-bypassing loads), bf16, store to out[u] (local).
// RS never waits on another rank, so there is no deadlock with all WGs resident. A peer can be at most one
// epoch ahead (it needs our epoch-e+1 pushes to finish epoch e+1), so the inbox alternates by epoch parity.
// Inbox and flag rows of destination s are indexed src * size[s] + u (the layout holds RCAP rows per source).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define H 4096
#define NPES 8
#define TOPK 8
#define THREADS 512
#define QV (H / 8 / 4)  // 16 B vectors per row quarter (one sum/push wave pair)
#define NSLOT 4         // LDS ring rows (8 KiB each)
#define SYS 17          // sc0|sc1
#define NT 2

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef const __attribute__((address_space(4))) int cint;  // scalar loads: lgkmcnt

struct Layout {  // byte offsets in every rank's symmetric buffer (rcap rows per source)
    uint64_t inbox, rs, total;
};

__host__ __device__ inline Layout layout(int rcap) {
    Layout L;
    L.inbox = 0;                                               // [2 parity][NPES src][rcap][H] bf16
    L.rs = L.inbox + 2ull * NPES * (uint64_t)rcap * H * 2;     // [NPES src][rcap][4 quarters] i32
    L.total = L.rs + (uint64_t)NPES * rcap * 16;
    return L;
}

struct Args {
    uint64_t base[NPES];  // every rank's symmetric buffer, mapped in this process
    int off[NPES + 1];    // shard row offsets in the gathered batch
};

__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(uint64_t base) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, (short)0, 0x7FFFFFFF, 0x00020000);
}
__device__ __forceinline__ void flag_put(uint64_t addr, int epoch) {
    __hip_atomic_exchange((int*)addr, epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void flag_wait(uint64_t addr, int epoch) {
    while (__hip_atomic_load((int*)addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < epoch)
        __builtin_amdgcn_s_sleep(1);
}
__device__ __forceinline__ void lds_wait(int* p, int v) {
    while (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) < v) __builtin_amdgcn_s_sleep(0);
}
__device__ __forceinline__ void lds_set(int* p, int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
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

// MODE 0: Y + INV (fused top-k sum); MODE 1: Y is a ready partial [T, H] (reduce-scatter only).
// tim[blockIdx][3] (optional): s_memrealtime at start, RS end, reduce end.
template <int MODE>
__global__ void __launch_bounds__(THREADS) sum_rs(const uint16_t* __restrict__ y, const int* __restrict__ inv,
                                                  uint16_t* __restrict__ out, Args a, int rmax, int rcap,
                                                  int rank, int epoch, uint64_t* tim) {
    constexpr int NK = MODE ? 1 : TOPK;
    __shared__ u32x4 ring[NSLOT][H / 8];
    __shared__ int prod[4], cons[4];
    const Layout L = layout(rcap);
    const int tx = threadIdx.x, wave = tx >> 6, lane = tx & 63, q = wave & 3;
    const uint64_t inbox_off = L.inbox + (uint64_t)(epoch & 1) * NPES * rcap * H * 2;
    const int units = (rmax - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x, rows = units * NPES;
    auto dest = [&](int seq, int& u, int& s) {  // seq -> (unit, destination); false if u is past s's shard
        u = blockIdx.x + (seq / NPES) * gridDim.x;
        s = (rank + u + seq % NPES) % NPES;
        return u < a.off[s + 1] - a.off[s];
    };
    if (tx < 4) prod[tx] = cons[tx] = 0;
    if (tim && tx == 0) tim[blockIdx.x * 3] = __builtin_amdgcn_s_memrealtime();
    __syncthreads();

    // ---- RS ----------------------------------------------------------------------------------------------
    if (wave < 4) {  // sum role: Y loads (prefetched one row ahead) -> fp32 sum -> bf16 -> LDS quarter
        cint* sinv = (cint*)inv;
        u32x4 buf[2][NK][2];
        auto fetch = [&](int seq, u32x4 (&dst)[NK][2]) {
            int u, s;
            if (!dest(seq, u, s)) return;
            int t = a.off[s] + u;
#pragma unroll
            for (int k = 0; k < NK; ++k) {
                long row = MODE ? t : sinv[t * TOPK + k];
                __amdgpu_buffer_rsrc_t r = rsrc((uint64_t)(y + row * H));
#pragma unroll
                for (int h = 0; h < 2; ++h)
                    dst[k][h] = __builtin_amdgcn_raw_buffer_load_b128(r, (q * QV + lane + h * 64) * 16, 0, NT);
            }
        };
        if (rows) fetch(0, buf[0]);
        for (int seq = 0; seq < rows; seq += 2) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                int cur = seq + p, u, s;
                if (cur >= rows) break;
                if (cur + 1 < rows) fetch(cur + 1, buf[p ^ 1]);
                if (!dest(cur, u, s)) {
                    if (lane == 0) lds_set(&prod[q], cur + 1);
                    continue;
                }
                u32x4 o[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    o[h] = buf[p][0][h];
                    if (!MODE) {
                        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                        for (int k = 0; k < NK; ++k) add8(acc, buf[p][k][h]);
                        o[h] = pack(acc);
                    }
                }
                lds_wait(&cons[q], cur - NSLOT + 1);  // slot free
#pragma unroll
                for (int h = 0; h < 2; ++h) ring[cur % NSLOT][q * QV + lane + h * 64] = o[h];
                __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0): quarter written before it is published
                if (lane == 0) lds_set(&prod[q], cur + 1);
            }
        }
    } else {  // push role: LDS quarter -> rank s's inbox, write-through; per unit, rs flags on its destinations
        for (int seq = 0; seq < rows; ++seq) {
            int u, s;
            if (dest(seq, u, s)) {
                lds_wait(&prod[q], seq + 1);
                u32x4 v[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) v[h] = ring[seq % NSLOT][q * QV + lane + h * 64];
                __builtin_amdgcn_s_waitcnt(0xc07f);
                if (lane == 0) lds_set(&cons[q], seq + 1);
                const int size = a.off[s + 1] - a.off[s];
                __amdgpu_buffer_rsrc_t d = rsrc(a.base[s] + inbox_off + ((uint64_t)rank * size + u) * H * 2);
#pragma unroll
                for (int h = 0; h < 2; ++h)
                    __builtin_amdgcn_raw_buffer_store_b128(v[h], d, (q * QV + lane + h * 64) * 16, 0, SYS);
            } else if (lane == 0) {
                lds_set(&cons[q], seq + 1);
            }
            if (seq % NPES == NPES - 1) {
                __builtin_amdgcn_s_waitcnt(0x0f70);  // vmcnt(0): this unit's rows acknowledged
                const int d = (rank + lane) % NPES, size = a.off[d + 1] - a.off[d];
                if (lane < NPES && u < size)
                    flag_put(a.base[d] + L.rs + (((uint64_t)rank * size + u) * 4 + q) * 4, epoch);
            }
        }
    }
    __syncthreads();
    if (tim && tx == 0) tim[blockIdx.x * 3 + 1] = __builtin_amdgcn_s_memrealtime();

    // ---- reduce this rank's rows -------------------------------------------------------------------------
    const int own = a.off[rank + 1] - a.off[rank];
    const __amdgpu_buffer_rsrc_t inbox = rsrc(a.base[rank] + inbox_off);
    const __amdgpu_buffer_rsrc_t dst = rsrc((uint64_t)out);
    for (int u = blockIdx.x; u < own; u += gridDim.x) {
        if (tx < NPES * 4)
            flag_wait(a.base[rank] + L.rs + (((uint64_t)(tx >> 2) * own + u) * 4 + (tx & 3)) * 4, epoch);
        __syncthreads();
        u32x4 w[NPES];
#pragma unroll
        for (int src = 0; src < NPES; ++src)
            w[src] = __builtin_amdgcn_raw_buffer_load_b128(inbox, (uint32_t)(((uint64_t)src * own + u) * H * 2) + tx * 16,
                                                           0, SYS);
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int src = 0; src < NPES; ++src) add8(acc, w[src]);
        __builtin_amdgcn_raw_buffer_store_b128(pack(acc), dst, (uint32_t)((uint64_t)u * H * 2) + tx * 16, 0, 0);
    }
    if (tim && tx == 0) tim[blockIdx.x * 3 + 2] = __builtin_amdgcn_s_memrealtime();
}

extern "C" {
uint64_t sym_bytes(int rcap) { return layout(rcap).total; }
// sizes: NPES shard row counts (max <= rcap); bases: every rank's mapped symmetric buffer
void launch(const void* y, const int* inv, void* out, const uint64_t* bases, const int* sizes, int rcap, int rank,
            int epoch, int mode, int grid, hipStream_t stream, uint64_t* tim) {
    Args a;
    int rmax = 0;
    a.off[0] = 0;
    for (int i = 0; i < NPES; ++i) {
        a.base[i] = bases[i];
        a.off[i + 1] = a.off[i] + sizes[i];
        rmax = sizes[i] > rmax ? sizes[i] : rmax;
    }
    if (mode == 0)
        sum_rs<0><<<grid, THREADS, 0, stream>>>((const uint16_t*)y, inv, (uint16_t*)out, a, rmax, rcap, rank, epoch,
                                               tim);
    else
        sum_rs<1><<<grid, THREADS, 0, stream>>>((const uint16_t*)y, inv, (uint16_t*)out, a, rmax, rcap, rank, epoch,
                                               tim);
}
}
