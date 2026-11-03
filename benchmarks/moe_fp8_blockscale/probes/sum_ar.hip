// MoE top-k sum fused with a two-shot all-reduce over 8 GPUs (MI300X, HIP IPC symmetric buffers).
//
// Every rank holds Y [rows, H] bf16 (its I-slice route rows, route weight folded in) and INV [T, TOPK]
// (sorted row of route (t, k)). out[t] = sum over ranks of bf16(sum_k Y[INV[t, k]]), bit-identical on
// every rank. Token rows are split into NPES shards (rank s owns rows [s*R, (s+1)*R)); unit u = row u of
// every shard. Persistent grid of 512-thread WGs (every WG resident), units u = blockIdx.x + j * gridDim.x.
//   RS (waves 0-3 "sum", waves 4-7 "push", paired by quarter of the row): sum wave w loads the TOPK rows of
//      token row (unit u, destination shard s rotated by u), adds them in fp32, rounds to bf16 and writes
//      its quarter into an LDS ring slot; push wave w+4 copies the quarter to rank s's inbox[rank][u] with
//      write-through sc0|sc1 stores. Sum waves issue no global stores, so their Y loads never wait behind
//      remote stores in the in-order vmcnt queue. After a unit, push wave w sets rs[rank][u][w] = epoch on
//      the 8 destination ranks (monotonic flags, never reset). mode 1: copy a ready partial row instead.
//   AG (all waves, after RS): wait rs[*][u][*], sum the NPES inbox rows in fp32 in rank order (L2-bypassing
//      loads), bf16, push to every rank's gather[rank * R + u], then ag[rank][u] = epoch on every rank.
//   end: wait ag[*][u] for this WG's units; grid barrier (monotonic counter) so that every XCD invalidates
//      its L2 (system acquire) only after the whole gather region has landed.
// RS never waits on another rank and AG only on RS: no deadlock with all WGs resident. A peer pushes epoch
// e+1 only after its epoch-e grid saw our ag flags (we had read our inbox) and after our e+1 launch (stream
// order: our consumer had read gather e), so single-buffered inbox/gather are safe.
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

struct Layout {  // byte offsets in every rank's symmetric buffer (sized for tmax tokens)
    uint64_t inbox, gather, rs, ag, done, total;
};

__host__ __device__ inline Layout layout(int tmax) {
    Layout L;
    uint64_t rmax = tmax / NPES;
    L.inbox = 0;                                  // [NPES src][R][H] bf16
    L.gather = L.inbox + (uint64_t)tmax * H * 2;  // [T][H] bf16: the all-reduced output
    L.rs = L.gather + (uint64_t)tmax * H * 2;     // [NPES src][R][4 quarters] i32
    L.ag = L.rs + NPES * rmax * 16;               // [NPES src][R] i32
    L.done = L.ag + NPES * rmax * 4;              // i32 grid-barrier counter (+= grid per launch)
    L.total = L.done + 256;
    return L;
}

struct Peers {
    uint64_t base[NPES];  // every rank's symmetric buffer, mapped in this process
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

// mode 0: Y + INV (fused top-k sum); mode 1: Y is a ready partial [T, H] (all-reduce only)
// diag (timing experiments, wrong results): bit0 RS pushes to this rank only, bit1 no Y loads, bit2 no AG;
// tim[blockIdx][4]: s_memrealtime at start, RS end, AG end, exit
template <int MODE>
__global__ void __launch_bounds__(THREADS) sum_ar(const uint16_t* __restrict__ y, const int* __restrict__ inv,
                                                  Peers peers, int T, int tmax, int rank, int epoch, int diag,
                                                  uint64_t* tim) {
    constexpr int NK = MODE ? 1 : TOPK;
    __shared__ u32x4 ring[NSLOT][H / 8];
    __shared__ int prod[4], cons[4];
    const Layout L = layout(tmax);
    const int R = T / NPES, tx = threadIdx.x, wave = tx >> 6, lane = tx & 63, q = wave & 3;
    const uint64_t own = peers.base[rank];
    const int units = (R - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x, rows = units * NPES;
    if (tx < 4) prod[tx] = cons[tx] = 0;
    if (tim && tx == 0) tim[blockIdx.x * 4] = __builtin_amdgcn_s_memrealtime();
    __syncthreads();

    // ---- RS ----------------------------------------------------------------------------------------------
    if (wave < 4) {  // sum role: Y loads (prefetched one row ahead) -> fp32 sum -> bf16 -> LDS quarter
        cint* sinv = (cint*)inv;
        u32x4 buf[2][NK][2];
        auto fetch = [&](int seq, u32x4 (&dst)[NK][2]) {
            int u = blockIdx.x + (seq / NPES) * gridDim.x, s = (rank + u + seq % NPES) % NPES, t = s * R + u;
#pragma unroll
            for (int k = 0; k < NK; ++k) {
                if (diag & 2) {
                    dst[k][0] = dst[k][1] = (u32x4){(unsigned)t, 0u, 0u, 0u};
                    continue;
                }
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
                int cur = seq + p;
                if (cur >= rows) break;
                if (cur + 1 < rows) fetch(cur + 1, buf[p ^ 1]);
                u32x4 out[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    out[h] = buf[p][0][h];
                    if (!MODE) {
                        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                        for (int k = 0; k < NK; ++k) add8(acc, buf[p][k][h]);
                        out[h] = pack(acc);
                    }
                }
                lds_wait(&cons[q], cur - NSLOT + 1);  // slot free
#pragma unroll
                for (int h = 0; h < 2; ++h) ring[cur % NSLOT][q * QV + lane + h * 64] = out[h];
                __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0): quarter written before it is published
                if (lane == 0) __hip_atomic_store(&prod[q], cur + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
            }
        }
    } else {  // push role: LDS quarter -> rank s's inbox, write-through; per unit, rs flags on 8 ranks
        for (int seq = 0; seq < rows; ++seq) {
            int u = blockIdx.x + (seq / NPES) * gridDim.x, s = (rank + u + seq % NPES) % NPES;
            if (diag & 1) s = rank;
            lds_wait(&prod[q], seq + 1);
            u32x4 v[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) v[h] = ring[seq % NSLOT][q * QV + lane + h * 64];
            __builtin_amdgcn_s_waitcnt(0xc07f);
            if (lane == 0) __hip_atomic_store(&cons[q], seq + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
            __amdgpu_buffer_rsrc_t d = rsrc(peers.base[s] + L.inbox + ((uint64_t)rank * R + u) * H * 2);
#pragma unroll
            for (int h = 0; h < 2; ++h)
                __builtin_amdgcn_raw_buffer_store_b128(v[h], d, (q * QV + lane + h * 64) * 16, 0, SYS);
            if (seq % NPES == NPES - 1) {
                __builtin_amdgcn_s_waitcnt(0x0f70);  // vmcnt(0): this unit's 8 rows acknowledged
                if (lane < NPES)
                    flag_put(peers.base[(rank + lane) % NPES] + L.rs + (((uint64_t)rank * R + u) * 4 + q) * 4, epoch);
            }
        }
    }
    __syncthreads();
    if (tim && tx == 0) tim[blockIdx.x * 4 + 1] = __builtin_amdgcn_s_memrealtime();

    // ---- AG ----------------------------------------------------------------------------------------------
    const __amdgpu_buffer_rsrc_t inbox = rsrc(own + L.inbox);
    for (int j = 0; j < units && !(diag & 4); ++j) {
        int u = blockIdx.x + j * gridDim.x;
        if (tx < NPES * 4) flag_wait(own + L.rs + (((uint64_t)(tx >> 2) * R + u) * 4 + (tx & 3)) * 4, epoch);
        __syncthreads();
        u32x4 w[NPES];
#pragma unroll
        for (int src = 0; src < NPES; ++src)
            w[src] = __builtin_amdgcn_raw_buffer_load_b128(inbox, (uint32_t)(((uint64_t)src * R + u) * H * 2) + tx * 16,
                                                           0, SYS);
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int src = 0; src < NPES; ++src) add8(acc, w[src]);
        u32x4 red = pack(acc);
        uint64_t row = (uint64_t)rank * R + u;
#pragma unroll
        for (int i = 0; i < NPES; ++i)
            __builtin_amdgcn_raw_buffer_store_b128(red, rsrc(peers.base[(rank + u + i) % NPES] + L.gather + row * H * 2),
                                                   tx * 16, 0, SYS);
        __builtin_amdgcn_s_waitcnt(0x0f70);
        __syncthreads();
        if (tx < NPES) flag_put(peers.base[(rank + tx) % NPES] + L.ag + row * 4, epoch);
    }
    // ---- every owner's rows landed here --------------------------------------------------------------------
    if (tim && tx == 0) tim[blockIdx.x * 4 + 2] = __builtin_amdgcn_s_memrealtime();
    for (int j = 0; j < units && !(diag & 4); ++j)
        if (tx < NPES) flag_wait(own + L.ag + ((uint64_t)tx * R + blockIdx.x + j * gridDim.x) * 4, epoch);
    __syncthreads();
    if (tx == 0) {
        int* done = (int*)(own + L.done);
        __hip_atomic_fetch_add(done, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        while (__hip_atomic_load(done, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < epoch * (int)gridDim.x)
            __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    if (tx < 64) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    if (tim && tx == 0) tim[blockIdx.x * 4 + 3] = __builtin_amdgcn_s_memrealtime();
}

extern "C" {
uint64_t sym_bytes(int tmax) { return layout(tmax).total; }
uint64_t gather_offset(int tmax) { return layout(tmax).gather; }
void launch(const void* y, const int* inv, const uint64_t* bases, int T, int tmax, int rank, int epoch, int mode,
            int grid, hipStream_t s, int diag, uint64_t* tim) {
    Peers p;
    for (int i = 0; i < NPES; ++i) p.base[i] = bases[i];
    if (mode == 0)
        sum_ar<0><<<grid, THREADS, 0, s>>>((const uint16_t*)y, inv, p, T, tmax, rank, epoch, diag, tim);
    else
        sum_ar<1><<<grid, THREADS, 0, s>>>((const uint16_t*)y, inv, p, T, tmax, rank, epoch, diag, tim);
}
}
