// Column-group variant of sum_ar.hip (MoE top-k sum + two-shot all-reduce over 8 GPUs, HIP IPC) for overlap
// with the down GEMM: one launch handles columns [cg * CW, (cg + 1) * CW) of every token row, i.e. the down
// kernel's column group cg, so it can run on its own CU-masked stream while down computes the later groups.
//
// Token rows are split into NPES shards (rank s owns rows [s*R, (s+1)*R)); unit u = row u of every shard.
// Persistent grid of 512-thread WGs; WG b owns units u = b + j * gridDim.x.
//   RS: wave pair p (sum wave p, push wave p + 4) takes rows seq = p, p + 4, ... of the WG's (unit, shard)
//      sequence. The sum wave adds the TOPK route-row pieces in fp32 (k order), rounds to bf16 and fills an
//      LDS ring slot; the push wave copies the slot to rank s's inbox[rank][u] with write-through sc0|sc1
//      stores and, one row later (s_waitcnt vmcnt(2)), sets rs[rank][u] = epoch on rank s. Sum waves issue no
//      global stores, so their loads never queue behind remote stores (vmcnt counts in issue order).
//   AG: every wave independently takes units: wait rs[*][u], sum the NPES inbox pieces in fp32 in rank order
//      (L2-bypassing loads), bf16, push to every rank's gather[rank * R + u], then ag[rank][u] = epoch there.
//   end: wait ag[*][u] for the launch's units. BAR > 0 (the last group): grid barrier (counter reaches BAR)
//      and a system acquire on every XCD so the consumer of gather reads no stale L2 line. Earlier groups skip
//      it: invalidating L2 would also drop the concurrently running down kernel's cached weights.
// Epochs grow by one per launch on every rank (monotonic flags). A launch ends only after every owner pushed
// its group, i.e. after every owner read its inbox, so the next launch (other columns, or the same columns of
// the next call) cannot overwrite unread data; the next call's launches start after the consumer (events).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define H 4096
#define CW 1024
#define NPES 8
#define TOPK 8
#define THREADS 512
#define NSLOT 4
#define SYS 17  // sc0|sc1
#define NT 2

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef const __attribute__((address_space(4))) int cint;  // scalar loads: lgkmcnt

struct Layout {  // byte offsets in every rank's symmetric buffer (sized for tmax tokens)
    uint64_t inbox, gather, rs, ag, done, err, dbg, total;
};

__host__ __device__ inline Layout layout(int tmax) {
    Layout L;
    uint64_t rmax = tmax / NPES;
    L.inbox = 0;                                  // [NPES src][R][H] bf16
    L.gather = L.inbox + (uint64_t)tmax * H * 2;  // [T][H] bf16: the all-reduced output
    L.rs = L.gather + (uint64_t)tmax * H * 2;     // [NPES src][R] i32
    L.ag = L.rs + NPES * rmax * 4;                // [NPES src][R] i32
    L.done = L.ag + NPES * rmax * 4;              // i32 grid-barrier counter
    L.err = L.done + 128;                         // i32 sticky error code of a wait that hit the 20 s guard
    L.dbg = L.done + 256;                         // [1024][2] u32: WG's CU key and start time (diagnostics)
    L.total = L.dbg + 8192;
    return L;
}

struct Peers {
    uint64_t base[NPES];
};

__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(uint64_t base) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, (short)0, 0x7FFFFFFF, 0x00020000);
}
__device__ __forceinline__ void flag_put(uint64_t addr, int epoch) {
    __hip_atomic_exchange((int*)addr, epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void flag_wait(uint64_t addr, int target, int* err, int code,
                                          int scope = __HIP_MEMORY_SCOPE_SYSTEM) {
    uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    while (__hip_atomic_load((int*)addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < target) {
        __builtin_amdgcn_s_sleep(1);
        if (__builtin_amdgcn_s_memrealtime() - t0 > 2000000000ull) {  // 20 s at 100 MHz: hang guard only
            __hip_atomic_store(err, code, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            return;
        }
    }
}
__device__ __forceinline__ void lds_wait(int* p, int v, int* err, int code) {
    uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    while (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) < v) {
        __builtin_amdgcn_s_sleep(0);
        if (__builtin_amdgcn_s_memrealtime() - t0 > 2000000000ull) {
            __hip_atomic_store(err, code, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            return;
        }
    }
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
template <int MODE>
__global__ void __launch_bounds__(THREADS) sum_ar_cols(const uint16_t* __restrict__ y, const int* __restrict__ inv,
                                                       Peers peers, int T, int tmax, int rank, int epoch, int cg,
                                                       int bar) {
    constexpr int NK = MODE ? 1 : TOPK;
    __shared__ u32x4 ring[4][NSLOT][CW / 8];
    __shared__ int prod[4], cons[4];
    const Layout L = layout(tmax);
    const int R = T / NPES, tx = threadIdx.x, wave = tx >> 6, lane = tx & 63, p = wave & 3;
    const uint64_t own = peers.base[rank];
    int* err = (int*)(own + L.err);  // 1 rs, 2 ag, 3 barrier, 4 ring space, 5 ring data: a wait hit the guard
    const uint32_t col = (cg * (CW / 8) + lane) * 16;  // this lane's first 16 B of the group; second at +1 KiB
    const int rows = ((R - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x) * NPES;
    if (tx < 4) prod[tx] = cons[tx] = 0;
    if (tx == 0 && blockIdx.x < 1024) {
        uint32_t hw, xcc;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw));
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc));
        uint32_t* dbg = (uint32_t*)(own + L.dbg) + 2 * blockIdx.x;
        dbg[0] = 0x80000000u | ((xcc & 7) << 8) | ((hw >> 8) & 0xFF);
        dbg[1] = (uint32_t)__builtin_amdgcn_s_memrealtime();
    }
    __syncthreads();

    // ---- RS ------------------------------------------------------------------------------------------------
    if (wave < 4) {
        cint* sinv = (cint*)inv;
        u32x4 buf[2][NK][2];
        auto fetch = [&](int seq, u32x4 (&dst)[NK][2]) {
            int u = blockIdx.x + (seq / NPES) * gridDim.x, t = ((rank + u + seq % NPES) % NPES) * R + u;
#pragma unroll
            for (int k = 0; k < NK; ++k) {
                long row = MODE ? t : sinv[t * TOPK + k];
                __amdgpu_buffer_rsrc_t r = rsrc((uint64_t)(y + row * H));
#pragma unroll
                for (int h = 0; h < 2; ++h) dst[k][h] = __builtin_amdgcn_raw_buffer_load_b128(r, col + h * 1024, 0, NT);
            }
        };
        int n = 0;
        if (p < rows) fetch(p, buf[0]);
        for (int seq = p; seq < rows; seq += 8, n += 2) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                int cur = seq + 4 * b;
                if (cur >= rows) break;
                if (cur + 4 < rows) fetch(cur + 4, buf[b ^ 1]);
                u32x4 out[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    out[h] = buf[b][0][h];
                    if (!MODE) {
                        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                        for (int k = 0; k < NK; ++k) add8(acc, buf[b][k][h]);
                        out[h] = pack(acc);
                    }
                }
                int slot = n + b;
                lds_wait(&cons[p], slot - NSLOT + 1, err, 4);
#pragma unroll
                for (int h = 0; h < 2; ++h) ring[p][slot % NSLOT][lane + h * 64] = out[h];
                __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0)
                if (lane == 0) __hip_atomic_store(&prod[p], slot + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
            }
        }
    } else {
        uint64_t pending = 0;  // rs flag of the previous row, set once its stores are acknowledged
        int slot = 0;
        for (int seq = p; seq < rows; seq += 4, ++slot) {
            int u = blockIdx.x + (seq / NPES) * gridDim.x, s = (rank + u + seq % NPES) % NPES;
            lds_wait(&prod[p], slot + 1, err, 5);
            u32x4 v[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) v[h] = ring[p][slot % NSLOT][lane + h * 64];
            __builtin_amdgcn_s_waitcnt(0xc07f);
            if (lane == 0) __hip_atomic_store(&cons[p], slot + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
            __amdgpu_buffer_rsrc_t d = rsrc(peers.base[s] + L.inbox + ((uint64_t)rank * R + u) * H * 2);
#pragma unroll
            for (int h = 0; h < 2; ++h) __builtin_amdgcn_raw_buffer_store_b128(v[h], d, col + h * 1024, 0, SYS);
            if (pending) {
                __builtin_amdgcn_s_waitcnt(0x0f72);  // vmcnt(2): the previous row's stores are acknowledged
                if (lane == 0) flag_put(pending, epoch);
            }
            pending = peers.base[s] + L.rs + ((uint64_t)rank * R + u) * 4;
        }
        if (pending) {
            __builtin_amdgcn_s_waitcnt(0x0f70);
            if (lane == 0) flag_put(pending, epoch);
        }
    }

    // ---- AG (waves independent) ----------------------------------------------------------------------------
    const __amdgpu_buffer_rsrc_t inbox = rsrc(own + L.inbox);
    const int gw = blockIdx.x * (THREADS / 64) + wave, nw = gridDim.x * (THREADS / 64);
    for (int u = gw; u < R; u += nw) {
        if (lane < NPES) flag_wait(own + L.rs + ((uint64_t)lane * R + u) * 4, epoch, err, 1);
        u32x4 w[NPES][2];
#pragma unroll
        for (int src = 0; src < NPES; ++src)
#pragma unroll
            for (int h = 0; h < 2; ++h)
                w[src][h] = __builtin_amdgcn_raw_buffer_load_b128(
                    inbox, (uint32_t)(((uint64_t)src * R + u) * H * 2) + col + h * 1024, 0, SYS);
        u32x4 red[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int src = 0; src < NPES; ++src) add8(acc, w[src][h]);
            red[h] = pack(acc);
        }
        uint64_t row = (uint64_t)rank * R + u;
#pragma unroll
        for (int i = 0; i < NPES; ++i) {
            __amdgpu_buffer_rsrc_t d = rsrc(peers.base[(rank + u + i) % NPES] + L.gather + row * H * 2);
#pragma unroll
            for (int h = 0; h < 2; ++h) __builtin_amdgcn_raw_buffer_store_b128(red[h], d, col + h * 1024, 0, SYS);
        }
        __builtin_amdgcn_s_waitcnt(0x0f70);
        if (lane < NPES) flag_put(peers.base[(rank + lane) % NPES] + L.ag + row * 4, epoch);
    }

    // ---- this group landed everywhere --------------------------------------------------------------------------
    for (int u = gw; u < R; u += nw)
        if (lane < NPES) flag_wait(own + L.ag + ((uint64_t)lane * R + u) * 4, epoch, err, 2);
    if (bar) {
        __syncthreads();
        if (tx == 0) {
            int* done = (int*)(own + L.done);
            __hip_atomic_fetch_add(done, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            flag_wait((uint64_t)done, bar, err, 3);
        }
        __syncthreads();
        if (tx < 64) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    }
}

extern "C" {
uint64_t sym_bytes(int tmax) { return layout(tmax).total; }
uint64_t gather_offset(int tmax) { return layout(tmax).gather; }
uint64_t err_offset(int tmax) { return layout(tmax).err; }
uint64_t dbg_offset(int tmax) { return layout(tmax).dbg; }
void launch(const void* y, const int* inv, const uint64_t* bases, int T, int tmax, int rank, int epoch, int mode,
            int cg, int bar, int grid, hipStream_t s) {
    Peers p;
    for (int i = 0; i < NPES; ++i) p.base[i] = bases[i];
    if (mode == 0)
        sum_ar_cols<0><<<grid, THREADS, 0, s>>>((const uint16_t*)y, inv, p, T, tmax, rank, epoch, cg, bar);
    else
        sum_ar_cols<1><<<grid, THREADS, 0, s>>>((const uint16_t*)y, inv, p, T, tmax, rank, epoch, cg, bar);
}
}
