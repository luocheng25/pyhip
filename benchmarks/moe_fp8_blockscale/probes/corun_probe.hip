// Can BF16 atomics overlap with MFMA work on another workgroup of the same CU (MI300X gfx942)?
// Exactly 2 WGs per CU (24 KB dynamic LDS each); the first WG to arrive on a CU takes the MFMA role
// (optionally with a streaming HBM read), the second takes the atomic role (row-contiguous
// global_atomic_pk_add_bf16 over the real top-8 routes, as a 1-stage MoE down epilogue would).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define H 4096
typedef float floatx4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void pk_add_bf16(uint16_t* base, uint32_t off, uint32_t v) {
    asm volatile("global_atomic_pk_add_bf16 %0, %1, %2" ::"v"(off), "v"(v), "s"(base) : "memory");
}

__device__ __forceinline__ uint32_t cu_key() {
    uint32_t hw, xcc;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw));
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc));
    return ((xcc & 7) << 7) | ((hw >> 8) & 0x7F);  // XCC, SE, SH, CU
}

// ctr[0]: atomic row ticket; ctr[1 + key]: arrivals per CU.
// MFMA role: 8 independent 16x16x32 FP8 MFMAs per iteration; one 1 KB/wave read every `load_every`
// iterations with DEPTH reads in flight (a pipelined GEMM keeps several K-steps in flight).
template <int DEPTH>
__device__ void mfma_role(const uint4* buf, uint32_t mask, int iters, int load_every, float* sink,
                          unsigned long long* clk) {
    uint64_t c0 = __builtin_amdgcn_s_memtime(), r0 = __builtin_amdgcn_s_memrealtime();
    int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    long a = 0x0101010101010101L * (lane + 1), b = 0x0202020202020202L * (wave + 1);
    floatx4 acc[8];
    for (int j = 0; j < 8; ++j) acc[j] = (floatx4){0.f, 0.f, 0.f, 0.f};
    uint32_t x = 0;
    uint4 pend[DEPTH];
    size_t idx = ((size_t)(blockIdx.x * 4 + wave) * (iters / 8 + 1)) * 64 + lane;
    for (int d = 0; d < DEPTH; ++d) {
        pend[d] = load_every ? buf[idx & mask] : make_uint4(0, 0, 0, 0);
        idx += 64;
    }
    int step = load_every ? load_every : (iters + DEPTH - 1) / DEPTH;
    for (int it = 0; it < iters; it += DEPTH * step) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            for (int k = 0; k < step; ++k) {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[j] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc[j], 0, 0, 0);
            }
            if (load_every) {
                x ^= pend[d].x ^ pend[d].y ^ pend[d].z ^ pend[d].w;
                pend[d] = buf[idx & mask];
                idx += 64;
            }
        }
    }
    for (int d = 0; d < DEPTH; ++d) x ^= pend[d].x ^ pend[d].y ^ pend[d].z ^ pend[d].w;
    float s = (float)x;
    for (int j = 0; j < 8; ++j) s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    if (sink) sink[blockIdx.x * 256 + threadIdx.x] = s;
    if (clk && threadIdx.x == 0) {  // shader-clock and 100 MHz ticks of this WG: sustained SCLK
        uint64_t c1 = __builtin_amdgcn_s_memtime(), r1 = __builtin_amdgcn_s_memrealtime();
        atomicAdd(&clk[0], (unsigned long long)(c1 - c0));
        atomicAdd(&clk[1], (unsigned long long)(r1 - r0));
    }
}

__global__ void __launch_bounds__(256) corun(uint16_t* out, const int* sid, int rows, int T, int* ctr,
                                             const uint4* buf, uint32_t mask, int iters, int load_every,
                                             int mode, int period, int depth, int store, float* sink,
                                             unsigned long long* clk) {
    __shared__ int role_s;
    int lane = threadIdx.x & 63;
    if (threadIdx.x == 0) role_s = atomicAdd(&ctr[1 + cu_key()], 1);
    __syncthreads();
    int role = role_s;  // 0: MFMA, 1: atomics
    if (role == 0 && (mode & 1)) {
        if (depth == 4)
            mfma_role<4>(buf, mask, iters, load_every, sink, clk);
        else
            mfma_role<1>(buf, mask, iters, load_every, sink, clk);
    } else if (role == 1 && (mode & 2)) {
        // period > 0: each wave issues at most one real row per `period` ticks of the 100 MHz clock.
        uint64_t t0 = __builtin_amdgcn_s_memrealtime(), n = 0;
        for (;;) {
            int chunk = 0;
            if (lane == 0) chunk = atomicAdd(&ctr[0], 1);
            chunk = __builtin_amdgcn_readfirstlane(chunk);
            int r0 = chunk * 16;
            if (r0 >= rows) break;
            int r1 = r0 + 16 < rows ? r0 + 16 : rows;
            for (int r = r0; r < r1; ++r) {
                int token = sid[r] & 0xFFFFFF;
                if (token >= T) continue;
                if (period) {
                    while (__builtin_amdgcn_s_memrealtime() - t0 < n * period) __builtin_amdgcn_s_sleep(4);
                    ++n;
                }
                if (store) {  // the down kernel's Y write: sorted row r, 16 B per lane, full rows
                    uint4* y = (uint4*)out + (size_t)r * (H / 8) + lane;
                    uint4 v = make_uint4(0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u);
#pragma unroll
                    for (int c = 0; c < H / 512; ++c) y[c * 64] = v;
                    continue;
                }
                uint32_t base = (uint32_t)token * H * 2 + lane * 4;
#pragma unroll 8
                for (int c = 0; c < H / 128; ++c) pk_add_bf16(out, base + c * 256, 0x3F803F80u);
            }
        }
    }
}

extern "C" {
void launch(void* out, const int* sid, int rows, int T, int* ctr, const void* buf, uint32_t mask, int iters,
            int load_every, int mode, int period, int depth, int store, float* sink, unsigned long long* clk,
            hipStream_t s) {
    (void)hipMemsetAsync(ctr, 0, 4 * 1025, s);
    corun<<<304 * 2, 256, 24 * 1024, s>>>((uint16_t*)out, sid, rows, T, ctr, (const uint4*)buf, mask, iters,
                                           load_every, mode, period, depth, store, sink, clk);
}
}
