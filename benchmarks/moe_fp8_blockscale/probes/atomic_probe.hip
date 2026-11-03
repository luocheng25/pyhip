// Atomic-accumulate throughput probes for the TP8 MoE down + TOPK reduce (MI300X gfx942).
// All kernels add 1.0 for each (route row, column) into OUT[token(row)] so totals are checkable.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define H 4096

__device__ __forceinline__ void pk_add_bf16(uint16_t* base, uint32_t byte_off, uint32_t v) {
    asm volatile("global_atomic_pk_add_bf16 %0, %1, %2" ::"v"(byte_off), "v"(v), "s"(base) : "memory");
}
__device__ __forceinline__ void add_f32(float* base, uint32_t byte_off, float v) {
    asm volatile("global_atomic_add_f32 %0, %1, %2" ::"v"(byte_off), "v"(v), "s"(base) : "memory");
}

// A: row-major, lane = 2 consecutive BF16 columns; one 256-byte row piece per instruction.
__global__ void __launch_bounds__(256) row_bf16(uint16_t* out, const int* sid, int rows, int T) {
    int wave = blockIdx.x * 4 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
    for (int r = wave; r < rows; r += gridDim.x * 4) {
        int token = sid[r] & 0xFFFFFF;
        if (token >= T) continue;
        uint32_t base = (uint32_t)token * H * 2 + lane * 4;
#pragma unroll 8
        for (int c = 0; c < H / 128; ++c) pk_add_bf16(out, base + c * 256, 0x3F803F80u);
    }
}

// B: MFMA 16x16 C layout: lane (r16, g) holds rows r16 and 4 columns 4g..4g+3 of each 16-column block.
//    Each wave handles 64 rows (4 row blocks) x one 1024-column group; 2 pk atomics per (block, row block).
__global__ void __launch_bounds__(256) mfma_bf16(uint16_t* out, const int* sid, int rows, int T, int tasks) {
    int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, r16 = lane & 15, g = lane >> 4;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        int mblk = task >> 2, cg = task & 3;
        uint32_t base[4];
        bool ok[4];
        for (int mb = 0; mb < 4; ++mb) {
            int row = mblk * 256 + wave * 64 + mb * 16 + r16;
            int token = row < rows ? (sid[row] & 0xFFFFFF) : T;
            ok[mb] = token < T;
            base[mb] = (uint32_t)(ok[mb] ? token : 0) * H * 2 + (cg * 1024 + 4 * g) * 2;
        }
        for (int n = 0; n < 64; ++n)
            for (int mb = 0; mb < 4; ++mb)
                if (ok[mb]) {
                    pk_add_bf16(out, base[mb] + n * 32, 0x3F803F80u);
                    pk_add_bf16(out, base[mb] + n * 32 + 4, 0x3F803F80u);
                }
    }
}

// C: same task shape as B, but after a lane-pair swap each lane owns 8 consecutive columns (16 bytes)
//    of one row: 4 pk atomics at consecutive dwords.
__global__ void __launch_bounds__(256) swap_bf16(uint16_t* out, const int* sid, int rows, int T, int tasks) {
    int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, r16 = lane & 15, g = lane >> 4;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        int mblk = task >> 2, cg = task & 3;
        uint32_t base[4];
        bool ok[4];
        for (int mb = 0; mb < 4; ++mb) {
            int row = mblk * 256 + wave * 64 + mb * 16 + r16;
            int token = row < rows ? (sid[row] & 0xFFFFFF) : T;
            ok[mb] = token < T;
            base[mb] = (uint32_t)(ok[mb] ? token : 0) * H * 2 + (cg * 1024 + 16 * (g & 1) + 4 * (g & 2)) * 2;
        }
        for (int i = 0; i < 32; ++i)
            for (int mb = 0; mb < 4; ++mb)
                if (ok[mb])
                    for (int k = 0; k < 4; ++k) pk_add_bf16(out, base[mb] + i * 64 + 4 * k, 0x3F803F80u);
    }
}

// D: row-major FP32 atomics (lane = 1 column; 256 bytes per instruction).
__global__ void __launch_bounds__(256) row_f32(float* out, const int* sid, int rows, int T) {
    int wave = blockIdx.x * 4 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
    for (int r = wave; r < rows; r += gridDim.x * 4) {
        int token = sid[r] & 0xFFFFFF;
        if (token >= T) continue;
        uint32_t base = (uint32_t)token * H * 4 + lane * 4;
#pragma unroll 8
        for (int c = 0; c < H / 64; ++c) add_f32(out, base + c * 256, 1.0f);
    }
}

// E: plain sorted-row stores (the current Y write) for reference: 16 bytes per lane, row-major.
__global__ void __launch_bounds__(256) row_store(uint4* y, int rows) {
    int wave = blockIdx.x * 4 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
    uint4 v = make_uint4(0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u);
    for (int r = wave; r < rows; r += gridDim.x * 4)
#pragma unroll
        for (int c = 0; c < H / 512; ++c) y[(size_t)r * (H / 8) + c * 64 + lane] = v;
}

// F/G: down task shape (wave = 64 rows x one 1024-column group), after an in-register transpose of the down
// epilogue: each instruction covers RPI rows x (64 / RPI) consecutive dwords, i.e. 256 / RPI contiguous bytes per
// row (F: RPI 4 -> 64 B, one 32-column unit; G: RPI 2 -> 128 B, an even/odd unit pair).
// XORDER (RPI 4): lane L holds row (L >> 2) & 3, dword 4 (L >> 4) + (L & 3) (the down epilogue's DPP quad
// exchange order) instead of row L >> 4, dword L & 15. CGMAJOR: all M-blocks of a column group before the next.
// GROUPS: column groups of 4096 / GROUPS columns (tasks = M-blocks x GROUPS).
template <int RPI, bool XORDER = false, bool CGMAJOR = false, int GROUPS = 4>
__global__ void __launch_bounds__(256) tile_bf16(uint16_t* out, const int* sid, int rows, int T, int tasks) {
    constexpr int LPR = 64 / RPI, COLS = LPR * 2;  // lanes per row, columns per instruction and row
    constexpr int CW = H / GROUPS;
    int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    int q = XORDER ? (lane >> 2) & 3 : lane / LPR, d = XORDER ? 4 * (lane >> 4) + (lane & 3) : lane % LPR;
    tasks = tasks / 4 * GROUPS;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        int nm = tasks / GROUPS;
        int mblk = CGMAJOR ? task % nm : task / GROUPS, cg = CGMAJOR ? task / nm : task % GROUPS;
        uint32_t base[64 / RPI];
        bool ok[64 / RPI];
#pragma unroll
        for (int i = 0; i < 64 / RPI; ++i) {
            int row = mblk * 256 + wave * 64 + i * RPI + q;
            int token = row < rows ? (sid[row] & 0xFFFFFF) : T;
            ok[i] = token < T;
            base[i] = (uint32_t)(ok[i] ? token : 0) * H * 2 + (cg * CW + 2 * d) * 2;
        }
        for (int n = 0; n < CW / COLS; ++n)
#pragma unroll
            for (int i = 0; i < 64 / RPI; ++i)
                if (ok[i]) pk_add_bf16(out, base[i] + n * COLS * 2, 0x3F803F80u);
    }
}

// H: tile64x_cgmaj8 plus the down kernel's reads per task: the M-block's A2 rows (64 KB, K = 256 FP8) and the
// expert's W2 slice of the group (512 columns x 256 = 128 KB), as plain dwordx4 loads.
__global__ void __launch_bounds__(256) tile_reads(uint16_t* out, const int* sid, const int* seid, const uint4* a2,
                                                  const uint4* w2, uint32_t* sink, int rows, int T, int tasks) {
    constexpr int GROUPS = 8, CW = H / GROUPS;
    int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, q = (lane >> 2) & 3, d = 4 * (lane >> 4) + (lane & 3);
    tasks = tasks / 4 * GROUPS;
    uint32_t acc = 0;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        int nm = tasks / GROUPS, mblk = task % nm, cg = task / nm;
        const uint4* ap = a2 + (size_t)mblk * 4096;                         // 256 rows x 256 B
        const uint4* wp = w2 + ((size_t)seid[mblk] * H + cg * CW) * 16;    // CW rows x 256 B
#pragma unroll
        for (int k = 0; k < 4; ++k) { uint4 v = ap[threadIdx.x + k * 256]; acc ^= v.x ^ v.w; }
#pragma unroll
        for (int k = 0; k < 32; ++k) { uint4 v = wp[threadIdx.x + k * 256]; acc ^= v.y ^ v.z; }
        uint32_t base[16];
        bool ok[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int row = mblk * 256 + wave * 64 + i * 4 + q;
            int token = row < rows ? (sid[row] & 0xFFFFFF) : T;
            ok[i] = token < T;
            base[i] = (uint32_t)(ok[i] ? token : 0) * H * 2 + (cg * CW + 2 * d) * 2;
        }
        for (int n = 0; n < CW / 32; ++n)
#pragma unroll
            for (int i = 0; i < 16; ++i)
                if (ok[i]) pk_add_bf16(out, base[i] + n * 64, 0x3F803F80u);
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

extern "C" {
void launch_reads(void* out, const int* sid, const int* seid, const void* a2, const void* w2, void* sink, int rows,
                  int T, int grid, hipStream_t s) {
    tile_reads<<<grid, 256, 0, s>>>((uint16_t*)out, sid, seid, (const uint4*)a2, (const uint4*)w2, (uint32_t*)sink,
                                    rows, T, (rows / 256) * 4);
}
void launch(int kind, void* out, const int* sid, int rows, int T, int grid, hipStream_t s) {
    int tasks = (rows / 256) * 4;
    if (kind == 0) row_bf16<<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T);
    if (kind == 1) mfma_bf16<<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
    if (kind == 2) swap_bf16<<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
    if (kind == 3) row_f32<<<grid, 256, 0, s>>>((float*)out, sid, rows, T);
    if (kind == 4) row_store<<<grid, 256, 0, s>>>((uint4*)out, rows);
    if (kind == 5) tile_bf16<4><<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
    if (kind == 6) tile_bf16<2><<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
    if (kind == 7) tile_bf16<4, true><<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
    if (kind == 8) tile_bf16<4, false, true><<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
    if (kind == 9) tile_bf16<4, true, true><<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
    if (kind == 10) tile_bf16<4, true, true, 8><<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
    if (kind == 11) tile_bf16<4, true, true, 16><<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
    if (kind == 12) tile_bf16<4, true, false, 8><<<grid, 256, 0, s>>>((uint16_t*)out, sid, rows, T, tasks);
}
}
