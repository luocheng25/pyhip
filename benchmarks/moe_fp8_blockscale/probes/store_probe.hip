// Y-store pattern probe for the TP8 down (rows x 4096 BF16, tasks = (256-row M-block, 1024 columns)).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define N 4096

// kind 0: MFMA layout, lane (r16, g) -> row 16 mb + r16, 16 bytes at column 32 u + 8 g (64 B per row / instr).
// kind 1: same, column sweep rotated by M-block.
// kind 2: 4 rows x 256 B per instruction (lane l -> row 4 i + (l >> 4), column 128 c + 8 (l & 15)).
// kind 3: 2 rows x 512 B per instruction.
// kind 4: like 0, but two units back to back (a lane's two 16 B pieces complete each 128 B line together).
__global__ void __launch_bounds__(256) store_pattern(uint4* y, int tasks, int kind) {
    int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, r16 = lane & 15, g = lane >> 4;
    uint4 v = make_uint4(0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u);
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        int mblk = task >> 2, cg = task & 3;
        size_t row0 = (size_t)mblk * 256 + wave * 64;
        int col0 = cg * 1024;
        if (kind == 0 || kind == 1) {
            int rot = kind == 1 ? (mblk & 7) * 4 : 0;
            for (int uu = 0; uu < 32; ++uu) {
                int u = (uu + rot) & 31;
                for (int mb = 0; mb < 4; ++mb)
                    y[((row0 + 16 * mb + r16) * N + col0 + 32 * u + 8 * g) / 8] = v;
            }
        } else if (kind == 2) {
            for (int c = 0; c < 8; ++c)
                for (int i = 0; i < 16; ++i)
                    y[((row0 + 4 * i + (lane >> 4)) * N + col0 + 128 * c + 8 * (lane & 15)) / 8] = v;
        } else if (kind == 3) {
            for (int c = 0; c < 4; ++c)
                for (int i = 0; i < 32; ++i)
                    y[((row0 + 2 * i + (lane >> 5)) * N + col0 + 256 * c + 8 * (lane & 31)) / 8] = v;
        } else if (kind == 4) {
            for (int u = 0; u < 32; u += 2)
                for (int mb = 0; mb < 4; ++mb) {
                    y[((row0 + 16 * mb + r16) * N + col0 + 32 * u + 8 * g) / 8] = v;
                    y[((row0 + 16 * mb + r16) * N + col0 + 32 * (u + 1) + 8 * g) / 8] = v;
                }
        }
    }
}

extern "C" void launch(void* y, int tasks, int kind, int grid, hipStream_t s) {
    store_pattern<<<grid, 256, 0, s>>>((uint4*)y, tasks, kind);
}
