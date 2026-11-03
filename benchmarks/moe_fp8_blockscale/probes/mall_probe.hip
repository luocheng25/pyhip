// MALL residency probe: write a buffer (full 128-byte lines) then read it back, for several sizes.
#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void __launch_bounds__(256) write_buf(uint4* y, size_t n16) {
    uint4 v = make_uint4(0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u);
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n16; i += gridDim.x * 256ull) y[i] = v;
}

__global__ void __launch_bounds__(256) read_buf(const uint4* y, size_t n16, unsigned* sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n16; i += gridDim.x * 256ull) {
        uint4 v = y[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

extern "C" void launch(int kind, void* y, size_t bytes, void* sink, int grid, hipStream_t s) {
    if (kind == 0) write_buf<<<grid, 256, 0, s>>>((uint4*)y, bytes / 16);
    else read_buf<<<grid, 256, 0, s>>>((const uint4*)y, bytes / 16, (unsigned*)sink);
}
