// Does a VMEM atomic issued with EXEC = 0 still count in vmcnt (gfx942)?
// A load from cold memory, then 20 atomics with EXEC = 0, then s_waitcnt vmcnt(20): if the atomics count,
// the load must have returned; if not, the copied register still holds the sentinel.
#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void probe(const uint32_t* src, uint32_t* dst, uint32_t* res, uint32_t stride, int zero_exec) {
    uint32_t off = (blockIdx.x * stride) * 4;
    uint32_t v = 0xDEAD, out;
    if (zero_exec)
        asm volatile(
            "v_mov_b32 %0, 0xdead\n\t"
            "global_load_dword %0, %2, %3\n\t"
            "s_mov_b64 s[40:41], exec\n\t"
            "s_mov_b64 exec, 0\n\t"
            ".rept 20\n\t"
            "global_atomic_add %2, %4, %5\n\t"
            ".endr\n\t"
            "s_mov_b64 exec, s[40:41]\n\t"
            "s_waitcnt vmcnt(20)\n\t"
            "v_mov_b32 %1, %0\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            : "+v"(v), "=v"(out)
            : "v"(off), "s"(src), "v"(1u), "s"(dst)
            : "s40", "s41", "memory");
    else
        asm volatile(
            "v_mov_b32 %0, 0xdead\n\t"
            "global_load_dword %0, %2, %3\n\t"
            "s_waitcnt vmcnt(20)\n\t"
            "v_mov_b32 %1, %0\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            : "+v"(v), "=v"(out)
            : "v"(off), "s"(src)
            : "memory");
    if (threadIdx.x == 0) res[blockIdx.x] = out;
}

extern "C" void launch(const uint32_t* src, uint32_t* dst, uint32_t* res, int blocks, uint32_t stride,
                       int zero_exec, hipStream_t s) {
    probe<<<blocks, 64, 0, s>>>(src, dst, res, stride, zero_exec);
}
