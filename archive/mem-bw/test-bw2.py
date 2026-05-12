from functools import cache
import sys
import os
import random
from pyhip import jit, JIT, cudaPerf
import torch

@cache
def get_lane_id(J):
    vgpr_lane_id = J.gpr(J.threadIdx.x[0] & 63)
    return vgpr_lane_id

@cache
def get_lane_id_div(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) // divisor)

@cache
def get_lane_id_mod(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) % divisor)

def div_up(x, y):
    return (x + y - 1) // y

@jit(with_debug_log=False)
def kernel(J:JIT,
           p_A:"void*",
           p_Out:"void*",
           STAGES,
           TILE_NUM,
           THREADS,
           TILE_SIZE,
           WG_MEM_SIZE,
           INS_NUM):
    # grid: [N2 // BLOCK_N, sorted_expert_ids.shape[0]], [256]
    e_idx = J.blockIdx.x
    # hide following initialization into s_waitcnt
    offset_64bit = J.gpr(2, "su32")
    offset_64bit = J.s_mul_u32_u64(e_idx, WG_MEM_SIZE)
    p_A[:] += offset_64bit
    buff_a = J.Buffer(p_A, WG_MEM_SIZE)

    B_reg = J.gpr(STAGES, TILE_NUM, 4, "vf32", align=8) # 8-bf16 == DWORDx4
    S_reg = J.gpr(1, "vf32", align=8)
    S_reg[:] = 0
    soffset_kb = J.gpr("su32")
    soffset_kb[0] = 0
    for write_stage in range(STAGES - 1):
        for n in range(TILE_NUM):
            buff_a.load_dwordx4(B_reg[write_stage, n], J.threadIdx.x * 16, soffset_kb, non_temporal=True)
            soffset_kb[0] += THREADS * 16

    write_stage = STAGES - 1
    read_stage = 0

    def loop_body():
        nonlocal read_stage

        for n in range(TILE_NUM):
            for i in range(4):
                S_reg[0] += B_reg[read_stage, n, i]
                for _ in range(INS_NUM):
                    S_reg[0] += 0

        read_stage = (read_stage + 1) % STAGES

    s_i = J.gpr('si32')
    s_i[0] = 0
    full_loop_end = WG_MEM_SIZE // TILE_SIZE - (STAGES - 1)
    full_loop_cnt = (full_loop_end) // STAGES

    # rolled loop to avoid big code size
    load_addr = J.gpr(J.threadIdx.x * 16)
    with J.While(s_i[0] < full_loop_cnt) as loop:
        s_i[0] += 1
    # for _ in range(full_loop_cnt):
        for _ in range(STAGES):
            for n in range(TILE_NUM):
                buff_a.load_dwordx4(B_reg[write_stage, n], load_addr, soffset_kb, non_temporal=True)
                soffset_kb[0] += THREADS * 16
            write_stage = (write_stage + 1) % STAGES
            wait_cnt = TILE_NUM * (STAGES - 1)
            J.s_waitcnt(mod=f"vmcnt({wait_cnt})")

            loop_body()

    # tail due to STAGE
    for blk_n in range(full_loop_cnt * STAGES, full_loop_end):
        for n in range(TILE_NUM):
            buff_a.load_dwordx4(B_reg[write_stage, n], load_addr, soffset_kb, non_temporal=True)
            soffset_kb[0] += THREADS * 16
        write_stage = (write_stage + 1) % STAGES
        wait_cnt = TILE_NUM * (STAGES - 1)
        J.s_waitcnt(mod=f"vmcnt({wait_cnt})")

        loop_body()

    # tail for loop
    for blk_n in range(STAGES - 1):
        prefetch_cnt = STAGES - 1 - 1 - blk_n
        wait_cnt = TILE_NUM * prefetch_cnt
        J.s_waitcnt(mod=f"vmcnt({wait_cnt})")

        loop_body()
    addr = J.gpr(1, 'vu32')
    addr[0] = 0
    J.global_atomic_add_f32(addr, S_reg[0], p_Out)


# torch.cuda.set_device(5)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
num_CU = torch.cuda.get_device_properties().multi_processor_count
num_CU *= 1
print(f"{torch.get_default_device()=} with {num_CU=}")

STAGE = 3
TILE_NUM = 4 # in unit of 1024 bytes
THREADS = 256
# #THREADS * 16 bytes * #ELE
TILE_SIZE = THREADS * 16 * TILE_NUM
WG_MEM_SIZE = 256 * 64 * 1024 * 8
SIZE = WG_MEM_SIZE * num_CU
if SIZE >= 1024 * 1024 * 1024:
    buf_size_str = f'{SIZE / 1024 / 1024 / 1024:,} GB'
elif SIZE >= 1024 * 1024:
    buf_size_str = f'{SIZE / 1024 / 1024:,} MB'
else:
    buf_size_str = f'{SIZE / 1024:,} KB'
OCCUPANCY = 1
INS_NUM = 0
print(f'buffer size = {buf_size_str}')
A = torch.randn([SIZE//4], dtype=torch.float32)
def test(INS_NUM):
    B = torch.zeros([1], dtype=torch.float32)
    kernel([num_CU], [THREADS], A.data_ptr(), B.data_ptr(), STAGE, TILE_NUM, THREADS, TILE_SIZE, WG_MEM_SIZE, INS_NUM)
    if not torch.allclose(B, A.sum(dim=0), atol=0.01, rtol=0.01):
        ref = A.sum(dim=0).item()
        cur = B[0].item()
        print(f'{ref=} {cur=}')
        assert 0
    for _ in range(10):
        with cudaPerf(0, SIZE, name=f"bw") as p:
            kernel([num_CU], [THREADS],
                            A.data_ptr(), B.data_ptr(), STAGE, TILE_NUM, THREADS, TILE_SIZE, WG_MEM_SIZE, INS_NUM)

iter_idx = 1
iter_instr_load = TILE_NUM
iter_instr_load_addr = TILE_NUM
iter_wait = 1
iter_add = TILE_NUM * 4
mem_time = SIZE / 4e6
for num in [0, 2, 4, 5, 6, 7, 8, 9, 10, 16]:
    test(num)
    iter_manual = TILE_NUM * 4 * num
    total_instr = (iter_instr_load + iter_instr_load_addr + iter_wait + iter_add + iter_manual) * WG_MEM_SIZE // TILE_SIZE
    compute_time = (total_instr * 4) / 1.8e3
    print(f'ins_num = {num} {total_instr=:,} estimated {compute_time=:.1f} us, {mem_time=:.1f} us')
