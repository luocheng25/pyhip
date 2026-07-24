import pyhip
import torch

from .common.loaders import get_mfma_loader, tb_swizzle

__all__ = [
    "gemm_8wave_fp8bf16fp16",
]

@pyhip.jit(with_debug_log = False)
def gemm_8wave_fp8bf16fp16(J,
                   AB_dtype, bpreshuffle,
                   use_f32_blockscales_128, # scale_BM,scale_BN,scale_BK = 1,128,128 
                   use_mfma_32x32,          # fp8: True=v_mfma_f32_32x32x64, False=v_mfma_f32_16x16x128
                   wg_M, wg_N, N, K, 
                   pA:"void*", # [M, K]  torch.float8_e4m3fn   row-major
                   pB:"void*", # [N, K]  torch.float8_e4m3fn   row-major
                   pC:"void*", # [M, N]  torch.bfloat16        row-major
                   pScaleA:"float*", #    [div_up(M,scale_BM), div_up(K, scale_BK)]
                                     # or [div_up(K, scale_BK), div_up(M,scale_BM)] if bpreshuffle
                   pScaleB:"float*", # [div_up(N,scale_BN), div_up(K, scale_BK) ]
                   M:"int"):
    """
    https://github.com/HazyResearch/HipKittens/blob/.../kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu
    """
    rotate_mfma_C = 0

    assert AB_dtype in ["fp8", "bf16", "fp16", "f16"]
    C_dtype = "bf16"
    # 是否使用 v_mfma_f32_32x32x64_f8f6f4（M=32,N=32,K=64）。由入参 use_mfma_32x32 控制(仅对 fp8 生效)；
    # bf16/fp16 恒为 16x16x128。此开关是编译期参数(进入 kernel cache key)，便于对比 32x32 与 16x16。
    mfma_32x32 = (AB_dtype == "fp8") and use_mfma_32x32
    assert not (mfma_32x32 and rotate_mfma_C), "32x32 路径不支持 rotate_mfma_C"
    M01 = 8
    GroupNum = 8

    # loader always load 128bytes (8 x DW4-lanes) along K dimension
    wg_K = J.div(128, J.sizeof(AB_dtype))

    stride_k = K * J.sizeof(AB_dtype)
    stride_C = N * J.sizeof(C_dtype)

    blk_m, blk_n = tb_swizzle(J, J.blockIdx.x, M, wg_M, wg_N, N, M01, GroupNum)
    pA[:] += blk_m * (wg_M * stride_k)
    pB[:] += blk_n * (wg_N * stride_k)
    pC[:] += blk_m * (wg_M * stride_C) # + blk_n * (wg_N * J.sizeof(C_dtype)))

    M0 = J.gpr("su32", blk_m * wg_M)
    M1 = J.gpr("su32")
    J.s_min_u32(M1, M0 + wg_M, M)
    Mc = J.gpr("su32", M1 - M0)

    assert N % wg_N == 0
    num_warps = 8
    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nbK = 2 # 2 MFMA 16x16 
    buff_a = J.Buffer(pA, Mc * stride_k)
    buff_b = J.Buffer(pB, wg_N * stride_k)
    buff_c = J.Buffer(pC, Mc * stride_C)

    WARPS_COL = 4
    WARPS_ROW = 2
    BLOCK_SIZE_ROW = wg_M
    BLOCK_SIZE_COL = wg_N
    BLOCK_K = 128
    HALF_BLOCK_SIZE_ROW = BLOCK_SIZE_ROW // 2
    HALF_BLOCK_SIZE_COL = BLOCK_SIZE_COL // 2

    lds_base = J.alloc_lds(HALF_BLOCK_SIZE_ROW * BLOCK_K * 4 + HALF_BLOCK_SIZE_COL * BLOCK_K * 4)
    ldsA = {}
    ldsB = {}
    lds = lds_base

    ldsA[0,0] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsA[0,1] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsA[1,0] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K
    ldsA[1,1] = lds; lds += HALF_BLOCK_SIZE_ROW * BLOCK_K

    ldsB[0,0] = lds; lds += HALF_BLOCK_SIZE_COL * BLOCK_K
    ldsB[0,1] = lds; lds += HALF_BLOCK_SIZE_COL * BLOCK_K
    ldsB[1,0] = lds; lds += HALF_BLOCK_SIZE_COL * BLOCK_K
    ldsB[1,1] = lds; lds += HALF_BLOCK_SIZE_COL * BLOCK_K

    nrM = J.div(nbM, WARPS_ROW, 2) # 4
    nrN = J.div(nbN, WARPS_COL, 2) # 2
    nrK = nbK

    if mfma_32x32:
        # 32x32 的 MFMA tile 在 M/N 方向各是 16x16 的两倍，故每 warp 需要的 tile 数减半
        nrM = nrM // 2   # 每 warp 每 half-block 沿 M 方向 2 个 32x32 tile
        nrN = nrN // 2   # 沿 N 方向 1 个 32x32 tile
        assert nrM >= 1 and nrN >= 1

    warp_m = J.gpr(J.warp_id[0] // WARPS_COL) # warp row: 0 to 1
    warp_n = J.gpr(J.warp_id[0] % WARPS_COL)  # warp col: 0 to 3

    use_pre_shuffle = False
    if rotate_mfma_C:
        vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = get_mfma_loader(J, use_pre_shuffle, num_warps, HALF_BLOCK_SIZE_ROW, BLOCK_K, stride_k, warp_n*32)
        vm_load_a, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = get_mfma_loader(J, bpreshuffle, num_warps, HALF_BLOCK_SIZE_COL, BLOCK_K, stride_k, warp_m*64)
        buff_a, buff_b = buff_b, buff_a
    else:
        vm_load_a, vm_load_cnt_a, vm_offset_inc_a, ds_read_a = get_mfma_loader(J, use_pre_shuffle, num_warps, HALF_BLOCK_SIZE_ROW, BLOCK_K, stride_k, warp_m*64,
                                                                              mfma_MN=(32 if mfma_32x32 else 16))
        vm_load_b, vm_load_cnt_b, vm_offset_inc_b, ds_read_b = get_mfma_loader(J, bpreshuffle, num_warps, HALF_BLOCK_SIZE_COL, BLOCK_K, stride_k, warp_n*32,
                                                                              mfma_MN=(32 if mfma_32x32 else 16))

    if use_f32_blockscales_128:
        # assert bpreshuffle == True, "exepct scaleA in [k,m] layout"
        scale_BM, scale_BN, scale_BK = 1,128,128 
        # tic-toc LDS buffer for 256 per-token per-k-128 scales
        # 1-warp is enough to load this buffer
        lds_scaleA = [J.alloc_lds(num_warps * 64 * J.sizeof_f32),
                      J.alloc_lds(num_warps * 64 * J.sizeof_f32)]
        # if pScaleA in [m,k] layout
        # pScaleA[:] += blk_m * (wg_M * J.div(K, scale_BK) * J.sizeof_f32)
        # buff_sa = J.Buffer(pScaleA, (M1 - M0) * J.div(K, scale_BK) * J.sizeof_f32)
        # scaleA : [div_up(K, scale_BK), div_up(M,scale_BM)]
        #   
        # pScaleA[:] += blk_m * (wg_M * J.sizeof_f32)
        buff_sa = J.Buffer(pScaleA, M * J.div(K, scale_BK) * J.sizeof_f32)
        voffset_scaleA = J.gpr(J.threadIdx.x[0] * J.sizeof_f32 + blk_m * (wg_M * J.sizeof_f32))
        assert wg_M <= num_warps * 64
        # vm_load_scaleA(lds_scaleA[toc])
        # ds_read scaleA must be in MFMA_16x4 format
        # ds_read scaleB broad-cast in to 16x4 too
        def vm_load_scaleA(lds, bk):
            # bk: index of k block with size of 128
            # use execmask to ensure same impact on vmcnt for all warps
            J.s_mov_b32("m0", lds + J.warp_id[0]*(64*J.sizeof_f32))
            voff = J.gpr("vu32", voffset_scaleA[0] + J.gpr("su32", M * (bk*J.sizeof_f32)))
            #with J.ExecMask(J.threadIdx.x[0] < wg_M, early_skip=False):
            buff_sa.load_dword(None, voff, 0)

        # scale of B(weights) are very small, can be all loaded into LDS
        lds_scaleB = J.alloc_lds(J.div(K, scale_BK) * J.div(wg_N, scale_BN) * J.sizeof_f32)
        pScaleB[:] += blk_n * (J.div(wg_N, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32)

        J.wg_load_lds(lds_scaleB, pScaleB, J.div(wg_N, scale_BN) * J.div(K, scale_BK) * J.sizeof_f32,
                      num_warps, wait_barrier = True)

        num_scaleB = J.div(wg_N, scale_BN)
        mfma_scaleA = J.gpr(nrM, "vf32")
        mfma_scaleB = J.gpr(num_scaleB, "vf32")
        # 每个 MFMA tile 沿 M 的行数：32x32 为 32，16x16 为 16。scaleA 为 per-token(scale_BM=1)，
        # 32x32 下每 lane 的 M = lane%32，故 scaleA 用 lane%32 索引。
        scale_mrow = 32 if mfma_32x32 else 16
        vaddr_scaleA = J.gpr("vu32", (J.lane_id[0] % scale_mrow)*J.sizeof_f32 + warp_m * (scale_mrow*nrM * J.sizeof_f32))
        def ds_read_scaleA(lds, m0):
            assert m0 in [0, 1]
            vaddr = J.gpr("vu32", vaddr_scaleA[0] + lds)
            for m in range(nrM):
                off = (m0*HALF_BLOCK_SIZE_ROW + m*scale_mrow)*J.sizeof_f32
                J.ds_read_b32(mfma_scaleA[m], vaddr, mod=f"offset:{off}")

        vaddr_scaleB = J.gpr(num_scaleB, "vu32")
        for i in range(num_scaleB):
            vaddr_scaleB[i] = lds_scaleB + i*J.div(K, scale_BK)*J.sizeof_f32
        def ds_read_scaleB(bk):
            # k0: in unit of scale_BK
            # n0: in unit of scale_BN
            # all warps share the same scaleB
            assert scale_BN >= nrN * 16 * 4
            if isinstance(bk, int):
                off = bk * J.sizeof_f32
                for i in range(num_scaleB):
                    J.ds_read_b32(mfma_scaleB[i], vaddr_scaleB[i], mod=f"offset:{off}")
            else:
                for i in range(num_scaleB):
                    J.ds_read_b32(mfma_scaleB[i], vaddr_scaleB[i] + bk * J.sizeof_f32)


    # v_mfma_f32_16x16x128_f8f6f4: 
    if mfma_32x32:
        # v_mfma_f32_32x32x64_f8f6f4: A/B 每次 8 VGPR(32 fp8)，C 16 VGPR(32x32 f32)
        #   mfma_A[m_tile, window, colgroup, 4]  window: BLOCK_K=128 内两个 K=64 窗口
        #   mfma_B[b_index, n_tile, window, colgroup, 4]
        #   mfma_C[cindex, m_tile, n_tile, 16]
        mfma_A = J.gpr(nrM, 2, 2, 4, "vfp8x4")
        mfma_B = J.gpr(2, nrN, 2, 2, 4, "vfp8x4")
        mfma_C = J.gpr(4, nrM, nrN, 16, "vf32")
    else:
        mfma_A = J.gpr(nrM, 2, 4, "vfp8x4")            # 4x[16,128]
        mfma_B = J.gpr(2, nrN, 2, 4, "vfp8x4")            # 2x[16,128]
        mfma_C = J.gpr(4, nrM, nrN, 4, "vf32")      # 4x[4,2]x[16,16]

    if use_f32_blockscales_128:
        MFMA_FIFO_CNT = nrM * nrN
        # circular fifo buffer for post-processing
        # prepare scales for next round
        mfma_fifo_scale = J.gpr(2, nrM, "vf32")
        C_PER_TILE = 16 if mfma_32x32 else 4
        mfma_fifo = J.gpr(MFMA_FIFO_CNT, C_PER_TILE, "vf32")
        mfma_fifo_scale[...] = 0
        mfma_fifo[...] = 0
        mfma_fifo_c_index = 0
        mfma_fifo_pending_slot = MFMA_FIFO_CNT - 1
        mfma_fifo_free_slot = 0
        mfma_fifo_pending_c_index = 0
        mfma_fifo_pending_tile = (nrM - 1, nrN - 1)

        if mfma_32x32:
            def mfma(c_index):
                # 仿照 16x16 的槽位级环形 FIFO，把所有 tile 视为跨 c_index 的连续流：
                #   w0(write free) -> 8xfmac(read pending) -> w1(write free) -> 8xfmac(read pending)
                # pending flush 完后，其槽成为下一个 free；当前 tile 成为新 pending。
                # 因此同一调用的 tile1 会 flush tile0，最后只留 tile1 给下一调用；两个原有槽即可，
                # 无需整拍 ping-pong、额外 FIFO 或 v_mov，且每对 MFMA 间都有 8 条 v_fmac。
                nonlocal mfma_fifo_scale, mfma_fifo, mfma_fifo_c_index
                nonlocal mfma_fifo_pending_slot, mfma_fifo_free_slot
                nonlocal mfma_fifo_pending_c_index, mfma_fifo_pending_tile
                b_index = c_index % 2
                for m_tile in range(nrM):
                    mfma_fifo_scale[c_index % 2, m_tile] = mfma_scaleA[m_tile] * mfma_scaleB[b_index]

                tiles = [(m_tile, n_tile) for m_tile in range(nrM) for n_tile in range(nrN)]

                def flush_pending(e0, e1):
                    m_tile, n_tile = mfma_fifo_pending_tile
                    sc = mfma_fifo_scale[mfma_fifo_pending_c_index % 2, m_tile]
                    for e in range(e0, e1):
                        J.v_fmac_f32(mfma_C[mfma_fifo_pending_c_index, m_tile, n_tile, e],
                                     mfma_fifo[mfma_fifo_pending_slot, e], sc)

                def mfma_w(t, slot, w):
                    # scale 路径用 srcA=mfma_B、srcB=mfma_A，使 M=lane%32(每 lane 固定)，
                    # 从而 scaleA(per-token) 与 scaleB(per-128N) 对该 tile 的 16 个 c_reg 相同。
                    m_tile, n_tile = t
                    cin = 0 if w == 0 else mfma_fifo[slot]
                    J.v_mfma_f32_32x32x64_f8f6f4(mfma_fifo[slot], mfma_B[b_index, n_tile, w], mfma_A[m_tile, w], cin)

                for t in tiles:
                    write_slot = mfma_fifo_free_slot
                    mfma_w(t, write_slot, 0)
                    flush_pending(0, 8)
                    mfma_w(t, write_slot, 1)
                    flush_pending(8, 16)

                    old_pending_slot = mfma_fifo_pending_slot
                    mfma_fifo_pending_slot = write_slot
                    mfma_fifo_free_slot = old_pending_slot
                    mfma_fifo_pending_c_index = c_index
                    mfma_fifo_pending_tile = t
                    yield 32
                mfma_fifo_c_index = c_index

            def mfma_tail():
                if mfma_fifo_pending_c_index is not None:
                    m_tile, n_tile = mfma_fifo_pending_tile
                    sc = mfma_fifo_scale[mfma_fifo_pending_c_index % 2, m_tile]
                    for e in range(16):
                        J.v_fmac_f32(mfma_C[mfma_fifo_pending_c_index, m_tile, n_tile, e],
                                     mfma_fifo[mfma_fifo_pending_slot, e], sc)
        else:
          def mfma(c_index):
            nonlocal mfma_fifo_scale, mfma_fifo, mfma_fifo_c_index
            b_index = c_index % 2

            fifo_read_id = 0
            fifo_write_id = 0
            for m in range(nrM):
                for n in range(nrN):
                    if n == 0:
                        mfma_fifo_scale[c_index%2, m] = mfma_scaleA[m] * mfma_scaleB[b_index]
                    J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 0], mfma_fifo[fifo_read_id, 0], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                    J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 1], mfma_fifo[fifo_read_id, 1], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                    J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 2], mfma_fifo[fifo_read_id, 2], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                    J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 3], mfma_fifo[fifo_read_id, 3], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                    fifo_read_id += 1

                    J.v_mfma_f32_16x16x128_f8f6f4(mfma_fifo[fifo_write_id % MFMA_FIFO_CNT], mfma_B[b_index, n], mfma_A[m], 0)
                    fifo_write_id += 1
                    yield 16
            mfma_fifo_c_index = c_index
        
          def mfma_tail():
            fifo_read_id = 0
            for m in range(nrM):
                for n in range(nrN):
                    if mfma_fifo_c_index is not None:
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 0], mfma_fifo[fifo_read_id, 0], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 1], mfma_fifo[fifo_read_id, 1], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 2], mfma_fifo[fifo_read_id, 2], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        J.v_fmac_f32(mfma_C[mfma_fifo_c_index, m, n, 3], mfma_fifo[fifo_read_id, 3], mfma_fifo_scale[mfma_fifo_c_index % 2,m])
                        fifo_read_id += 1

    elif AB_dtype == "fp8":
        if mfma_32x32:
            def mfma(c_index):
                # 每次 mfma() 处理一个 half-block(cindex) 中该 warp 的 nrM x nrN 个 32x32 tile，
                # 每个 tile 沿 K 方向有 2 个 window(每个 K=64)。
                # srcA=mfma_A(M) 决定输出行, srcB=mfma_B(N) 决定输出列(lane%32)，与 C 写回布局一致。
                # kk 放外层：先发所有 tile 的 w0(写各自 mfma_C，相互独立)，再发所有 tile 的 w1
                # (累加各自 w0)。这样每个 w0→w1 的 RAW 之间夹着其它 tile 的 MFMA，可掩盖
                # v_mfma_f32_32x32x64 的累加延迟(16x16x128 单条吃满 K=128 无此累加链)。
                b_index = c_index % 2
                for kk in range(2):
                    for m_tile in range(nrM):
                        for n_tile in range(nrN):
                            J.v_mfma_f32_32x32x64_f8f6f4(
                                mfma_C[c_index, m_tile, n_tile],
                                mfma_A[m_tile, kk],
                                mfma_B[b_index, n_tile, kk],
                                mfma_C[c_index, m_tile, n_tile])
                            yield 32
            def mfma_tail():
                pass
        else:
            def mfma(c_index):
                b_index = c_index % 2
                for m in range(nrM):
                    for n in range(nrN):
                        if rotate_mfma_C:
                            J.v_mfma_f32_16x16x128_f8f6f4(mfma_C[c_index, m, n], mfma_A[m], mfma_B[b_index, n], mfma_C[c_index, m, n])
                        else:
                            J.v_mfma_f32_16x16x128_f8f6f4(mfma_C[c_index, m, n], mfma_B[b_index, n], mfma_A[m], mfma_C[c_index, m, n])
                        yield 16
            def mfma_tail():
                pass
    elif AB_dtype == "bf16":
        def mfma(c_index):
            b_index = c_index % 2
            for k in range(2):
                for m in range(nrM):
                    for n in range(nrN):
                        if rotate_mfma_C:
                            J.v_mfma_f32_16x16x32_bf16(mfma_C[c_index, m, n], mfma_A[m, k], mfma_B[b_index, n, k], mfma_C[c_index, m, n])
                        else:
                            J.v_mfma_f32_16x16x32_bf16(mfma_C[c_index, m, n], mfma_B[b_index, n, k], mfma_A[m, k], mfma_C[c_index, m, n])
                        yield 16
        def mfma_tail():
            pass
    else:
        assert AB_dtype == "fp16" or AB_dtype == "f16" 
        def mfma(c_index):
            b_index = c_index % 2
            for k in range(2):
                for m in range(nrM):
                    for n in range(nrN):
                        if rotate_mfma_C:
                            J.v_mfma_f32_16x16x32_f16(mfma_C[c_index, m, n], mfma_A[m, k], mfma_B[b_index, n, k], mfma_C[c_index, m, n])
                        else:
                            J.v_mfma_f32_16x16x32_f16(mfma_C[c_index, m, n], mfma_B[b_index, n, k], mfma_A[m, k], mfma_C[c_index, m, n])
                        yield 16
        def mfma_tail():
            pass

    loop_cnt = J.div(K, wg_K)
    assert HALF_BLOCK_SIZE_ROW == HALF_BLOCK_SIZE_COL

    a_moffsets = J.gpr(2, "su32", 0, stride_k * HALF_BLOCK_SIZE_ROW)
    b_moffsets = J.gpr(2, "su32", 0, stride_k * HALF_BLOCK_SIZE_ROW)

    def step_k():
        a_moffsets[0] += vm_offset_inc_a
        a_moffsets[1] += vm_offset_inc_a
        b_moffsets[0] += vm_offset_inc_b
        b_moffsets[1] += vm_offset_inc_b

    bb_moffset = b_moffsets if bpreshuffle else a_moffsets

    def vm_loadA(k, m):
        assert m in [0, 1]
        assert k in [0, 1]
        return vm_load_a(ldsA[k,m], buff_a, a_moffsets[m])

    def vm_loadB(k, m):
        assert m in [0, 1]
        assert k in [0, 1]
        return vm_load_b(ldsB[k,m], buff_b, bb_moffset[m])

    def ds_readA(k, m):
        for i in range(nrM):
            ds_read_a(ldsA[k,m], mfma_A[i, 0], i, 0)
            ds_read_a(ldsA[k,m], mfma_A[i, 1], i, 1)

    def ds_readB(k, m):
        for i in range(nrN):
            ds_read_b(ldsB[k,m], mfma_B[m, i, 0], i, 0)
            ds_read_b(ldsB[k,m], mfma_B[m, i, 1], i, 1)

    if mfma_32x32:
        # 32x32x64 需要不同于 16x16 的 LDS 读取布局：
        #   MFMA A/B operand: lane(row=lane%32, klane=lane//32) 持有 32 个 fp8(8 VGPR)
        #   klane 选择 K 窗口(K=64)内的一半(32 bytes = 2 个 col_group)
        # LDS 仍是 loader 写入的 row-major swizzle 布局：
        #   LDS[abs_row*BLOCK_K + swizzle(abs_row&7, cg)*16] = X[abs_row, cg*16:+16]
        #   swizzle(row, cg) = (cg ^ (row&7)) & 7    (BLOCK_K=128 => 8 个 col_group)
        lds_stride_32 = BLOCK_K

        def make_ds_read_32(warp_row0):
            vrow   = J.gpr(J.lane_id[0] % 32)   # tile 内行(A) / 列(B)
            vklane = J.gpr(J.lane_id[0] // 32)  # K 窗口内的半区选择
            row_h  = J.gpr("vu32", vrow[0] >> 1)  # swizzle 用 (row>>1) 避免 CDNA4 16-lane 组内 bank conflict
            row_base = J.gpr("vu32", (vrow[0] + warp_row0) * lds_stride_32)
            voff  = {}
            voff2 = {}
            for kk in range(2):        # K 窗口(每个 K=64)
                for ci in range(2):    # klane 内的 2 个 col_group
                    cg  = J.gpr("vu32", kk * 4 + vklane[0] * 2 + ci)
                    swz = J.gpr("vu32", (cg[0] ^ row_h[0]) & 7)
                    v   = J.gpr("vu32", row_base[0] + swz[0] * 16)
                    voff[kk, ci]  = v
                    voff2[kk, ci] = J.gpr("vu32", v[0] + 64 * 1024)

            def ds_read_32(lds, dst, tile_idx, kk, ci):
                offset = lds + tile_idx * 32 * lds_stride_32
                if offset >= 64 * 1024:
                    J.ds_read_b128(dst, voff2[kk, ci], mod=f"offset:{offset - 64*1024}")
                else:
                    J.ds_read_b128(dst, voff[kk, ci], mod=f"offset:{offset}")
            return ds_read_32

        ds_read_a32 = make_ds_read_32(warp_m * 64)
        ds_read_b32 = make_ds_read_32(warp_n * 32)

        def ds_readA(k, m):
            for m_tile in range(nrM):
                for kk in range(2):
                    for ci in range(2):
                        ds_read_a32(ldsA[k, m], mfma_A[m_tile, kk, ci], m_tile, kk, ci)

        if bpreshuffle:
            # B 采用 pre_shuffle(mfma_MN=32) 布局，每个 1KB block=[klane_ps=2, row=32, 16B]，
            # block(nb,kb) 位于 lds + (nb*nbK_ps + kb)*1024；nb=warp_n(每 warp 一个 32-N-block)。
            # MFMA operand: lane(n=l%32, klane=l//32), window kk 用 block=nb*nbK_ps+(kk*2+klane)，
            #   两个 col_group 分别取 klane_ps=0(偏移0) 与 klane_ps=1(偏移512)。
            nbK_ps = BLOCK_K // 32   # tile 内 K-block 数(每 block K=32)
            vn_b     = J.gpr(J.lane_id[0] % 32)
            vklane_b = J.gpr(J.lane_id[0] // 32)
            vbase_b  = J.gpr("vu32", warp_n * (nbK_ps * 1024) + vklane_b[0] * 1024 + vn_b[0] * 16)
            vbase_b2 = J.gpr("vu32", vbase_b[0] + 64 * 1024)

            def ds_read_bps32(lds, dst, kk, ci):
                offset = lds + kk * 2 * 1024 + ci * 512
                if offset >= 64 * 1024:
                    J.ds_read_b128(dst, vbase_b2, mod=f"offset:{offset - 64*1024}")
                else:
                    J.ds_read_b128(dst, vbase_b, mod=f"offset:{offset}")

            def ds_readB(k, m):
                for n_tile in range(nrN):
                    for kk in range(2):
                        for ci in range(2):
                            ds_read_bps32(ldsB[k, m], mfma_B[m, n_tile, kk, ci], kk, ci)
        else:
            def ds_readB(k, m):
                for n_tile in range(nrN):
                    for kk in range(2):
                        for ci in range(2):
                            ds_read_b32(ldsB[k, m], mfma_B[m, n_tile, kk, ci], n_tile, kk, ci)

    #print(nrM, nrN); assert 0
    if 1:
        # 8-wave pipeline invented by HipKittens
        tic = 0
        toc = 1
        if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[tic], 0)
        J.emit(vm_loadB(tic,0))
        J.emit(vm_loadA(tic,0))
        J.emit(vm_loadB(tic,1))
        J.emit(vm_loadA(tic,1))

        with J.If(warp_m[0] == 1):
            J.s_barrier()

        mfma_C[...] = 0

        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b})"); J.s_barrier()

        step_k()

        if use_f32_blockscales_128:
            vm_load_scaleA(lds_scaleA[toc], 1)
            vm_load_cnt_scaleA = 1
        else:
            vm_load_cnt_scaleA = 0
        J.emit(vm_loadA(toc,0))
        J.emit(vm_loadB(toc,0))
        J.emit(vm_loadB(toc,1))

        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b*2 + vm_load_cnt_scaleA})"); J.s_barrier()

        def loop_body(k, loop_cnt):
            nonlocal tic, toc
            ds_readB(tic, 0)    # lgkmcnt += nrN*2 (2*2)
            ds_readA(tic, 0)    # lgkmcnt += nrM*2 (4*2)

            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[tic], 0)
                ds_read_scaleB(k)

            J.emit(vm_loadA(toc,1))
            step_k()
            J.s_waitcnt(mod=f"lgkmcnt(0)"); J.s_barrier()

            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_setprio(1)
            J.emit(mfma(0))
            J.s_setprio(0); J.s_barrier()
            #===============================================================
            # after this s_barrier, lgkmcnt(8) ensures all 8-waves has finished
            # accessing B[tic,0], so next vm_load can overwrite A[toc,0],B[toc,0],B[toc,1],A[toc,1]

            ds_readB(tic, 1)
            J.emit(vm_loadA(tic,0))                         # vm_load_cnt_a
            J.s_barrier()

            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_setprio(1)
            J.emit(mfma(1))
            J.s_setprio(0); J.s_barrier()

            ds_readA(tic, 1)
            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[tic], 1)
            J.emit(vm_loadB(tic,0))                         # vm_load_cnt_b
            J.s_barrier()

            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_setprio(1)
            J.emit(mfma(2))
            J.s_setprio(0); J.s_barrier()

            J.emit(vm_loadB(tic,1))                         # vm_load_cnt_b
            if use_f32_blockscales_128:
                vm_load_scaleA(lds_scaleA[tic], k+2)
            J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_a + vm_load_cnt_b*2 + vm_load_cnt_scaleA})"); J.s_barrier()

            J.s_setprio(1)
            J.emit(mfma(3))
            J.s_setprio(0); J.s_barrier()
            #===============================================================
            # after this s_barrier, we have all A[toc] & B[toc] loaded in LDS
            # so in next iteration, we can ds_read A[tic] & B[tic] w/o waitting for any vmcnt

            tic ^= 1
            toc ^= 1

        if 1:
            for k in range(loop_cnt):
                loop_body(k, loop_cnt)
        else:
            assert not use_f32_blockscales_128, "there is an unknown accuracy issue for f32 blockscale-128 case"
            assert (loop_cnt % 2) == 0
            k = J.gpr("su32", 0)

            with J.While(k[0] < loop_cnt):
                loop_body(k, loop_cnt)
                k[0] += 1
                loop_body(k, loop_cnt)
                k[0] += 1

        mfma_tail()
        J.s_waitcnt(mod="vmcnt(0)")

        with J.If(warp_m[0] == 0):
            J.s_barrier()

    else:
        # 第一步确保基础设施正确，使用最低效简单的pipeline，8-wave一起读入LDS，一起读出到寄存器，计算
        # naive pipeline, for debugging basic building blocks
        mfma_C[...] = 0

        J.debug_setup((J.warp_id[0] == 0) & (J.blockIdx.x[0] == 0))
        for k in range(loop_cnt):
            J.emit(vm_loadB(0,0))
            J.emit(vm_loadA(0,0))
            if use_f32_blockscales_128: vm_load_scaleA(lds_scaleA[0], k)
            J.s_waitcnt(mod="vmcnt(0)"); J.s_barrier()

            ds_readA(0,0)
            ds_readB(0,0)
            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[0], 0)
                ds_read_scaleB(k)
            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_barrier()
            J.emit(mfma(0))

            #J.debug_log(mfma_A[0,0], torch.float8_e4m3fn, "4h.16v.16h")
            #J.debug_log(mfma_A[0,1], torch.float8_e4m3fn, "4h.16v.16h")
            #J.s_endpgm()

            J.emit(vm_loadB(0,1))
            J.s_waitcnt(mod="vmcnt(0)"); J.s_barrier()

            ds_readB(0,1)
            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_barrier()
            J.emit(mfma(1))

            #J.debug_log(mfma_B[1,0,0], torch.float8_e4m3fn, "4h.16v.16h")
            #J.debug_log(mfma_B[1,0,1], torch.float8_e4m3fn, "4h.16v.16h")
            #J.s_endpgm()

            J.emit(vm_loadA(0,1))
            J.s_waitcnt(mod="vmcnt(0)"); J.s_barrier()

            ds_readA(0,1)
            if use_f32_blockscales_128:
                ds_read_scaleA(lds_scaleA[0], 1)
            J.s_waitcnt(mod="lgkmcnt(0)"); J.s_barrier()

            #J.debug_log(mfma_A[0,0], torch.float8_e4m3fn, "4h.16v.16h")
            #J.debug_log(mfma_A[0,1], torch.float8_e4m3fn, "4h.16v.16h")
            #J.s_endpgm()

            J.emit(mfma(2))
            J.emit(mfma(3))

            step_k()
        mfma_tail()
    #J.debug_log(mfma_C[1,0,0], torch.float, "4h.16v.4h")
    #J.s_endpgm()

    stride_c = N * J.sizeof_bf16

    if mfma_32x32:
        sizeof_bf16 = J.sizeof_bf16
        if use_f32_blockscales_128:
            # scale 路径: srcA=mfma_B => 输出 M=lane%32(每 lane 固定), N=(lane//32)*4 + i*8 + j
            #   c_reg[i*4+j] = out[M=lane%32, N=(lane//32)*4+i*8+j]
            # lane 与 lane+32 分别持有同一行相邻的 4 列。参考 16x16 写回，用 permlane swap
            # 将两个 half-wave 的结果交织为连续 8 列，再用一次 store_dwordx4 写 16B。
            # 这里必须用 v_permlane32_swap_b32；v_permlane16 配对 lane+16，会混合不同行。
            vm_lane  = J.gpr(J.lane_id[0] % 32)    # tile 内行(M)
            vn_klane = J.gpr(J.lane_id[0] // 32)
            vaddr_base = J.gpr("vu32",
                (warp_m * 64 + vm_lane[0]) * stride_c
                + (blk_n * wg_N + warp_n * 32 + vn_klane[0] * 8) * sizeof_bf16)
            vbf16 = J.gpr(4, "vbf16x2")
            for cindex in range(4):
                cm = cindex // 2
                cn = cindex % 2
                vaddr = J.gpr("vu32", vaddr_base[0] + cm * HALF_BLOCK_SIZE_ROW * stride_c)
                for m_tile in range(nrM):
                    for i in range(0, 4, 2):
                        J.uni_cvt_pk_bf16_f32(vbf16[0],
                            mfma_C[cindex, m_tile, 0, i * 4],
                            mfma_C[cindex, m_tile, 0, i * 4 + 1])
                        J.uni_cvt_pk_bf16_f32(vbf16[1],
                            mfma_C[cindex, m_tile, 0, i * 4 + 2],
                            mfma_C[cindex, m_tile, 0, i * 4 + 3])
                        J.uni_cvt_pk_bf16_f32(vbf16[2],
                            mfma_C[cindex, m_tile, 0, (i + 1) * 4],
                            mfma_C[cindex, m_tile, 0, (i + 1) * 4 + 1])
                        J.uni_cvt_pk_bf16_f32(vbf16[3],
                            mfma_C[cindex, m_tile, 0, (i + 1) * 4 + 2],
                            mfma_C[cindex, m_tile, 0, (i + 1) * 4 + 3])

                        # lower half-wave receives i's q=0/q=1 data; upper half-wave receives
                        # (i+1)'s q=0/q=1 data. vbf16[0:4] then represent 8 contiguous bf16.
                        J.v_permlane32_swap_b32(vbf16[0], vbf16[2])
                        J.v_permlane32_swap_b32(vbf16[1], vbf16[3])
                        buff_c.store_dwordx4(vbf16, vaddr, 0,
                            offset12=cn * HALF_BLOCK_SIZE_COL * sizeof_bf16 + i * 8 * sizeof_bf16)
                    vaddr[0] += 32 * stride_c
        else:
            # 非 scale 路径: srcA=mfma_A => 输出 col=lane%32=N(沿 N coalesced), 16 个 f32 分布为
            #   c_reg[i*4+j] = out[M=(lane//32)*4+i*8+j, N=lane%32]
            vrow   = J.gpr(J.lane_id[0] % 32)   # tile 内列(N)
            vklane = J.gpr(J.lane_id[0] // 32)
            vaddr_base = J.gpr("vu32",
                (vklane[0] * 4 + warp_m * 64) * stride_c
                + (blk_n * wg_N + warp_n * 32 + vrow[0]) * sizeof_bf16)
            vbf = J.gpr("vbf16x2")
            for cindex in range(4):
                cm = cindex // 2
                cn = cindex % 2
                ncol = cn * HALF_BLOCK_SIZE_COL
                for m_tile in range(nrM):
                    m_row_base = cm * HALF_BLOCK_SIZE_ROW + m_tile * 32
                    for i in range(4):
                        for j in range(0, 4, 2):
                            J.uni_cvt_pk_bf16_f32(vbf,
                                mfma_C[cindex, m_tile, 0, i * 4 + j],
                                mfma_C[cindex, m_tile, 0, i * 4 + j + 1])
                            row_lo = m_row_base + i * 8 + j
                            soff_lo = J.gpr("su32", row_lo * stride_c + ncol * sizeof_bf16)
                            soff_hi = J.gpr("su32", (row_lo + 1) * stride_c + ncol * sizeof_bf16)
                            J.buffer_store_short(vbf, vaddr_base, buff_c.desc, soff_lo, mod="offen")
                            J.buffer_store_short_d16_hi(vbf, vaddr_base, buff_c.desc, soff_hi, mod="offen")
    elif not rotate_mfma_C:
        vbf16 = J.gpr(4, "vbf16x2")
        col = J.lane_id // 16
        swap_12_col = (col & 1) * 2 + (col >> 1)

        vaddr0 = J.gpr(((J.lane_id % 16) + warp_m * 64)*stride_c + swap_12_col * J.sizeof_DW4 + warp_n * 32 * J.sizeof_bf16 + \
                blk_n * (wg_N * J.sizeof(C_dtype)))

        for cindex in range(4):
            cm = cindex // 2
            cn = cindex % 2
            vaddr = J.gpr("vu32", vaddr0[0] + cm*HALF_BLOCK_SIZE_ROW*stride_c)
            for m in range(nrM):
                for n in range(0, nrN, 2):
                    J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[cindex, m,n,0], mfma_C[cindex, m,n,1]) 
                    J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[cindex, m,n,2], mfma_C[cindex, m,n,3])
                    J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[cindex, m,n+1,0], mfma_C[cindex, m,n+1,1])
                    J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[cindex, m,n+1,2], mfma_C[cindex, m,n+1,3])
                    #    a0    a1   a2   a3   | 01 23
                    #    b0    b1   b2   b3   | 45 67
                    #  v_permlane16_swap_b32(a, b)
                    #    a0    b0   a2   b2   |
                    #    a1    b1   a3   b3   |
                    #
                    # swap of row 1 & 2 are done by swapping lane-address 
                    J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
                    J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
                    buff_c.store_dwordx4(vbf16, vaddr, 0, offset12 = n*4*J.sizeof_DW2 + cn*HALF_BLOCK_SIZE_COL*J.sizeof_bf16)
                vaddr[0] += 16*stride_c
    else: # rotate_mfma_C
        #
        # [2, 4] warp  [4, 2i]x[16, 16] mfma_C  `i means inner-most dimension`
        #     when rotate_mfma_C=1
        # [4, 2] warp  [2i, 4]x[16, 16] mfma_C
        #
        # cindex is also rotated, cn determines A's row
        #
        vbf16 = J.gpr(4, "vbf16x2")
        col = J.lane_id // 16
        swap_12_col = (col & 1) * 2 + (col >> 1)
        vaddr0 = J.gpr(((J.lane_id % 16) + warp_n * 32)*stride_c + swap_12_col * J.sizeof_DW4 + warp_m * 64 * J.sizeof_bf16 + \
                        blk_n * (wg_N * J.sizeof(C_dtype)))
        for cindex in range(4):
            cm = cindex // 2
            cn = cindex % 2
            vaddr = J.gpr("vu32", vaddr0[0] + cn*HALF_BLOCK_SIZE_ROW*stride_c)
            for n in range(nrN):
                for m in range(0, nrM, 2):
                    J.uni_cvt_pk_bf16_f32(vbf16[0], mfma_C[cindex, m,n,0], mfma_C[cindex, m,n,1]) 
                    J.uni_cvt_pk_bf16_f32(vbf16[1], mfma_C[cindex, m,n,2], mfma_C[cindex, m,n,3])
                    J.uni_cvt_pk_bf16_f32(vbf16[2], mfma_C[cindex, m+1,n,0], mfma_C[cindex, m+1,n,1])
                    J.uni_cvt_pk_bf16_f32(vbf16[3], mfma_C[cindex, m+1,n,2], mfma_C[cindex, m+1,n,3])
                    J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
                    J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
                    buff_c.store_dwordx4(vbf16, vaddr, 0, offset12 = m*16*J.sizeof_bf16 + cm*HALF_BLOCK_SIZE_COL*J.sizeof_bf16)
                vaddr[0] += 16*stride_c