import functools
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl._mlir.dialects import llvm, vector
from flydsl.expr.typing import Vector as Vec

import pyhip.contrib.flydsl.helpers as fxh

fxh.dump_ir(True)

def _maxnumf(a, b):
    """Non-NaN-propagating f32 max used by the wave softmax reduction."""
    return type(a)(arith.maxnumf(arith.unwrap(a), arith.unwrap(b)))

@flyc.jit
def online_softmax(fragS, fragO, sm_scale_log2, old_max, l_in,
                   q_pos0, kv_block_n, kv_len, qo_len,
                   is_all_kv_valid: fx.Constexpr[bool],
                   KV_BLOCK_SIZE: fx.Constexpr[int],
                   is_causal: fx.Constexpr[bool],
                   return_bf16_probability: fx.Constexpr[bool] = False,
                   window_left: fx.Constexpr[int] = -1,
                   probability_mfma_k: fx.Constexpr[int] = 8):
    """
    old_max/l_in是会被更新的，使用SSA方式return更新后值，不要使用mutable container例如list来修改

    is_causal为True时， kv_len >= qo_len, 并且attention只需要计算causal_mask合法区域即可：

                rows = torch.arange(qo_len, device="cuda").unsqueeze(1)
                cols = torch.arange(kv_len, device="cuda").unsqueeze(0)
                causal_mask = cols <= (kv_len - qo_len + rows)
     - num_kv_pages 只需循环到某个位置即可，后面的page都不用参考
     - 某个kv-page之前都是non-causal的，之后才需要施加causal_mask
     - causal_mask 施加于 32x32 的 score 矩阵上，    
    """
    # assert 0, f"{fragS}"
    if fx.const_expr(not is_all_kv_valid):
        # mask out invalid kv positions
        lane_id = fx.thread_idx.x & 63
        col_lane = (lane_id < 32).select(fx.Int32(0), fx.Int32(16))
        bf16_col_lane = (lane_id < 32).select(fx.Int32(0), fx.Int32(8))
        col_block = fx.Int32(kv_block_n * KV_BLOCK_SIZE)
        if fx.const_expr(not is_causal):
            # Keep both sides explicitly i32.  A Python constexpr loop index
            # otherwise promotes this comparison to MLIR index, whose ordered
            # comparison is unsigned; a negative limit would then look huge
            # and leave invalid tail columns unmasked.
            for i in fx.range_constexpr(16):
                if fx.const_expr(return_bf16_probability):
                    column = bf16_col_lane + fx.Int32((i // 8) * 16 + i % 8)
                else:
                    column = col_lane + fx.Int32(i)
                kv_pos = col_block + column
                if kv_pos >= kv_len:
                    fragS[i,0,0] = float("-inf")
        else:
            # Bottom-right causal mask:
            #   kv_pos <= kv_len - qo_len + q_pos
            wave_id = fx.thread_idx.x // 64
            row_lane = fx.thread_idx.x & 31
            q_pos = q_pos0 + wave_id * 32 + row_lane
            causal_limit = kv_len - qo_len + q_pos
            for i in fx.range_constexpr(16):
                if fx.const_expr(return_bf16_probability):
                    column = bf16_col_lane + fx.Int32((i // 8) * 16 + i % 8)
                else:
                    column = col_lane + fx.Int32(i)
                kv_pos = col_block + column
                outside_window = kv_pos > causal_limit
                if fx.const_expr(window_left >= 0):
                    outside_window = outside_window | (
                        kv_pos < causal_limit - fx.Int32(window_left)
                    )
                if outside_window:
                    fragS[i,0,0] = float("-inf")


    scores = fxh.eltwise_op("v_mul_f32", fragS.load(), sm_scale_log2)

    row_max = scores.reduce("max")
    row_max = _maxnumf(row_max, row_max.shuffle_xor(32, 64))

    new_max = old_max
    corr = fx.Float32(1.0)
    threshold = fxh.eltwise_op("v_add_f32", old_max, fx.Float32(7.0))
    if row_max > threshold:
        new_max = fxh.eltwise_op("v_add_f32", row_max, fx.Float32(1.0))
        # do not use inline asm inside scf.If, use intrinsic instead
        corr = fxh.eltwise_op("llvm.amdgcn.exp2.f32", old_max - new_max)

    centered_scores = fxh.eltwise_op("v_sub_f32", scores, new_max)
    probs = fxh.eltwise_op("v_exp_f32", centered_scores)
    row_sum = probs.reduce("add")

    # this fake instruction avoids spills for some reason, but seems to be not required anymore
    # row_sum = fxh.eltwise_op("; fake inst", row_sum, 0.0)
    l_out = fxh.eltwise_op("v_fma_f32", l_in, corr, row_sum)
    fragS.store(probs)

    # Rebase the accumulated numerator only when the lazy max advances.
    def rescale_output():
        fragO.store(fxh.eltwise_op("v_mul_f32", fragO.load(), corr))

    @flyc.jit
    def rescale_if_needed():
        if corr < fx.Float32(1.0):
            rescale_output()

    rescale_if_needed()
    if fx.const_expr(return_bf16_probability):
        probability = fxh.cvt_f32_to_bf16(fragS)
        if fx.const_expr(probability_mfma_k == 16):
            probability_layout = fx.make_layout((8, 1, 2), (1, 0, 8))
        else:
            probability_layout = fx.make_layout(
                (4, 1, (2, 2)), (1, 0, (4, 8))
            )
        probability = fx.make_view(
            fx.get_iter(probability), probability_layout
        )
        return new_max, l_out, probability
    return new_max, l_out


@functools.cache
def PagedAttention(
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
    page_size,
    is_causal,
    quant_query_mode="per-token",
    key_layout="vectorized",
    window_left=-1,
    has_sink=False,
):
    """
    cu_seqlens_q: [batch_size + 1] cu_seqlens_q[i] ~ cu_seqlens_q[i+1] is the range of query tokens in batch i
    kv_indptr   : [batch_size + 1] kv_indptr[i] ~ kv_indptr[i+1] is the range of virtual page ids in batch i
    kv_page_indices : [num_pages] kv_page_indices[i] is the physical page id of virtual page i (used to index into K and V)

    k_vector_size is number of elements that 16 bytes can hold

    persistent kernel, each 8wave workgroup occupies one CU to handles part(BM) of the query/output tokens. and loop
    over cu_seqlens_q to find next part of query tokens to handle， until all query tokens are handled.

    任务复杂，从极简pipeline开始构建，保证框架正确之后再开始性能调优迭代
    """
    BM, BN = 256, 32
    num_threads = 512
    num_waves = num_threads // 64
    is_gfx950 = "gfx950" in torch.cuda.get_device_properties().gcnArchName
    assert page_size in [32, 64, 128]
    assert key_layout in ("vectorized", "linear")
    assert window_left == -1 or window_left >= 0
    assert num_qo_heads % num_kv_heads == 0
    if key_layout == "linear":
        assert page_size == 32
        assert head_dim_qk == head_dim_v == 128
    if window_left >= 0:
        assert is_causal
        assert key_layout == "vectorized"
        assert has_sink
        assert page_size == 64
        assert head_dim_qk in (128, 192)
        assert head_dim_v in (128, 192)
    else:
        assert not has_sink
    num_BN_per_page = page_size // BN
    LOG2E = 1.4426950408889634
    sm_scale_log2 = float(LOG2E / (head_dim_qk**0.5))

    assert (page_size % BN) == 0, f"{page_size=} must be a multiple of {BN=}"

    assert quant_query_mode in ["per-token", "per-tensor"], f"quant_query_mode={quant_query_mode} is not supported"

    @flyc.jit
    def attn_pipeline(q_tile, # [BM, head_dim_qk]
                      k_tile, # [BN, (k_vector_size, head_dim_qk // k_vector_size), num_physical_pages, num_BN_per_page]
                      v_tile, # [head_dim_v, (k_vector_size, BN // k_vector_size), num_physical_pages, num_BN_per_page]
                      o_tile, # [BM, head_dim_v]
                      q_pos0, query_len, kv_len, full_qo_len,
                      ptr_kv_page_table,
                      first_page, num_kv_pages, last_page_len,
                      qk_scale_log2, v_s, sink_logit,
                      shared_allocator):
        tid = fx.thread_idx.x
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4

        flyobj = fxh.FlyObjCache()
        dtype = k_tile.dtype
        pv_mfma_k = 16
        if fx.const_expr(dtype == fx.BFloat16 and not is_gfx950):
            pv_mfma_k = 8
        # gfx950 can consume 64 FP8 reduction elements per QK instruction.
        # PV still reduces over BN=32, so it retains the K16 atom.
        use_fp8_qk_k64 = (
            is_gfx950
            and dtype == fx.Float8E4M3FN
            and head_dim_qk % 64 == 0
        )
        qk_mfma_k = 64 if use_fp8_qk_k64 else pv_mfma_k
        if fx.const_expr(use_fp8_qk_k64):
            qk_mma_atom = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(32, 32, 64, dtype)
            )
            qk_mma_atom = fx.atom_set_value(
                qk_mma_atom, "scale_a", fx.Int32(0)
            )
            qk_mma_atom = fx.atom_set_value(
                qk_mma_atom, "scale_b", fx.Int32(0)
            )
        else:
            qk_mma_atom = fx.make_mma_atom(
                fx.rocdl.MFMA(32, 32, qk_mfma_k, dtype)
            )
        pv_mma_atom = fx.make_mma_atom(
            fx.rocdl.MFMA(32, 32, pv_mfma_k, dtype)
        )
        atom_values = pv_mfma_k // 2
        vector_values = 128 // dtype.width
        packed_atoms = vector_values // atom_values
        wave_layout = fx.make_layout((1, 8, 1), (1, 1, 0))
        if fx.const_expr(use_fp8_qk_k64):
            qk_tiled_mma = fx.make_tiled_mma(
                qk_mma_atom,
                wave_layout,
                (None, None, fx.make_layout((32, 2), (1, 32))),
            )
        elif fx.const_expr(packed_atoms == 1):
            qk_tiled_mma = fx.make_tiled_mma(qk_mma_atom, wave_layout)
        else:
            qk_permutation = fx.make_layout(
                (atom_values, 2, packed_atoms),
                (1, vector_values, atom_values),
            )
            qk_tiled_mma = fx.make_tiled_mma(
                qk_mma_atom,
                wave_layout,
                (None, None, qk_permutation),
            )
        if fx.const_expr(packed_atoms == 1):
            pv_tiled_mma = fx.make_tiled_mma(
                pv_mma_atom,
                wave_layout,
            )
        else:
            pv_permutation = fx.make_layout(
                (atom_values, 2, packed_atoms),
                (1, vector_values, atom_values),
            )
            pv_tiled_mma = fx.make_tiled_mma(
                pv_mma_atom,
                wave_layout,
                (None, None, pv_permutation),
            )
        tmma1 = qk_tiled_mma.thr_slice(tid)
        tmma2 = pv_tiled_mma.thr_slice(tid)

        """
        [TRICKY]:
        P@V的gemm中V的reduction维度的layout：
            lane[0] 0..15
            lane[32] 16..31
        而P的reduction维度的layout，跟K的n维度的关系满足MFMA的输出layout:
            lane[0] 0..3/8..11/16..29/24..27
            lane[32] 4..7/12..15/20..23/28..31
        
        为了避免额外的reorder开销：
            对K的layout中n维度进行remap, 用这个layout进行compose (4, 2, 4):(1, 16, 4) 使得P的lane[0]/[32]跟V一致
            对P的寄存器排布，按照 fx.gemm 对 A/B 输入的要求重新解释
        """
        if fx.const_expr(k_tile.dtype == fx.BFloat16):
            k_row_layout = fx.make_layout((4, 2, 2, 2), (1, 8, 4, 16))
        else:
            k_row_layout = fx.make_layout((4, 2, 4), (1, 16, 4))
        k_tile = fx.composition(
            k_tile,
            fx.make_tile(k_row_layout, None, None, None),
        )

        fragQ = flyobj.load_tiled_mma_fragB(tmma1, q_tile)

        k_fake = fx.Tensor(
            fx.make_view(
                fx.get_iter(k_tile),
                fx.make_layout((BN, head_dim_qk), (head_dim_qk, 1)),
            )
        )
        v_fake = fx.Tensor(
            fx.make_view(
                fx.get_iter(v_tile),
                fx.make_layout((head_dim_v, BN), (BN, 1)),
            )
        )
        fragK = tmma1.make_fragment_A(k_fake)
        fragV = tmma2.make_fragment_A(v_fake)
        num_bits_fragK = (fx.size(fragK.shape).get_static_leaf_int * fragK.dtype.width)
        num_bits_fragV = (fx.size(fragV.shape).get_static_leaf_int * fragV.dtype.width)
        num_vm_cnt_load_v = (num_bits_fragV)//128

        fakeCt = fx.make_rmem_tensor(fx.make_layout((BN, BM), (BM, 1)), fx.Float32)
        fragS = tmma1.make_fragment_C(fakeCt)
        fragO = tmma2.make_fragment_C(fx.select(o_tile, [1, 0]))
        """
        [TRICKY]:
        
        """
        prob_operand = fx.make_rmem_tensor(
            fx.make_layout((8, 1, 2), (1, 0, 8)),
            v_tile.dtype,
        )

        # gfx950 BF16 register-pressure change (Dqk=192 production case):
        #
        # Before:
        #   prefetch[ping] = global_load(K[i + 3])  # 8 VGPR/thread
        #   prefetch[pong] = global_load(K[i + 4])  # 8 VGPR/thread
        #   ... QK -> softmax -> PV -> loop backedge ...
        #   LDS[ping_or_pong] = prefetch[ping_or_pong]
        #
        # Both fragments stayed live across several stages and the loop
        # backedge. Together with Q/K/V/O fragments this exceeded 256 VGPRs,
        # so LLVM repeatedly spilled K data in the steady loop.
        #
        # After:
        #   tmp = global_load(K[i + 3])             # one transient fragment
        #   ... overlap the load with QK/softmax ...
        #   wait(tmp); LDS[(i + 3) % 3] = tmp
        #   fragK = LDS[(i + 1) % 3]
        #
        # The three-stage LDS ring carries the lookahead across iterations.
        # Only one 8-VGPR temporary lives from global load to LDS store, and it
        # dies before the backedge. This trades 12 KiB LDS for zero scratch in
        # the steady KV loop.
        use_k_ring = is_gfx950 and k_tile.dtype == fx.BFloat16
        k_lds_stages = 3 if use_k_ring else 2

        # let all 512 threads participate in the copy so no extra if condition involved
        # 512*16/32 = 256, so all head_dim <= 256 can be padded to 256
        copy_atom_bits = (
            128 if k_tile.dtype == fx.BFloat16
            else 64 if head_dim_qk == 128
            else 128
        )

        @fx.union
        class SharedStorage:
            k_lds: fx.Array[
                k_tile.dtype, k_lds_stages * BN * head_dim_qk, 16
            ]
            o_lds: fx.Array[o_tile.dtype, (BM//8) * head_dim_v, 16]

        # mask,base,shift, swizzle always in unit of 128b,
        swz_base = ((128 // k_tile.dtype.width) - 1).bit_length()
        swz = fx.SwizzleType.get(3, swz_base, 3)
        lds = shared_allocator.allocate(SharedStorage)
        layout_k_lds = fx.make_composed_layout(
            fx.static(swz),
            fx.make_ordered_layout(
                (BN, head_dim_qk, k_lds_stages), (1, 0, 2)
            ),
        )
        lds_k = lds.k_lds.peek().view(layout_k_lds)

        # assert 0, f"{lds_ku32} {lds_k}"

        def is_valid_block_n(bn):
            #return fx.const_expr(bn >= 0 and bn < num_kv_pages) if fx.const_expr(isinstance(bn, int)) else True
            return fx.const_expr(bn >= 0) if fx.const_expr(isinstance(bn, int)) else True

        # k_tile layout: Tensor<f8E4M3FNUZ, global, ((4,2,4),(16,8),?):((16,256,64),(1,512),4096)>
        # assert 0, f"{k_tile}"
        num_copy_threads = BN * head_dim_qk * k_tile.dtype.width // copy_atom_bits
        assert BN * head_dim_qk * k_tile.dtype.width % copy_atom_bits == 0
        if fx.const_expr(k_tile.dtype == fx.BFloat16 and head_dim_qk == 192):
            # Fit 768 b128 atoms into 384 threads; each thread copies two atoms.
            num_copy_threads //= 2
        assert num_copy_threads <= num_threads

        # [TRICKY]
        # Keep the global->register->LDS pipeline packed in 32-bit dwords.  If
        # the FP8 fragment crosses a loop backedge as vector<16xi8>, LLVM
        # scalarizes it into byte values and later emits shifts/v_perm to pack
        # it again for ds_write_b128.  Recasting the already-partitioned,
        # contiguous per-thread slice preserves its byte address (including
        # the LDS swizzle) while making the loop-carried value vector<Nxi32>.
        def recast_tensor(src, new_dtype):
            result_type = fx.PointerType.get(new_dtype.ir_type, src.memspace, new_dtype.width//8)
            new_iter = fx.recast_iter(result_type, fx.get_iter(src))
            new_layout = fx.recast_layout(src.layout, src.dtype.width, new_dtype.width)
            return fx.make_view(new_iter, new_layout)

        lds_k_u32 = recast_tensor(lds_k, fx.Uint32)
        k_tile_u32 = recast_tensor(k_tile, fx.Uint32)

        glk_thrcopy, glk_load_atom = flyobj.get_tiled_copy_coalesced_mn(
            k_tile_u32[None, None, 0, 0],
            copy_atom_bits=copy_atom_bits,
            num_threads=num_copy_threads,
        )
        glk_srck = glk_thrcopy.partition_S(k_tile_u32)
        glk_dstk = glk_thrcopy.partition_D(lds_k_u32)

        glk_store_atom = flyobj.get_universal_copy_atom(
            fx.Uint32, copy_atom_bits
        )
        glk_frag = fx.make_fragment_like(glk_dstk[None, None, None, 0])
        num_vm_cnt_load_k = (fx.size(glk_frag.shape).get_static_leaf_int * glk_frag.dtype.width)//copy_atom_bits
        if fx.const_expr(use_k_ring):
            prefetch_fragk = fx.make_fragment_like(
                glk_srck[None, None, None, 0, 0]
            )
        else:
            prefetch_fragk_list = [
                fx.make_fragment_like(glk_srck[None, None, None, 0, 0]),
                fx.make_fragment_like(glk_srck[None, None, None, 0, 0]),
            ]

        def global_load_k(block_n, page_id, bn_id, frag_id):
            if fx.const_expr(is_valid_block_n(block_n)):
                if fx.const_expr(key_layout == "linear"):
                    last_block = num_kv_pages * num_BN_per_page - 1
                    source_block = fx.Int32(arith.minsi(
                        arith.unwrap(fx.Int32(block_n)), arith.unwrap(last_block)
                    ))
                    source = glk_srck[None, None, None, source_block, 0]
                else:
                    source = glk_srck[None, None, None, page_id, bn_id]
                if fx.const_expr(use_k_ring):
                    if fx.const_expr(num_copy_threads == num_threads):
                        fx.copy(glk_load_atom, source, prefetch_fragk)
                    else:
                        if tid < num_copy_threads:
                            fx.copy(glk_load_atom, source, prefetch_fragk)
                else:
                    if fx.const_expr(num_copy_threads == num_threads):
                        fx.copy(glk_load_atom, source, prefetch_fragk_list[frag_id])
                    else:
                        if tid < num_copy_threads:
                            fx.copy(glk_load_atom, source, prefetch_fragk_list[frag_id])
                return num_vm_cnt_load_k
            else:
                return 0

        def ds_store_k(block_n, frag_id, lds_buff_id):
            if fx.const_expr(is_valid_block_n(block_n)):
                if fx.const_expr(use_k_ring):
                    source = prefetch_fragk
                else:
                    source = prefetch_fragk_list[frag_id]
                if fx.const_expr(num_copy_threads == num_threads):
                    fx.copy(
                        glk_store_atom,
                        source,
                        glk_dstk[None, None, None, lds_buff_id],
                    )
                else:
                    if tid < num_copy_threads:
                        fx.copy(
                            glk_store_atom,
                            source,
                            glk_dstk[None, None, None, lds_buff_id],
                        )
    
        fragO.fill(0.0)

        v_copy_atom = flyobj.get_universal_copy_atom(v_tile.dtype, 128)
        v_tcopy = flyobj.get_tiled_mma_copy(v_copy_atom, tmma2, "A")
        v_thrcopy = v_tcopy.get_slice(tid)

        def kv_step(page_n, lds_buff_id, cur_max, l_in,
                    kv_page_id0, kv_page_id1, kv_page_id2, kv_page_id3,
                    is_all_kv_valid: fx.Constexpr[bool] = True):
            # first block_n in pipeline is -3
            kv_page_id4 = ptr_kv_page_table[page_n + 4]

            kv_page_0123 = [kv_page_id0, kv_page_id1, kv_page_id2, kv_page_id3]

            for bn_i in fx.range_constexpr(num_BN_per_page):
                bn0_page = kv_page_0123[(bn_i + 0)//num_BN_per_page]
                bn0_part = (bn_i + 0) % num_BN_per_page
                bn3_page = kv_page_0123[(bn_i + 3)//num_BN_per_page]
                bn3_part = (bn_i + 3) % num_BN_per_page

                pipeline_block_n = page_n * num_BN_per_page + bn_i
                logical_block_n = (
                    (first_page + fx.Int32(page_n)) * num_BN_per_page + bn_i
                )

                # Q@K part for block_n
                prefetch_frag_id = lds_buff_id^1
                vm_cnt = 0

                if fx.const_expr(not use_k_ring):
                    ds_store_k(
                        pipeline_block_n + 1,
                        prefetch_frag_id,
                        lds_buff_id ^ 1,
                    )
                vm_cnt += global_load_k(
                    pipeline_block_n + 3,
                    bn3_page,
                    bn3_part,
                    prefetch_frag_id,
                )
                
                if fx.const_expr(is_valid_block_n(pipeline_block_n)):
                    fragS.fill(0.0)
                    #s_waitcnt(lgkmcnt=0)
                    fx.gemm(tmma1, fragS, fragK, fragQ, fragS)
                    fx.copy(
                        v_copy_atom,
                        v_thrcopy.partition_S(v_tile[None, None, bn0_page, bn0_part]),
                        v_thrcopy.retile(fragV),
                    )

                    vm_cnt += num_vm_cnt_load_v
                    #assert 0, f"{vm_cnt} {num_vm_cnt_load_v}"

                    # Interleave all eight V loads with QK MFMA groups. The
                    # trailing catch-all handles the remaining K8/K16/K64 ops.
                    fx.rocdl.sched_group_barrier(0x200, 1, 0)
                    fx.rocdl.sched_mfma(2)
                    fx.rocdl.sched_vmem(1)
                    for _ in fx.range_constexpr(num_vm_cnt_load_v//2):
                        fx.rocdl.sched_mfma(3)
                        fx.rocdl.sched_vmem(2)
                    fx.rocdl.sched_vmem(100)
                    fx.rocdl.sched_mfma(100)

                rocdl.sched_barrier(0)
                fxh.s_waitcnt(vmcnt=vm_cnt, lgkmcnt=0)
                rocdl.s_barrier() # ::::::::: wave-group barrier ::::::::: 切换调度
                rocdl.s_setprio(0)
                rocdl.sched_barrier(0)

                if fx.const_expr(is_valid_block_n(pipeline_block_n)):
                    # q_pos0, kv_len
                    if fx.const_expr(k_tile.dtype == fx.BFloat16):
                        cur_max, l_in, probability_operand = online_softmax(
                            fragS, fragO, qk_scale_log2, cur_max, l_in,
                            q_pos0, logical_block_n, kv_len, full_qo_len,
                            is_all_kv_valid, BN, is_causal, True, window_left,
                            pv_mfma_k,
                        )
                    else:
                        cur_max, l_in = online_softmax(
                            fragS, fragO, qk_scale_log2, cur_max, l_in,
                            q_pos0, logical_block_n, kv_len, full_qo_len,
                            is_all_kv_valid, BN, is_causal, False, window_left,
                        )

                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                rocdl.sched_barrier(0)

                # MFMA-stage :
                #   1st half: P@V part for block_n
                #   2nd half: Q@K part for block_n+1

                if fx.const_expr(use_k_ring and is_valid_block_n(pipeline_block_n + 3)):
                    fxh.s_waitcnt(vmcnt=0)
                    ds_store_k(
                        pipeline_block_n + 3,
                        prefetch_frag_id,
                        (pipeline_block_n + 3) % k_lds_stages,
                    )
                
                if fx.const_expr(is_valid_block_n(pipeline_block_n)):
                    if fx.const_expr(k_tile.dtype != fx.BFloat16):
                        vecS = fragS.load()
                        packed_words = []
                        for fn in fx.range_constexpr(4):
                            i = fn * 4
                            lo = rocdl.cvt_pk_fp8_f32(T.i32, vecS[i], vecS[i + 1], fx.Int32(0), False)
                            packed = rocdl.cvt_pk_fp8_f32(T.i32, vecS[i + 2], vecS[i + 3], lo, True)
                            packed_words.append(packed)
                        packed_fp8 = Vec.from_elements(packed_words, fx.Int32).bitcast(prob_operand.dtype)
                        prob_operand.store(packed_fp8)
                        probability_operand = prob_operand
                    if fx.const_expr(not use_k_ring):
                        fxh.s_waitcnt(vmcnt=0)
                    fx.gemm(tmma2, fragO, fragV, probability_operand, fragO)

                if fx.const_expr(is_valid_block_n(pipeline_block_n + 1)):
                    # Before, partitioning with the outer `tid` let LLVM hoist
                    # all Dqk=192 LDS addresses before the KV loop:
                    #
                    #   k_addr[0:12] = partition(outer_tid, LDS)
                    #   for each KV block: fragK = ds_read(k_addr[0:12])
                    #
                    # Those address VGPRs then stayed live for the whole loop.
                    # The side-effecting move creates a local SSA root instead:
                    #
                    #   local_tid = opaque_move(outer_tid)
                    #   fragK = ds_read(partition(local_tid, current_LDS_slot))
                    #
                    # Recomputing cheap address ALU at the consumer avoids
                    # keeping twelve address values live or spilling them.
                    k_load_tid = fx.Int32(
                        llvm.inline_asm(
                            fx.Int32.ir_type,
                            [as_ir_value(tid)],
                            "v_mov_b32 $0, $1",
                            "=v,v",
                            has_side_effects=True,
                        )
                    )
                    flyobj.load_tiled_mma_fragA(
                        tmma1,
                        lds_k,
                        [
                            None,
                            None,
                            (
                                (pipeline_block_n + 1) % k_lds_stages
                                if use_k_ring
                                else lds_buff_id ^ 1
                            ),
                        ],
                        dst=fragK,
                        tid=k_load_tid,
                    )


                # leave some LDS bandwidth in head of MFMA-stage
                # because head of online-softmax-stage needs LDS
                for _ in fx.range_constexpr(num_bits_fragK//128//2):
                    fx.rocdl.sched_group_barrier(0x100, 2, 0)
                    fx.rocdl.sched_mfma(3)
                fx.rocdl.sched_mfma(100)
                #fx.rocdl.sched_group_barrier(0x200, 1, 0)
                fx.rocdl.sched_barrier(0)
                lds_buff_id = lds_buff_id^1

            return lds_buff_id, cur_max, l_in, kv_page_id1, kv_page_id2, kv_page_id3, kv_page_id4

        if wave_m == 1:
            gpu.barrier()
        cur_max = fx.Float32(float("-inf"))
        # No sink means an empty online-softmax state. With a sink, seed each
        # 32-lane half with 0.5 so the final xor-32 reduction contributes 1.0.
        l_in = fx.Float32(0.0)
        if fx.const_expr(has_sink):
            cur_max = sink_logit * fx.Float32(LOG2E)
            l_in = fx.Float32(0.5)
        page0, page1, page2, page3 = 0, 0, 0, ptr_kv_page_table[0]
        lds_buff_id = 1
        lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(-3, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)
        lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(-2, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)
        lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(-1, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)

        if fx.const_expr(window_left >= 0):
            num_kv_pages_valid = fx.Int32(0)
            num_kv_pages_to_process = num_kv_pages
        elif fx.const_expr(is_causal):
            # Bottom-right causal diagonal for this Q tile:
            #   kv_pos <= kv_len - full_qo_len + q_pos
            #
            # Pages [0, causal_full_pages) are valid for even the first query
            # row in this tile, so they need no element mask.  Round this
            # prefix down to an even count because the hot loop processes two
            # pages with compile-time LDS buffer IDs 0/1.
            causal_base = kv_len - full_qo_len + q_pos0
            causal_full_pages = (causal_base + 1) // page_size
            num_kv_pages_valid = (causal_full_pages // 2) * 2

            # Only pages intersecting at least one active query row need to be
            # visited by the masked tail.  Later pages are fully causal-masked
            # for the whole Q tile and must be skipped, rather than sent
            # through online softmax as an all-minus-infinity block.
            causal_pages = (causal_base + query_len + page_size - 1) // page_size
            num_kv_pages_to_process = (causal_pages < num_kv_pages).select(
                causal_pages, num_kv_pages
            )
        else:
            # Reserve the final one or two pages for the masked tail.  The last
            # physical page may be ragged; for an even page count its partner
            # is handled by the same specialized pair.
            num_kv_pages_valid = num_kv_pages - 2
            if (num_kv_pages & 1) == 1:
                num_kv_pages_valid = num_kv_pages - 1
            num_kv_pages_to_process = num_kv_pages

        # Seed the loop-carried result outside the loop.  For one-page inputs
        # num_kv_pages_valid is zero, so a value assigned only by `yield`
        # would not dominate the epilogue (and FlyDSL rejects the IR).
        results = [cur_max, l_in, page0, page1, page2, page3]
        for page_i, state in range(0, num_kv_pages_valid, 2, init=results):
            cur_max, l_in, page0, page1, page2, page3 = state
            lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(page_i, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)
            lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(page_i+1, lds_buff_id, cur_max, l_in, page0, page1, page2, page3)
            results = yield [cur_max, l_in, page0, page1, page2, page3]

        # Process the specialized tail in page pairs.  Non-causal has only one
        # or two tail pages; causal may have several pages intersected by this
        # Q tile's diagonal.
        # Keep lds_buff_id as the compile-time constants 0/1: kv_step uses it
        # to index Python fragment lists, so deriving it from the dynamic
        # induction variable (page_i & 1) is not legal FlyDSL.
        for page_i, state in range(
            num_kv_pages_valid, num_kv_pages_to_process, 2, init=results
        ):
            cur_max, l_in, page0, page1, page2, page3 = state
            lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(
                page_i,
                lds_buff_id, cur_max, l_in, page0, page1, page2, page3,
                is_all_kv_valid=False,
            )
            
            if fx.Int32(page_i + 1) < num_kv_pages_to_process:
                lds_buff_id, cur_max, l_in, page0, page1, page2, page3 = kv_step(
                    page_i+1,
                    lds_buff_id, cur_max, l_in, page0, page1, page2, page3,
                    is_all_kv_valid=False,
                )
            results = yield [cur_max, l_in, page0, page1, page2, page3]

        cur_max, l_in, page0, page1, page2, page3 = results
        l = fxh.eltwise_op("v_add_f32", l_in, l_in.shuffle_xor(32, 64))
        output_scale = v_s / l
        fragO.store(fxh.eltwise_op("v_mul_f32", fragO.load(), output_scale))

        fragO_bf16 = fxh.cvt_f32_to_bf16(fragO)

        if fx.const_expr(0):
            # direct store to vmem
            if wave_m == 0:
                gpu.barrier()

            flyobj.store_tiled_mma_fragC(tmma2, fragO_bf16, fx.select(o_tile, [1,0]), copy_atom_bits=64)
        else:
            # 128-bit C-shuffle epilogue:
            #   MFMA C registers --64b--> LDS --128b--> registers --128b--> HBM.
            # The first barrier makes it safe to reuse the K/O union storage;
            # the next ticket-fetch barrier protects the final transition back
            # to K, so only intermediate source waves need a trailing barrier.
            if wave_m == 0:
                gpu.barrier()

            # C-shuffle aliases the same output LDS bytes through two layouts:
            # tmma2 writes its logical C=(N, M) fragment with N contiguous, while
            # the epilogue reads the physical tensor as row-major (M, N).  The
            # bf16 swizzle removes the bank conflicts from the 64-bit C stores.
            swz_o = fx.SwizzleType.get(3, 3, 3)
            layout_o_lds_store = fx.make_composed_layout(
                fx.static(swz_o),
                fx.make_ordered_layout((head_dim_v, BM//num_waves), order=(0, 1)),
            )
            assert head_dim_v % 8 == 0, f"{head_dim_v=} must be a multiple of 8"
            num_dw4_items = (head_dim_v // 8) * (BM//num_waves)
            layout_o_lds_read = fx.make_composed_layout(
                fx.static(swz_o),
                fx.make_ordered_layout((8, num_dw4_items), order=(0, 1)),
            )
            o_lds_store = lds.o_lds.peek().view(layout_o_lds_store)
            o_lds_read = lds.o_lds.peek().view(layout_o_lds_read)

            # Before:
            #   cshuffle_addresses = partition(thread_id, output_LDS)
            #   run_entire_KV_loop()
            #   store_output(cshuffle_addresses)
            #
            # LLVM hoisted the epilogue partition and kept dozens of address
            # VGPRs live throughout attention, indirectly forcing hot-loop K
            # values to scratch. Anchor the address tree after the loop:
            #
            #   run_entire_KV_loop()
            #   epilogue_tid = opaque_move(thread_id)
            #   store_output(partition(epilogue_tid, output_LDS))
            #
            # The addresses now exist only during the epilogue.
            epilogue_tid = fx.Int32(
                llvm.inline_asm(
                    fx.Int32.ir_type,
                    [as_ir_value(fx.thread_idx.x)],
                    "v_mov_b32 $0, $1",
                    "=v,v",
                    has_side_effects=True,
                )
            )
            epilogue_lane_id = epilogue_tid % 64
            epilogue_wave_id = epilogue_tid // 64
            cshuf_atom_w = flyobj.get_universal_copy_atom(fx.BFloat16, 64)
            cshuf_store = fx.make_tiled_copy_C(cshuf_atom_w, tmma2).get_slice(
                epilogue_lane_id
            )
            cshuf_atom_r = flyobj.get_universal_copy_atom(fx.BFloat16, 128)
            out_atom_w = flyobj.get_buffer_copy_atom(fx.BFloat16, 128)

            # o_lds_read [32, head_dim_v]
            # o_tile (BM, head_dim_v):(d0, 1)
            o_tile = fx.select(o_tile, [1,0]) # (head_dim_v, BM):(1, d0)
            o_tile = fx.flat_divide(o_tile, [8, 32]) # (8, 32, head_dim_v//8, BM//32):(1, d0)
            o_tile = fx.group(fx.select(o_tile, [0, 2, 1, 3]), 1, 3)

            fragO_r = cshuf_store.retile(fragO_bf16)
            thrv_o_lds_store = cshuf_store.partition_D(o_lds_store)

            for src_wave in fx.range_constexpr(num_waves):
                # due to limited LDS space for output C-shuffle, do it one wave after another
                if epilogue_wave_id == src_wave:
                    fx.copy(cshuf_atom_w, fragO_r, thrv_o_lds_store)

                gpu.barrier()

                frag = fx.make_fragment_like(o_lds_read[None, 0])
                for item in range(epilogue_tid, num_dw4_items, num_threads):
                    src = o_lds_read[None, item]
                    dst = o_tile[None, item, src_wave]
                    fx.copy(cshuf_atom_r, src, frag)
                    fx.copy(out_atom_w, frag, dst)
                # The ticket barrier closes the final C-shuffle lifetime before
                # the next work item reuses the union as K storage.
                if fx.const_expr(src_wave + 1 < num_waves):
                    gpu.barrier()


    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attn_kernel(
        Q_: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        sink_ptr: fx.Tensor,
        O_: fx.Tensor,
        work_counter: fx.Tensor,
    ):
        tid = fx.thread_idx.x

        batch_size = fx.size(cu_seqlens_q.shape).to_py_value() - 1
        shared_allocator = fx.SharedAllocator()
        # Allocate a dedicated LDS word before the 16-byte-aligned attention
        # union. Static SharedAllocator leaves both allocations independently
        # aligned, so the existing K/O bank mapping is unchanged.
        ticket_mailbox = shared_allocator.allocate(
            fx.Array[fx.Int32, 1, 4]
        ).peek()

        #assert 0, f"{Q_}\n{K_}\n{V_}\n{cu_seqlens_q}\n{kv_indptr}\n{kv_page_indices}\n{q_descale}\n{k_descale}\n{v_descale}\n{kv_last_page_lens}\n{out}"

        #if tid == 0:
        #    fx.printf("[{}.{}.{}] batch_size = {}", i_wg, i_head_qo, i_head_kv,  batch_size)

        @flyc.jit
        def fetch_work(work_counter, ticket_mailbox, tid):
            # Only lane 0 of wave 0 performs one device-scope fetch-add for
            # the whole workgroup. One LDS store and one barrier broadcast the
            # result across all eight waves without global mailbox traffic.
            if tid == 0:
                addr = fx.ptrtoint(fx.get_iter(work_counter))
                llvm_ptr = llvm.inttoptr(
                    ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)
                )
                old = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    llvm_ptr,
                    as_ir_value(fx.Int32(1)),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
                ticket_mailbox[0] = fx.Int32(old.result)
                fxh.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            ticket = ticket_mailbox[0]
            fxh.s_waitcnt(lgkmcnt=0)
            return ticket

        @flyc.jit
        def finish_work(work_counter, tid):
            # Every workgroup exits exactly once. The final completion resets
            # the cached header for the next launch on this stream.
            if tid == 0:
                addr = fx.ptrtoint(fx.get_iter(work_counter) + 1)
                llvm_ptr = llvm.inttoptr(
                    ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)
                )
                old = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    llvm_ptr,
                    as_ir_value(fx.Int32(1)),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
                if fx.Int32(old.result) == fx.Int32(fx.grid_dim.x - 1):
                    work_counter[0] = fx.Int32(fx.grid_dim.x)
                    work_counter[1] = fx.Int32(0)

        # Dynamic ticket dispenser: the host initializes the counter to the
        # number of initially resident workgroups.  Each workgroup first owns
        # its block id, then fetches additional work when it finishes.
        linear_work_idx = fx.Int32(fx.block_idx.x)
        batch_i = fx.Int32(0)
        head_i = fx.Int32(0)
        cur_work_idx = fx.Int32(0)
        works_per_head = fx.Int32(((cu_seqlens_q[1] - cu_seqlens_q[0]) + (BM - 1))//(BM))
        k_s = k_descale[0]
        v_s = v_descale[0]

        @flyc.jit
        def skip_works(num_works, cur_work_idx, head_i, batch_i, works_per_head):
            cur_work_idx += num_works
            while (batch_i < batch_size) & (cur_work_idx >= works_per_head):
                cur_work_idx -= works_per_head
                head_i = head_i + 1
                if head_i >= num_qo_heads:
                    head_i = 0
                    batch_i = batch_i + 1
                    if batch_i < batch_size:
                        works_per_head = ((cu_seqlens_q[batch_i + 1] - cu_seqlens_q[batch_i]) + (BM - 1))//(BM)
            return cur_work_idx, head_i, batch_i, works_per_head

        cur_work_idx, head_i, batch_i, works_per_head = skip_works(
            linear_work_idx, cur_work_idx, head_i, batch_i, works_per_head
        )

        while batch_i < batch_size:
            # process the work
            query_pos0 = cur_work_idx * BM
            query_start = cu_seqlens_q[batch_i] + query_pos0
            query_end = fx.Int32(arith.minsi(arith.unwrap(query_start + BM), arith.unwrap(cu_seqlens_q[batch_i + 1])))
            query_len = query_end - query_start
            full_qo_len = cu_seqlens_q[batch_i + 1] - cu_seqlens_q[batch_i]

            kv_ind_start = kv_indptr[batch_i]   # i32
            kv_ind_end = kv_indptr[batch_i + 1] # i32
            num_kv_pages = kv_ind_end - kv_ind_start # i32
            last_page_len = kv_last_page_lens[batch_i]
            if fx.const_expr(key_layout == "linear"):
                kv_len = cu_seqlens_k[batch_i + 1] - cu_seqlens_k[batch_i]
            else:
                kv_len = (num_kv_pages - 1) * page_size + last_page_len


            """
            page_size 是一个在kv-length维度上的天然的分块，因为我们步进 BN 选择了32,
            因此page_size也要求是32的倍数以降低复杂度。
            """
            head_qo = head_i
            head_kv = (head_qo * num_kv_heads) // num_qo_heads

            # process:
            #      Q_[query_start:query_end, head_qo, head_dim]
            #      O_[query_start:query_end, head_qo, head_dim]
            # q_descale[query_start:query_end, head_qo, 1]
            q_tile = fx.make_view(fx.get_iter(Q_) + query_start * num_qo_heads * head_dim_qk,
                                  fx.make_ordered_layout((BM, num_qo_heads, head_dim_qk),(2, 1, 0)))
            q_tile = fx.rocdl.make_buffer_tensor(q_tile, max_size=False,
                                                 num_records_bytes = query_len * num_qo_heads * head_dim_qk * (q_tile.dtype.width // 8))
            q_tile = q_tile[None, head_qo, None]

            if fx.const_expr(quant_query_mode == "per-token"):
                qs_tile = fx.make_view(fx.get_iter(q_descale) + query_start * num_qo_heads,
                                    fx.make_ordered_layout((BM, num_qo_heads),(1, 0)))
                qs_tile = fx.rocdl.make_buffer_tensor(qs_tile, max_size=False,
                                                    num_records_bytes = query_len * num_qo_heads * (qs_tile.dtype.width // 8))
                qs_tile = qs_tile[None, head_qo]
                # [TRICKY#1] this scale assumes 1 32x32 MFMA
                query_in_tile = (fx.Int32(tid // 64) * fx.Int32(32)) + fx.Int32(tid % 32)
                value_q_descale = qs_tile[query_in_tile]
            else:
                # per-tensor
                value_q_descale = (fx.get_iter(q_descale))[0]

            qk_scale_log2 = value_q_descale * k_s * fx.Float32(sm_scale_log2)

            o_tile = fx.make_view(fx.get_iter(O_) + query_start * num_qo_heads * head_dim_v,
                                  fx.make_ordered_layout((BM, num_qo_heads, head_dim_v),(2, 1, 0)))
            o_tile = fx.rocdl.make_buffer_tensor(o_tile, max_size=False,
                                                 num_records_bytes = query_len * num_qo_heads * head_dim_v * (o_tile.dtype.width // 8))
            o_tile = o_tile[None, head_qo, None]

            # Vectorized K: [page, BN-page, kv_head, D/vector, BN, vector]
            # Linear K: [num_kv_tokens, num_kv_heads, head_dim]
            # Public V is [page, kv_head, page/vector, D, vector]. The launch
            # view splits page/vector into [num_BN_per_page, BN/vector].
            #       =>
            # k_tile: [BN, (k_vector_size, head_dim // k_vector_size), num_physical_pages, num_BN_per_page]
            # v_tile: [head_dim, (k_vector_size, BN // k_vector_size), num_physical_pages, num_BN_per_page]
            if fx.const_expr(key_layout == "linear"):
                num_kv_tokens = K.shape[0].to_py_value()
                num_linear_blocks = (num_kv_tokens + BN - 1) // BN
                k_vector_size = 128 // K.dtype.width
                sequence_start = cu_seqlens_k[batch_i]
                k_tile = fx.make_view(
                    fx.get_iter(K) + (sequence_start * num_kv_heads + head_kv) * head_dim_qk,
                    fx.make_layout(
                        (
                            BN,
                            (k_vector_size, head_dim_qk // k_vector_size),
                            num_linear_blocks,
                            1,
                        ),
                        (
                            num_kv_heads * head_dim_qk,
                            (1, k_vector_size),
                            BN * num_kv_heads * head_dim_qk,
                            0,
                        ),
                    ),
                )
                k_tile = fx.rocdl.make_buffer_tensor(
                    k_tile,
                    max_size=False,
                    num_records_bytes=(
                        ((kv_len - 1) * num_kv_heads + 1)
                        * head_dim_qk
                        * (K.dtype.width // 8)
                    ),
                )
            else:
                k_tile = K[None, None, head_kv, None, None, None]
                k_tile = fx.select(k_tile, (3, 4, 2, 0, 1))
                k_tile = fx.group(k_tile, 1, 3)
            # V has a public [page, head, page/vector, D, vector] layout. The
            # launch view splits page/vector into BN pages without moving data.
            v_tile = V[None, None, head_kv, None, None, None] # [num_physical_pages, num_BN_per_page, BN // k_vector_size, head_dim, k_vector_size]
            v_tile = fx.select(v_tile, (3, 4, 2, 0, 1))    # [head_dim, k_vector_size, BN // k_vector_size, num_physical_pages, num_BN_per_page]
            v_tile = fx.group(v_tile, 1, 3)                # [head_dim, (k_vector_size, BN // k_vector_size), num_physical_pages, num_BN_per_page]

            first_page = fx.Int32(0)
            pages_to_process = num_kv_pages
            if fx.const_expr(window_left >= 0):
                absolute_first_q = kv_len - full_qo_len + query_pos0
                first_key = absolute_first_q - fx.Int32(window_left)
                first_key = (first_key > fx.Int32(0)).select(
                    first_key, fx.Int32(0)
                )
                first_page = first_key // page_size
                absolute_last_q_exclusive = absolute_first_q + query_len
                last_page = (
                    absolute_last_q_exclusive + (page_size - 1)
                ) // page_size
                last_page = (last_page < num_kv_pages).select(
                    last_page, num_kv_pages
                )
                pages_to_process = last_page - first_page
            page_table_records = pages_to_process
            if fx.const_expr(window_left >= 0):
                # SGLang appends 256 zero page IDs as a guard region for the
                # pipeline's lookahead loads; they are not logical KV pages.
                page_table_records = num_kv_pages - first_page + 256
            buf_kv_page_table = fx.make_view(
                fx.get_iter(kv_page_indices) + kv_ind_start + first_page,
                fx.make_layout(page_table_records, 1),
            )
            buf_kv_page_table = fx.rocdl.make_buffer_tensor(
                buf_kv_page_table, max_size=False
            )
            sink_logit = fx.Float32(0.0)
            if fx.const_expr(has_sink):
                sink_logit = sink_ptr[head_qo]
            attn_pipeline(q_tile, k_tile, v_tile, o_tile,
                          query_pos0, query_len, kv_len, full_qo_len,
                          buf_kv_page_table,
                          first_page, pages_to_process, last_page_len,
                          qk_scale_log2, v_s, sink_logit,
                          shared_allocator)

            next_linear_work_idx = fetch_work(
                work_counter, ticket_mailbox, tid
            )
            linear_work_delta = next_linear_work_idx - linear_work_idx
            linear_work_idx = next_linear_work_idx
            cur_work_idx, head_i, batch_i, works_per_head = skip_works(
                linear_work_delta, cur_work_idx, head_i, batch_i, works_per_head
            )
        finish_work(work_counter, tid)


    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        sink_ptr: fx.Tensor,
        out: fx.Tensor,
        work_counter: fx.Tensor,
        num_workgroups: fx.Int32,
        stream: fx.Stream,
    ):
        num_query_tokens = Q.shape[0].to_py_value()
        num_physical_pages = V.shape[0].to_py_value()
        k_vector_size = 128 // K.dtype.width
        Q = fxh.view_as_torch_tensor(Q, (num_query_tokens, num_qo_heads, head_dim_qk))
        if fx.const_expr(key_layout == "linear"):
            num_kv_tokens = K.shape[0].to_py_value()
            K = fxh.view_as_torch_tensor(
                K, (num_kv_tokens, num_kv_heads, head_dim_qk)
            )
        else:
            K = fxh.view_as_torch_tensor(
                K,
                (
                    num_physical_pages,
                    num_kv_heads,
                    head_dim_qk // k_vector_size,
                    num_BN_per_page,
                    BN,
                    k_vector_size,
                ),
            )
            K = fx.select(K, (0, 3, 1, 2, 4, 5))
        V = fxh.view_as_torch_tensor(V, (num_physical_pages, num_kv_heads, num_BN_per_page, BN//k_vector_size, head_dim_v, k_vector_size))
        V = fx.select(V, (0, 2, 1, 3, 4, 5))

        if fx.const_expr(quant_query_mode == "per_tensor"):
            q_descale = fx.make_view(fx.get_iter(q_descale), fx.make_layout((num_query_tokens, num_qo_heads, 1), (0, 0, 0)))
        else:
            q_descale = fxh.view_as_torch_tensor(q_descale, (num_query_tokens, num_qo_heads, 1))
        k_descale = fxh.view_as_torch_tensor(k_descale, (1,))
        v_descale = fxh.view_as_torch_tensor(v_descale, (1,))
        out = fxh.view_as_torch_tensor(out, (num_query_tokens, num_qo_heads, head_dim_v))
        value_attrs = {
            "passthrough": [
                ["target-features", "-packed-fp32-ops"] # disable v_pk_mul (which has co-issue problem with MFMA)
            ],
        }
        attn_kernel(
            Q,
            K,
            V,
            cu_seqlens_q,
            cu_seqlens_k,
            kv_indptr,
            kv_page_indices,
            q_descale,
            k_descale,
            v_descale,
            kv_last_page_lens,
            sink_ptr,
            out,
            work_counter,
            value_attrs=value_attrs,
        ).launch(grid=(num_workgroups, 1, 1), block=(num_threads, 1, 1), stream=stream)

    def callable(
        Q: torch.Tensor,  # [num_query_tokens, num_qo_heads, head_dim]
        K: torch.Tensor,  # vectorized [page, kv_head, D/vector, page_size, vector] or linear [token, kv_head, D]
        V: torch.Tensor,  # [num_physical_pages, num_kv_heads, (page_size // k_vector_size, head_dim, k_vector_size)]
        cu_seqlens_q: torch.Tensor,  # [batch_size + 1] cu_seqlens_q[i] ~ cu_seqlens_q[i+1] is the range of query tokens in batch i
        cu_seqlens_k: torch.Tensor,  # [batch_size + 1], required for linear K
        kv_indptr: torch.Tensor,  # [batch_size + 1]    kv_indptr[i] ~ kv_indptr[i+1] is the range of virtual page ids in batch i
        kv_page_indices: torch.Tensor,  # [num_pages] kv_page_indices[i] is the physical page id of virtual page i (used to index into K and V)
        max_seqlen_q: int,  # a hint for scheduler
        max_seqlen_k: int,  # a hint for scheduler
        causal: bool,
        q_descale: torch.Tensor,  # per-token/per-tensor descaling factor for Q, shape [num_query_tokens, num_qo_heads, 1]
        k_descale: torch.Tensor,  # per-tensor descaling factor for K, shape [1]  (per-layer scalar, not per-head nor per-sequence)
        v_descale: torch.Tensor,  # per-tensor descaling factor for V, shape [1]  (per-layer scalar, not per-head nor per-sequence)
        kv_last_page_lens: torch.Tensor,  # [batch_size] kv_last_page_lens[i] is the number of valid tokens in the last page of batch i, used to mask out invalid tokens in the last page
        out: torch.Tensor = None,  # [num_query_tokens, num_qo_heads, head_dim]
        sink_ptr: torch.Tensor = None,  # [num_qo_heads] fp32 attention sink logits
        stream=None,
    ):
        stream = torch.cuda.current_stream() if stream is None else stream

        assert causal == is_causal
        assert not causal or max_seqlen_k >= max_seqlen_q, (
            "bottom-right causal attention requires max_seqlen_k >= max_seqlen_q"
        )
        if cu_seqlens_k is None:
            if key_layout != "vectorized":
                raise ValueError(
                    "cu_seqlens_k=None requires key_layout='vectorized'"
                )
            cu_seqlens_k = cu_seqlens_q
        assert k_descale.numel() == 1
        assert v_descale.numel() == 1
        native_fp8_dtype = (
            torch.float8_e4m3fn
            if is_gfx950
            else torch.float8_e4m3fnuz
        )
        assert Q.dtype in (native_fp8_dtype, torch.bfloat16)
        assert K.dtype == Q.dtype
        assert V.dtype == Q.dtype
        num_query_tokens, _num_qo_heads, _head_dim = Q.shape
        assert _num_qo_heads == num_qo_heads
        assert _head_dim == head_dim_qk
        k_vector_size = 16 // K.element_size()
        num_physical_pages = V.shape[0]
        if key_layout == "linear":
            assert Q.dtype == K.dtype == V.dtype == torch.bfloat16
            assert K.ndim == 3
            assert K.shape[1:] == (num_kv_heads, head_dim_qk)
            assert K.shape[0] == int(cu_seqlens_k[-1].item())
        else:
            assert K.shape == (
                num_physical_pages,
                num_kv_heads,
                head_dim_qk // k_vector_size,
                page_size,
                k_vector_size,
            )
        assert V.shape == (
            num_physical_pages,
            num_kv_heads,
            page_size // k_vector_size,
            head_dim_v,
            k_vector_size,
        )
        batch_size = cu_seqlens_q.shape[0] - 1
        assert cu_seqlens_k.shape == cu_seqlens_q.shape
        assert kv_indptr.shape[0] == batch_size + 1
        assert kv_last_page_lens.shape[0] == batch_size
        if has_sink:
            assert sink_ptr is not None
            assert sink_ptr.shape == (num_qo_heads,)
            assert sink_ptr.dtype == torch.float32
            assert sink_ptr.device == Q.device
        elif sink_ptr is None:
            sink_ptr = q_descale
        # some internal logic use i32 address
        # assert K.numel()*K.element_size() <= 2**31 - 1, f"KV cache size ={K.numel()*K.element_size()} > 2**31 - 1"

        multi_processor_count = torch.cuda.get_device_properties().multi_processor_count
        # One 8-wave workgroup already consumes 36 KiB LDS. A 1/2/3/4
        # WG-per-CU sweep found one best for B=4 (295.49/298.49/300.95/
        # 302.72 us); B=1's 0.4% preference for two was measurement-sized.
        num_workgroups = multi_processor_count
        # Cache the two-word ticket/completion header per device and stream.
        # The last exiting workgroup resets it, so steady calls launch no fill
        # or seed-copy kernels.
        work_counter_cache = getattr(launch, "_work_counter_cache", {})
        work_counter_key = (
            torch.cuda.current_device(),
            stream.cuda_stream,
            num_workgroups,
        )
        work_counter = work_counter_cache.get(work_counter_key)
        if work_counter is None:
            with torch.cuda.stream(stream):
                work_counter = torch.zeros(
                    2, device=Q.device, dtype=torch.int32
                )
                work_counter[0] = num_workgroups
            work_counter_cache[work_counter_key] = work_counter
            launch._work_counter_cache = work_counter_cache

        with torch.cuda.stream(stream):

            if out is None:
                out = torch.empty(
                    (num_query_tokens, num_qo_heads, head_dim_v),
                    dtype=torch.bfloat16,
                    device="cuda",
                )

        if 0:
            # reference implementation using torch.nn.functional.scaled_dot_product_attention
            # de-vectorize & de-quantize K and V
            q_ref = Q.float() * q_descale
            k_cache_ref = (
                K.permute(0, 3, 1, 2, 4).reshape(num_physical_pages, page_size, num_kv_heads, head_dim_qk).float()
                * k_descale
            )
            v_cache_ref = (
                V.permute(0, 2, 4, 1, 3).reshape(num_physical_pages, page_size, num_kv_heads, head_dim_v).float()
                * v_descale
            )
            # reference
            for batch_idx in range(batch_size):
                page0 = kv_indptr[batch_idx]
                page1 = kv_indptr[batch_idx + 1]
                query0 = cu_seqlens_q[batch_idx]
                query1 = cu_seqlens_q[batch_idx + 1]
                pages = kv_page_indices[page0:page1].long().to("cuda")
                kv_len = (page1 - page0 - 1) * page_size + kv_last_page_lens[batch_idx]
                #print(batch_idx, kv_last_page_lens[batch_idx], kv_len)
                k_ref = k_cache_ref[pages].view(-1, num_kv_heads, head_dim_qk)[:kv_len].float()
                v_ref = v_cache_ref[pages].view(-1, num_kv_heads, head_dim_v)[:kv_len].float()
                k_ref = k_ref.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
                v_ref = v_ref.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
                # rows = torch.arange(qo_len, device="cuda").unsqueeze(1)
                # cols = torch.arange(kv_len, device="cuda").unsqueeze(0)
                # causal_mask = cols <= (kv_len - qo_len + rows) if causal else None
                out[query0:query1] = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q_ref[query0:query1].transpose(0, 1).unsqueeze(0),
                        k_ref.transpose(0, 1).unsqueeze(0),
                        v_ref.transpose(0, 1).unsqueeze(0),
                        # attn_mask=causal_mask,
                        is_causal=causal,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
            return out

        if q_descale.ndim == 0: q_descale = q_descale.view(1)
        if k_descale.ndim == 0: k_descale = k_descale.view(1)
        if v_descale.ndim == 0: v_descale = v_descale.view(1)

        compiled_cache = getattr(launch, "_compiled", {})
        def tensor_signature(tensor):
            return tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())

        cache_key = (
            torch.cuda.current_device(),
            torch.cuda.get_device_properties().gcnArchName,
            tensor_signature(Q),
            tensor_signature(K),
            tensor_signature(V),
            tensor_signature(cu_seqlens_q),
            tensor_signature(cu_seqlens_k),
            tensor_signature(kv_indptr),
            tensor_signature(kv_page_indices),
            tensor_signature(q_descale),
            tensor_signature(k_descale),
            tensor_signature(v_descale),
            tensor_signature(kv_last_page_lens),
            tensor_signature(sink_ptr),
            tensor_signature(out),
        )
        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            compiled = flyc.compile(
                launch,
                Q,
                K,
                V,
                cu_seqlens_q,
                cu_seqlens_k,
                kv_indptr,
                kv_page_indices,
                q_descale,
                k_descale,
                v_descale,
                kv_last_page_lens,
                sink_ptr,
                out,
                work_counter,
                num_workgroups,
                stream,
            )
            compiled_cache[cache_key] = compiled
            launch._compiled = compiled_cache
        else:
            compiled(
                Q,
                K,
                V,
                cu_seqlens_q,
                cu_seqlens_k,
                kv_indptr,
                kv_page_indices,
                q_descale,
                k_descale,
                v_descale,
                kv_last_page_lens,
                sink_ptr,
                out,
                work_counter,
                num_workgroups,
                stream,
            )
        return out

    return callable
