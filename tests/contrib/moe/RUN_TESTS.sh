#!/bin/bash
# MOE GEMM 1x4 prefill gateup 测试命令集
# 切换机器后: source /path/to/.env/bin/activate && cd /path/to/pyhip/tests/contrib/moe && bash RUN_TESTS.sh
#
# 环境要求:
#   - PATH 中有 ninja (pip install ninja)
#   - FLYDSL_RUNTIME_ENABLE_CACHE=0 确保每次重新编译
#   - 性能测试用 profile_gateup.py, 精度测试用 test_moe.py

ENV_BIN="/raid/users/xisun/luocheng/.env/bin"
PYTHON="$ENV_BIN/python3"
export PATH="$ENV_BIN:$PATH"
export FLYDSL_RUNTIME_ENABLE_CACHE=0

cd "$(dirname "$0")"

# ============================================================
# 精度测试 (test_moe.py, 默认参数 E=64)
# 期望: 全部 acc OK, bf16 diff≈0.00002, fp8 diff≈0.00074
# ============================================================
run_accuracy() {
    echo "=== 精度测试 ==="
    $PYTHON test_moe.py 2>&1 | tee /tmp/acc_test.txt
    echo ""
    echo "--- 统计 ---"
    grep -c 'acc OK' /tmp/acc_test.txt | xargs -I{} echo "通过: {} 个"
    grep -v 'acc OK' /tmp/acc_test.txt | grep -ci 'fail\|error\|mismatch' | xargs -I{} echo "失败: {} 个"
}

# ============================================================
# 性能测试 Set1: E=192 TOPK=8 INTER_TP=192 (N1=384, K1=4096)
# 对应模型: 小规模 MoE
# 注意: N1=384, 384%256≠0, 所以 BN 只能用 128
# ============================================================
run_perf_set1() {
    echo "=== 性能测试 Set1: E=192 TOPK=8 INTER_TP=192 ==="
    $PYTHON profile_gateup.py \
        waves=1x4 \
        dtypes=bf16,per_tensor,ptpc \
        batches=1024,2048,4096,8192,16384 \
        bms=64 \
        bns=128 \
        bks=64,128,256 \
        e=192 topk=8 inter_tp=192 \
        2>&1 | tee /tmp/prof_set1.txt
}

# ============================================================
# 性能测试 Set2: E=512 TOPK=10 INTER_TP=128 (N1=256, K1=4096)
# 对应模型: 大规模 MoE (需 buf_copy=5 防止 OOM)
# 注意: BN=256 时 N1=256, 256%256=0 可用
# ============================================================
run_perf_set2() {
    echo "=== 性能测试 Set2: E=512 TOPK=10 INTER_TP=128 ==="
    $PYTHON profile_gateup.py \
        waves=1x4 \
        dtypes=bf16,per_tensor,ptpc \
        batches=1024,2048,4096,8192,16384 \
        bms=64 \
        bns=128,256 \
        bks=64,128,256 \
        e=512 topk=10 inter_tp=128 buf_copy=5 \
        2>&1 | tee /tmp/prof_set2.txt
}

# ============================================================
# 汇总 B=16384 峰值性能
# ============================================================
summarize() {
    for f in /tmp/prof_set1.txt /tmp/prof_set2.txt; do
        [ -f "$f" ] || continue
        echo ""
        echo "=== $(head -1 "$f" | grep -oP 'E=\S+ TOPK=\S+ .* K1=\d+') B=16384 ==="
        grep -E '^\s*1x4' "$f" | grep '16384' | sort -k2,2 -k3,3n -k4,4n -k5,5n
    done
}

# ============================================================
# 历史基线 (B=16384 TFLOPS, BM=64)
# ============================================================
# Set1 基线 (commit 62eb2b7, E=192 TOPK=8 INTER_TP=192, N1=384):
#   注意: N1=384, 384%256≠0, BN只能用128
#   BM×BN×BK     bf16    fp8_pt   fp8_ptpc
#   64×128×64    173.9      -        -
#   64×128×128   220.7   379.0     370.7
#   64×128×256     -     435.7     425.6
#
# Set1 当前版 (gemm_with_setprio):
#   BM×BN×BK     bf16    fp8_pt   fp8_ptpc
#   64×128×64    192.6      -        -       (+10.8%)
#   64×128×128   217.9   393.6     372.5     (bf16 -1.3%, pt +3.9%)
#   64×128×256     -     420.0     408.1     (pt -3.6%, ptpc -4.1%)
#
# Set2 基线 (commit 62eb2b7, E=512 TOPK=10 INTER_TP=128, N1=256):
#   BM×BN×BK     bf16    fp8_pt   fp8_ptpc
#   64×128×64    173.5      -        -
#   64×128×128   219.1   379.3     373.5
#   64×128×256     -     432.2     428.0
#   64×256×64    218.5      -        -
#   64×256×128   209.2   441.9     436.8
#   64×256×256     -     392.7     386.6
#
# Set2 当前版 (gemm_with_setprio):
#   BM×BN×BK     bf16    fp8_pt   fp8_ptpc
#   64×128×64    164.6      -        -       (-5.1%)
#   64×128×128   217.3   373.2     358.3
#   64×128×256     -     417.6     410.4
#   64×256×64    218.6      -        -
#   64×256×128   139.6   440.0     438.5     (bf16退化, fp8≈持平)
#   64×256×256     -     266.3     263.3     (退化)
#
# 最优配置 (基线):
#   bf16:       64×128×128  ~219-221 TFLOPS
#   fp8_pt:     Set1 64×128×256 ~436 / Set2 64×256×128 ~442
#   fp8_ptpc:   Set1 64×128×256 ~426 / Set2 64×256×128 ~437

# ---- 执行 ----
case "${1:-all}" in
    acc)       run_accuracy ;;
    set1)      run_perf_set1 ;;
    set2)      run_perf_set2 ;;
    summary)   summarize ;;
    all)
        run_accuracy
        run_perf_set1
        run_perf_set2
        summarize
        ;;
    *)
        echo "用法: bash RUN_TESTS.sh [acc|set1|set2|summary|all]"
        exit 1
        ;;
esac
