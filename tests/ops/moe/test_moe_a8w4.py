"""A8W4 MoE（FP8 激活 × MXFP4 权重）基本功能测试：公开入口每个分档一例。

每次小改动先跑本文件。``prepare_a8w4_weights`` + ``fused_moe_a8w4`` 跑完整调用，与独立 FP32 参考比较
（``pyhip.testing.moe_a8w4`` 的判据：calc_diff <= 1e-3 且 max|out-ref| <= 2%·max|ref|，无 NaN/Inf），
参考的激活精度按算子实际启动的 kernel 选择，并断言确实启动了 A8W4 / A16W4 kernel 与预期分档的 kernel。
H=2560、topk=11，最后一个 expert 是每个 token 必选的融合 shared expert；E 缩到 33 以控制参考耗时。
三个分档：M≤32 免排序的 decode（TP2 M=3，激活在 kernel 内量化）；M>32 且每个 expert 平均不超过 22 行
（E=33 时 M≤66）的流式路径（TP2 M=64：排序块 32，Gate/Up 尾部量化中间激活）；其余走 prefill（TP2 M=1031；
TP4 M=257，I=160 由 prepare 补零到 192）。prefill 的例子含多块 expert 与部分填充的尾块。其余覆盖见 test_moe_a8w4_extra.py。
"""

import pytest
import torch

from pyhip.ops.moe.flydsl import moe_a8w4
from pyhip.testing import moe_a8w4 as oracle

HIDDEN, EXPERTS, TOPK = 2560, 33, 11


@pytest.fixture(autouse=True)
def _gpu_environment(monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm GPU")
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("the A8W4 kernels target gfx942")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # 测当前源码，不复用旧磁盘编译产物。
    try:
        yield
    finally:
        torch.set_default_device(previous)


def _check(inter, tokens, seed, kernel):
    w = oracle.make_weights(EXPERTS, HIDDEN, inter, seed=seed)
    prepared = moe_a8w4.prepare_a8w4_weights(w["w1"], w["w1_scale"], w["w2"], w["w2_scale"])
    x, topk_weight, topk_ids = oracle.make_inputs(tokens, HIDDEN, EXPERTS, TOPK, seed=seed + 1)
    out = torch.full_like(x, float("nan"))
    result = moe_a8w4.fused_moe_a8w4(x, prepared, topk_weight, topk_ids, output=out)
    assert result is out
    assert any(kernel in name for name in moe_a8w4.last_path), moe_a8w4.last_path
    mode = oracle.path_activation(moe_a8w4.last_path)
    ref = oracle.reference(x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight, topk_ids,
                           activation=mode)
    oracle.assert_close(out, ref)
    return prepared


# kernel：该分档的 kernel 名里一定出现的片段（decode / 流式 / prefill）。
@pytest.mark.parametrize("tokens, kernel", [(3, "_decode"), (64, "_stream_"), (1031, "_prefill_")],
                         ids=["m3", "m64", "m1031"])
def test_tp2(tokens, kernel):
    _check(320, tokens, seed=11, kernel=kernel)


def test_tp4_padded():
    prepared = _check(160, 257, seed=21, kernel="_prefill_")
    assert (prepared.inter, prepared.inter_padded) == (160, 192)
