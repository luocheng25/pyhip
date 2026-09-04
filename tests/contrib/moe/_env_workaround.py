# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""可选的环境修补，供 `test_down_8x1.py` 在异常环境里使用。

两个已知问题（见 docs/design_moe_gemm2_8x1.md §9.1）：

1. `pyhip` 若以 editable 方式装在别的 checkout 上，`import pyhip` 会解析到那份陈旧代码；
   因为 editable finder 是 MetaPathFinder，`sys.path` / `PYTHONPATH` 都盖不住它。
2. `aiter` 的 prebuilt `.so` 与源码不同步时 `import aiter` 直接抛异常，而
   `moe_gemm_2stage/{quant,moe_reduce}.py` 只用到 `tensor_shim._run_compiled`。

正常环境不需要 import 本模块。用法::

    import _env_workaround  # noqa: F401   必须在 import pyhip 之前
"""

import pathlib
import sys
import types

_REPO_SRC = str(pathlib.Path(__file__).resolve().parents[3] / "src")


def redirect_pyhip_to(src_dir=_REPO_SRC):
    """把 editable 安装的 `pyhip` 指向本 checkout。"""
    try:
        import __editable___pyhip_1_0_0_finder as finder
    except ImportError:
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        return
    finder.MAPPING["pyhip"] = src_dir
    for name in [k for k in sys.modules if k == "pyhip" or k.startswith("pyhip.")]:
        del sys.modules[name]


def stub_aiter_if_broken():
    """`import aiter` 失败时，补一个只含 `tensor_shim._run_compiled` 的最小桩。"""
    try:
        import aiter  # noqa: F401

        return False
    except Exception:
        pass
    for name in (
        "aiter",
        "aiter.ops",
        "aiter.ops.flydsl",
        "aiter.ops.flydsl.kernels",
        "aiter.ops.flydsl.kernels.tensor_shim",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    def _run_compiled(*args, **kwargs):
        raise RuntimeError("aiter tensor_shim stub: 真实 aiter 不可用")

    sys.modules["aiter.ops.flydsl.kernels.tensor_shim"]._run_compiled = _run_compiled
    return True


redirect_pyhip_to()
stub_aiter_if_broken()
