"""Entry point of the ``qwen38_monokernel`` SGLang plugin (inert unless ``PYHIP_MONOKERNEL=1``)."""

from pyhip.ops.monokernel.qwen38.sglang_plugin import register  # noqa: F401
