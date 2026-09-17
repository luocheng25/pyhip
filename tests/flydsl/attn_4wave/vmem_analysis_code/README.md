# VMEM文档关联核心代码

整理日期：2026-09-17。对应[GPU内存参数与HTML分析模型文档](../tools/vmem-analysis.md)。

本目录包含此前列出的 **21个核心代码文件副本**，原文件保留，未移动或修改。

| 类别 | 数量 | 主要入口 |
|---|---:|---|
| 硬件参数测量、CU/XCD扫描及结果合并 | 11 | [vmem_hardware_tables.py](vmem_hardware_tables.py)、[vmem_matched_latency128.py](vmem_matched_latency128.py)、[vmem_read_cu_scaling.py](vmem_read_cu_scaling.py)、[vmem_xcd_dense.py](vmem_xcd_dense.py) |
| 当前HTML生成、FIFO模型及前端 | 8 | [build_vmem_fifo_issue.py](build_vmem_fifo_issue.py)、[vmem_fifo_issue.py](vmem_fifo_issue.py)、[vmem_inflight_timeline.js](vmem_inflight_timeline.js) |
| ATT采集与解码 | 2 | [vmem_att_suite.py](vmem_att_suite.py)、[vmem_inflight_suite.py](vmem_inflight_suite.py) |

**这是核心代码快照，不是独立运行包。** 未复制公共依赖、PMC配置、字体、code object、原始trace、测试和生成结果；运行仍需原项目环境及依赖路径。复制过程中没有改写导入、路径或模型参数。