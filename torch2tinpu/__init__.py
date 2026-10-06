# -*- coding: utf-8 -*-
"""torch2tinpu —— 把任意 PyTorch 模型转换并部署到 TI NPU 的工具包.

三段流水线:
  Stage A  convert : PyTorch float 模型  --TI-QAT/PTQ 量化-->  int8 ONNX
  Stage B  compile : int8 ONNX           --ti_mcu_nnc 编译-->  mod.a + tvmgen_default.h
  Stage C  deploy  : 编译产物 + CCS 集成说明 + C 调用模板

每个模型只需要提供两个文件:
  - job.yaml    : 量化/编译参数 (模板见 examples/saw2sin/job_saw2sin.yaml)
  - adapter.py  : 模型与数据接口 (契约见 torch2tinpu/adapter.py 顶部注释)

用法 (ti-npu 环境):
  E:\\anaconda\\envs\\ti-npu\\python.exe -m torch2tinpu check   examples\\saw2sin\\job_saw2sin.yaml
  E:\\anaconda\\envs\\ti-npu\\python.exe -m torch2tinpu convert examples\\saw2sin\\job_saw2sin.yaml
  E:\\anaconda\\envs\\ti-npu\\python.exe -m torch2tinpu compile examples\\saw2sin\\job_saw2sin.yaml
  E:\\anaconda\\envs\\ti-npu\\python.exe -m torch2tinpu deploy  examples\\saw2sin\\job_saw2sin.yaml
  E:\\anaconda\\envs\\ti-npu\\python.exe -m torch2tinpu all     examples\\saw2sin\\job_saw2sin.yaml
"""

__version__ = "0.1.0"


class ToolError(Exception):
    """使用者可读错误: CLI 捕获后以 [ERROR] 前缀输出, 不打印 traceback."""
