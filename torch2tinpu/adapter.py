# -*- coding: utf-8 -*-
"""adapter.py 契约与动态加载.

adapter 是你为每个模型写的唯一接口文件 (与 job.yaml 放在一起)。
约定函数如下 —— 只实现用得到的, 标 [必需] 的在该流程里必须存在:

  get_model() -> torch.nn.Module                     [必需]
      返回"已加载权重"的 float 模型 (eval 状态)。结构任意, 但不能含
      数据依赖的动态控制流 (FX 符号追踪要求, 与 TI 官方流程一致)。

  get_example_input() -> torch.Tensor                [必需]
      batch=1 的样例输入张量; 其形状即部署静态形状 (TI NPU 不支持动态 shape)。
      时序流惯例: (1, 1, frame, 1); 图像: (1, C, H, W)。

  sample_batch(batch_size) -> (x, y)                 [QAT 必需]
      随机训练批次 (张量, float32)。PTQ 模式下作为校准数据的回退来源;
      未提供 get_eval_inputs 时也作为验收输入来源。

  get_calibration_batches(num_batches) -> x 的可迭代对象   [PTQ 可选]
      校准输入批次迭代。缺省用 sample_batch 的 x。

  get_eval_inputs(num_samples) -> torch.Tensor       [验收可选]
      验收输入集: 堆叠的 (num_samples, ...) 张量 (首维为样本数, 逐样本 batch=1)
      或逐元素的列表/可迭代对象。缺省用 sample_batch 的 x。

  evaluate(predict) -> dict[str, float]              [验收可选]
      自定义领域指标。predict 是统一接口: 输入 numpy (batch=1, 形状同样例输入),
      输出 numpy。工具会用 float 模型和 int8 ONNX 各调一次, 并列打印。
      例: saw2sin 用它在 s 网格上算相对理论正弦的 max/mean err。
"""

import importlib.util
import inspect
import os
import sys

from . import ToolError


def load_adapter(path):
    """按文件路径加载 adapter 模块 (同时把其所在目录加入 sys.path, 便于 import 同目录辅助文件)."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ToolError(f"adapter 文件不存在: {path}")
    folder = os.path.dirname(path)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    mod_name = "t2t_adapter_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ToolError(f"无法加载 adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def require(mod, fn_name, why):
    """取必需函数; 缺失时报出可读错误."""
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise ToolError(f"adapter 缺少必需的 {fn_name}() —— 需要它{why}")
    return fn


def has(mod, fn_name):
    return callable(getattr(mod, fn_name, None))


def call_supported(fn, *args):
    """按函数签名的位置参数个数调用 (兼容少参/无参写法).

    例: sample_batch 既可以写成 sample_batch(batch_size), 也可以写成无参形式。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args)
    n = 0
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
        elif p.kind == p.VAR_POSITIONAL:
            return fn(*args)
    return fn(*args[:n])
