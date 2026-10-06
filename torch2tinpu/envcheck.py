# -*- coding: utf-8 -*-
"""环境自检: 解释器 / 依赖包 / C2000 交叉编译器定位.

关键背景:
  - 全部环节都必须在 conda 环境 ti-npu 中运行 (TI 工具链以本地源码 editable 安装在这里)
  - 编译 (Stage B) 需要 C2000 交叉编译器, 通过环境变量 C2000_CG_ROOT 传给 TI 流程;
    本模块负责: 显式配置 > 环境变量 > 自动探测 (E:\\ccs\\ccs\\tools\\compiler\\ti-cgt-c2000_*.LTS)
"""

import glob
import importlib
import importlib.util
import os
import platform
import sys

from . import ToolError

_REQUIRED_MODULES = [
    ("torch", "PyTorch (QAT 微调 / 模型载入)"),
    ("onnx", "ONNX 模型读写"),
    ("onnxruntime", "int8 验收推理"),
    ("yaml", "job 配置"),
    ("tinyml_torchmodelopt", "TI-QAT 包装器 (tinyml-tensorlab 本地源码)"),
    ("tinyml_modelmaker", "TI 编译流程 (tinyml-tensorlab 本地源码)"),
]

_TOOLCHAIN_GLOBS = [
    r"E:\ccs\ccs\tools\compiler\ti-cgt-c2000_*.LTS",
    r"C:\ti\ccs*\tools\compiler\ti-cgt-c2000_*.LTS",
]


def check_python_env():
    """返回缺失依赖列表 (空列表 = 环境正常)."""
    missing = []
    for mod, desc in _REQUIRED_MODULES:
        if importlib.util.find_spec(mod) is None:
            missing.append(f"{mod} ({desc})")
    return missing


def _is_toolchain_root(p):
    return os.path.isfile(os.path.join(p, "bin", "cl2000.exe")) or \
        os.path.isfile(os.path.join(p, "bin", "cl2000"))


def find_toolchain():
    """自动探测 C2000 编译器根目录, 找不到返回 None."""
    hits = []
    for pattern in _TOOLCHAIN_GLOBS:
        hits += glob.glob(pattern)
    hits = sorted({h for h in hits if _is_toolchain_root(h)})
    return hits[-1] if hits else None


def resolve_toolchain(configured=None):
    """优先级: job 显式配置 > 环境变量 C2000_CG_ROOT > 自动探测."""
    if configured:
        p = os.path.abspath(configured)
        if not _is_toolchain_root(p):
            raise ToolError(f"compile.toolchain_root 无效 (找不到 bin/cl2000): {p}")
        return p
    env = os.environ.get("C2000_CG_ROOT")
    if env and _is_toolchain_root(env):
        return os.path.abspath(env)
    found = find_toolchain()
    if found:
        return found
    raise ToolError(
        "未找到 C2000 交叉编译器 (cl2000)。请任选其一:\n"
        "  1) 设置环境变量 C2000_CG_ROOT 指向 ti-cgt-c2000_*.LTS 目录\n"
        "  2) 在 job.yaml 的 compile.toolchain_root 中显式指定"
    )


def describe(console=print):
    """打印环境摘要 (供 check 命令使用), 返回缺失依赖列表."""
    console(f"[env   ] python: {sys.executable} ({platform.python_version()})")
    missing = check_python_env()
    if missing:
        console("[env   ] 缺少依赖: " + "; ".join(missing))
        console("[env   ] 提示: 用 ti-npu 环境运行 -> E:\\anaconda\\envs\\ti-npu\\python.exe")
    else:
        console("[env   ] 依赖包: 齐全 (torch/onnx/onnxruntime/tinyml_* 均可用)")
    for mod in ("torch", "onnx", "onnxruntime"):
        try:
            m = importlib.import_module(mod)
            console(f"[env   ]   {mod}: {getattr(m, '__version__', '?')}")
        except Exception:
            pass
    env = os.environ.get("C2000_CG_ROOT")
    console(f"[env   ] C2000_CG_ROOT 环境变量: {env or '(未设置)'}")
    found = find_toolchain()
    console(f"[env   ] 自动探测编译器: {found or '(未找到)'}")
    return missing
