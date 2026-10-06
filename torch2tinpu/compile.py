# -*- coding: utf-8 -*-
"""Stage B: int8 ONNX -> ti_mcu_nnc 编译 -> mod.a + tvmgen_default.h (+ 报告解析).

做了什么:
  1. 生成 modelmaker 编译配置 (纯 ASCII! TI 脚本按系统默认编码 GBK 读文件)
     - training.quantization: 2  (TI-NPU 格式标记; 缺了会把 type=hard 降级 soft)
     - 全部路径绝对化, 子进程工作目录固定到 <run>/compile/work (产物隔离)
  2. 注入 C2000_CG_ROOT (自动探测), 子进程跑 tinyml_modelmaker, 输出实时透传 + 落盘
  3. 收集产物 (mod.a / tvmgen_default.h / zip / run.log), 解析 run.log 生成 summary.md:
     - NPU 卸载表 / 未卸载算子表 / mod.a 内存占用 / 警告 (回落原因)
"""

import os
import re
import shutil
import subprocess
import sys

import yaml

from . import ToolError
from . import envcheck

_LINE_RE = re.compile(r"^\s*(INFO|WARNING|ERROR)\s*:\s*([^:]+?)\s*:\s*(.*)$")
_ROW_OFFLOAD_RE = re.compile(r"[A-Za-z_][\w]*\s+\d+")
_ROW_OP_RE = re.compile(r"[A-Za-z_.][\w\.]*\s+\d+")
_MEM_RE = re.compile(r"^(Code|RO Data|RW Data|Total)\s*:\s*([\d,]+)\s*bytes")


def _build_modelmaker_yaml(job, onnx_path):
    """生成 modelmaker "BYOM 只编译" 配置 (键名以实测为准, 勿凭文档改)."""
    cfg = {
        "common": {
            "target_module": job.compile.target_module,
            "task_type": job.compile.task_type,
            "target_device": job.compile.target_device,
            "run_name": job.name,
        },
        "dataset": {"enable": False, "dataset_name": job.compile.dataset_name},
        "data_processing_feature_extraction": {"feature_extraction_name": "None"},
        "training": {"enable": False, "model_name": job.name,
                     "quantization": int(job.compile.quantization_version)},
        "testing": {},
        "compilation": {"enable": True, "model_path": onnx_path},
    }
    return cfg


def run_compile(job, console=print):
    onnx_path = job.compile.model_path or job.int8_onnx
    if not os.path.isfile(onnx_path):
        raise ToolError(f"找不到 int8 ONNX: {onnx_path}\n"
                        f"先运行: python -m torch2tinpu convert \"{job.path}\"")

    toolchain = envcheck.resolve_toolchain(job.compile.toolchain_root)
    console(f"[compile] 交叉编译器: {toolchain}")

    compile_dir = os.path.join(job.run_dir, "compile")
    work_dir = os.path.join(compile_dir, "work")
    os.makedirs(work_dir, exist_ok=True)

    cfg = _build_modelmaker_yaml(job, onnx_path)
    yaml_path = os.path.join(compile_dir, "modelmaker_job.yaml")
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False)
    with open(yaml_path, "w", encoding="ascii") as f:  # 必须纯 ASCII (防 GBK 读取崩溃)
        f.write(text)
    console(f"[compile] modelmaker 配置: {yaml_path}")

    env = os.environ.copy()
    env["C2000_CG_ROOT"] = toolchain
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.update({k: str(v) for k, v in (job.compile.extra_env or {}).items()})

    cmd = [sys.executable, "-m", "tinyml_modelmaker.run_tinyml_modelmaker", os.path.abspath(yaml_path)]
    console(f"[compile] cwd: {work_dir}")
    proc = subprocess.Popen(cmd, cwd=work_dir, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    log_lines = []
    for line in proc.stdout:
        log_lines.append(line)
        console(line.rstrip("\n"))
    proc.wait()
    with open(os.path.join(compile_dir, "console.log"), "w", encoding="utf-8") as f:
        f.write("".join(log_lines))
    if proc.returncode != 0:
        raise ToolError(f"modelmaker 编译失败 (returncode={proc.returncode}); "
                        f"完整输出见 {os.path.join(compile_dir, 'console.log')}")

    # ---- 定位产物 ----
    comp = os.path.join(work_dir, "data", "projects", job.compile.dataset_name,
                        "run", job.name, "compilation")
    run_log = os.path.join(comp, "run.log")
    mod_a = os.path.join(comp, "artifacts", "mod.a")
    header = os.path.join(comp, "artifacts", "tvmgen_default.h")
    zip_path = os.path.join(comp, f"{job.name}_{job.compile.target_device}.zip")
    for p in (run_log, mod_a, header):
        if not os.path.isfile(p):
            raise ToolError(f"编译产物缺失: {p}\n请检查 {os.path.join(compile_dir, 'console.log')}")
    shutil.copyfile(run_log, os.path.join(compile_dir, "run.log"))

    # ---- 报告解析 ----
    summary = parse_run_log(run_log)
    summary_md = format_summary(job, onnx_path, mod_a, header,
                                zip_path if os.path.isfile(zip_path) else None, summary)
    with open(os.path.join(compile_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    if summary["offloaded"]:
        console("[compile] NPU 卸载: " + ", ".join(f"{k} x{v}" for k, v in summary["offloaded"].items()))
    else:
        console("[compile] 注意: 未发现任何 NPU 卸载 (全部回落 CPU), 详见 summary.md")
    if summary["memory"]:
        console("[compile] mod.a 内存: " + " | ".join(f"{k} {v}" for k, v in summary["memory"].items()))
    if summary["is_soft"]:
        console("[compile] 注意: 编译为 CPU 软库 (type=soft), 未启用 NPU —— 检查 quantization_version 配置")
    for w in summary["warnings"][:5]:
        console(f"[compile] warn: {w}")
    if len(summary["warnings"]) > 5:
        console(f"[compile] ... 其余 {len(summary['warnings']) - 5} 条警告见 summary.md")

    console(f"[done  ] 编译产物: {mod_a}")
    if job.deploy.enabled:
        console(f"[next  ] python -m torch2tinpu deploy \"{job.path}\"")
    return {"mod_a": mod_a, "header": header, "run_log": run_log, "compile_dir": compile_dir,
            "summary": summary}


def parse_run_log(run_log_path):
    """解析 run.log -> {is_hard/is_soft/offloaded/not_offloaded/memory/warnings}."""
    with open(run_log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    msgs = []
    for ln in lines:
        m = _LINE_RE.match(ln)
        if m:
            msgs.append((m.group(1), m.group(3).strip()))
        elif ln.strip() and not ln.strip().startswith("SUCCESS"):
            msgs.append(("RAW", ln.strip()))

    text = "\n".join(lines)
    info = {
        "is_hard": "type=hard" in text,
        "is_soft": "type=soft" in text,
        "offloaded": {},
        "not_offloaded": {},
        "memory": {},
        "warnings": [],
    }

    section = None
    for sev, msg in msgs:
        if "Layer Patterns Offloaded" in msg:
            section = "offloaded"
            continue
        if "Operators not Offloaded" in msg:
            section = "not_offloaded"
            continue
        if "MODEL PARTITION" in msg or "Memory Usage" in msg:
            section = "other"
            continue

        if section == "offloaded" and _ROW_OFFLOAD_RE.fullmatch(msg):
            name, cnt = msg.split()
            info["offloaded"][name] = int(cnt)
        elif section == "not_offloaded" and _ROW_OP_RE.fullmatch(msg):
            name, cnt = msg.split()
            info["not_offloaded"][name] = int(cnt)

        m = _MEM_RE.match(msg)
        if m:
            kb = re.search(r"\(([^)]+)\)", msg)
            info["memory"][m.group(1)] = m.group(2).replace(",", "") + " B" + \
                (f" ({kb.group(1).strip()})" if kb else "")

        if sev == "WARNING" and msg not in info["warnings"]:
            info["warnings"].append(msg)
    return info


def format_summary(job, onnx_path, mod_a, header, zip_path, s):
    """编译摘要 markdown."""
    lines = [
        f"# 编译报告 — {job.name}",
        "",
        f"- 输入模型: `{onnx_path}`",
        f"- 目标器件: {job.compile.target_device} | task: {job.compile.task_type}",
        f"- 编译模式: {'type=hard (启用 NPU)' if s['is_hard'] else 'type=soft (仅 CPU!)' if s['is_soft'] else '未知'}",
        f"- 产物: `{mod_a}` | `{header}`" + (f" | 打包 `{zip_path}`" if zip_path else ""),
        "",
        "## NPU 卸载 (Layer Patterns Offloaded)",
        "",
    ]
    lines += [f"- {k} x {v}" for k, v in s["offloaded"].items()] or ["- (无)"]
    lines += ["", "## 未卸载算子 (Operators not Offloaded)", ""]
    lines += [f"- {k} x {v}" for k, v in s["not_offloaded"].items()] or ["- (无)"]
    lines += ["", "## mod.a 内存占用", ""]
    lines += [f"- {k}: {v}" for k, v in s["memory"].items()] or ["- (未解析到)"]
    lines += ["", "## 警告 (NPU 回落原因等)", ""]
    lines += [f"- {w}" for w in s["warnings"]] or ["- (无)"]
    lines += ["", "> 完整日志: `compile/run.log`", ""]
    return "\n".join(lines)
