# -*- coding: utf-8 -*-
"""torch2tinpu 命令行入口.

子命令:
  check   环境 + adapter + 模型 自检 (不产生产物)
  convert Stage A: TI-QAT/PTQ 量化 -> int8 ONNX + 审计 + 验收
  compile Stage B: int8 ONNX -> mod.a + tvmgen_default.h
  deploy  Stage C: 生成部署集成包 (INTEGRATION.md + C 模板)
  all     convert -> compile -> deploy 一条龙
"""

import argparse
import os
import sys
import traceback

from . import ToolError, __version__


def safe_print(*args):
    """GBK 控制台安全打印 (不可编码字符替换为 ?)."""
    text = " ".join(str(a) for a in args)
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(enc, "replace").decode(enc, "replace") + "\n")
    sys.stdout.flush()


def _cmd_check(job, console):
    from . import adapter as adapter_mod
    from . import envcheck
    envcheck.describe(console=console)

    ad = adapter_mod.load_adapter(job.adapter_path)
    console(f"[adapter] {job.adapter_path} 已加载")
    model = adapter_mod.require(ad, "get_model", "返回已加载权重的 float 模型")()
    n = sum(p.numel() for p in model.parameters())
    console(f"[model ] {type(model).__name__} | 参数量 {n:,}")
    x = adapter_mod.require(ad, "get_example_input", "返回 batch=1 的样例输入")()
    console(f"[model ] 样例输入: shape={tuple(x.shape)} dtype={x.dtype}")
    if x.shape[0] != 1:
        console("[model ] 注意: batch != 1, convert 会拒绝 (TI NPU 硬约束)")
    for fn, why in (("sample_batch", "QAT 训练 / 校准回退"),
                    ("get_calibration_batches", "PTQ 校准"),
                    ("get_eval_inputs", "验收输入"),
                    ("evaluate", "自定义指标")):
        console(f"[adapter] {fn:24s} {'有' if adapter_mod.has(ad, fn) else '无'}  ({why})")

    if os.path.isfile(job.int8_onnx):
        from .audit import audit_onnx
        audit_onnx(job.int8_onnx, console=console)
    else:
        console(f"[audit ] 未发现 {job.int8_onnx} (先 convert 后再审计)")


def _cmd_convert(job, console):
    from .quant import run_convert
    run_convert(job, console=console)


def _cmd_compile(job, console):
    from .compile import run_compile
    run_compile(job, console=console)


def _cmd_deploy(job, console):
    from .deploy import run_deploy
    run_deploy(job, console=console)


def _cmd_all(job, console):
    from .quant import run_convert
    from .compile import run_compile
    from .deploy import run_deploy
    run_convert(job, console=console)
    if job.compile.enabled:
        run_compile(job, console=console)
        if job.deploy.enabled:
            run_deploy(job, console=console)
    console("[done  ] all: 全流程完成")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="torch2tinpu",
        description="PyTorch -> TI NPU 转换/部署工具 (C2000 F28P55 等)")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (
            ("check", "环境 + adapter + 模型 自检 (不产生产物)"),
            ("convert", "Stage A: TI-QAT/PTQ 量化 -> int8 ONNX + 审计 + 验收"),
            ("compile", "Stage B: int8 ONNX -> mod.a + tvmgen_default.h"),
            ("deploy", "Stage C: 生成部署集成包 (INTEGRATION.md + C 模板)"),
            ("all", "convert -> compile -> deploy 一条龙")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("job", help="job.yaml 路径")
    args = parser.parse_args(argv)

    handlers = {"check": _cmd_check, "convert": _cmd_convert, "compile": _cmd_compile,
                "deploy": _cmd_deploy, "all": _cmd_all}
    try:
        from .job import JobConfig
        job = JobConfig.load(args.job)
        console = safe_print
        console(f"[job   ] {job.name} | 输出目录: {job.run_dir}")
        handlers[args.cmd](job, console)
        return 0
    except ToolError as e:
        safe_print(f"[ERROR] {e}")
        return 1
    except KeyboardInterrupt:
        safe_print("[ERROR] 已中断")
        return 130
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
