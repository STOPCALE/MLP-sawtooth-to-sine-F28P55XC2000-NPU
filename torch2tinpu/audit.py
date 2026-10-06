# -*- coding: utf-8 -*-
"""编译前 ONNX 审计: 找出已知会导致 NPU 回落 / 编译异常的图结构.

判据全部来自本工程实测 (F28P55, 详见 docs/saw2sin_case.md):
  A. opset 应为 18 (官方流程默认; 影响 ORT 的图融合行为 -> FC 的 Gemm 形态)
  B. batch 必须为静态 1 (TI NPU 硬约束)
  C. 量化格式: TI-NPU 格式把量化表达为 Add(offset)->Mul(scale)->Floor->Clip;
     若出现 QuantizeLinear/DequantizeLinear/MatMulInteger 等通用量化算子,
     说明是 ORT/fbgemm 系量化产物 -> 不符合 NPU 硬件约束, 上不了 NPU
  D. MatMul(+独立 Add) 形态的全连接 -> TI 编译器只识别 Gemm -> 整体回落 CPU;
     改法: Linear <-> Conv2d(1x1) 等价替换 (首选), 或让输入为 2D 以融合成 Gemm
  E. 1x1 卷积 (PWCONV): 输入/输出通道必须是 4 的倍数
     (实测报错原文: "PWCONV does not support 1 input channels. Please change to multiples of 4.")
  F. Gemm (FC): 8bit 权重要求输入特征数 >= 16 (4bit >= 8)
"""

import os
import re
from collections import Counter

from . import ToolError

_FOREIGN_QUANT_OPS = ("QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul",
                      "MatMulInteger", "ConvInteger", "DynamicQuantizeLinear")


def _dims(vi):
    return [d.dim_value if d.dim_value > 0 else (d.dim_param or "?")
            for d in vi.type.tensor_type.shape.dim]


def _initializer_shape(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


def audit_onnx(onnx_path, console=print):
    """审计 ONNX 并打印摘要; 返回 findings 列表 (可直接写进报告文件)."""
    import onnx  # 延迟导入: check 环境阶段更早报出依赖缺失

    if not os.path.isfile(onnx_path):
        raise ToolError(f"ONNX 文件不存在: {onnx_path}")
    model = onnx.load(onnx_path)
    findings = []

    def add(level, msg):
        findings.append({"level": level, "msg": msg})

    # ---- A. opset ----
    opsets = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    v = opsets.get("ai.onnx")
    if v == 18:
        add("info", "opset=18 (与 TI 官方流程一致)")
    else:
        add("warn", f"opset={v} (官方流程默认 18; 图融合行为可能不同, FC 识别异常时改回 18)")

    # ---- B. 输入 shape / batch ----
    for vi in model.graph.input:
        dims = _dims(vi)
        if any(isinstance(d, str) for d in dims):
            add("warn", f"输入 {vi.name} 含动态维度 {dims}; TI NPU 需要静态形状")
        elif dims and dims[0] != 1:
            add("warn", f"输入 {vi.name} batch={dims[0]}, 必须为 1 (TI NPU 硬约束)")
        else:
            add("info", f"输入 {vi.name} 形状 {dims}")
    for vi in model.graph.output:
        add("info", f"输出 {vi.name} 形状 {_dims(vi)}")

    ops = Counter(n.op_type for n in model.graph.node)
    add("info", f"算子统计: {dict(ops.most_common())}")

    # ---- C. 量化格式 ----
    foreign = {k: ops[k] for k in _FOREIGN_QUANT_OPS if k in ops}
    if foreign:
        add("error", f"检测到通用量化算子 {foreign}: ORT QDQ / fbgemm 等通用量化不符合 "
                     f"TI NPU 硬件约束, 无法编译上 NPU (官方原文: CPU quantized and float "
                     f"models can only execute on CPU)。请用本工具 convert 做 TI-QAT 重新量化。")
    elif ops.get("Floor") and ops.get("Clip"):
        add("info", "量化格式: TI-NPU 风格 (Add/Mul/Floor/Clip 整数表达), 可作为编译输入")
    else:
        add("warn", "未发现 TI-NPU 量化特征 (Floor/Clip) —— 若是 float 模型, 只能编译为 CPU 版本")

    # ---- D. FC 形态: MatMul vs Gemm ----
    if ops.get("MatMul"):
        add("warn", f"存在 {ops['MatMul']} 个 MatMul 节点: TI 编译器只把 Gemm 形态识别为 FC 上 NPU, "
                    f"MatMul(+独立 Add) 形态会整体回落 CPU。改法: (1) Linear 等价替换为 1x1 卷积 "
                    f"Conv2d(ci, co, 1x1) (推荐); (2) 或让输入为 2D 使 ORT 融合成 Gemm")
    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) < 2:
            continue
        trans_b = any(a.name == "transB" and a.i for a in node.attribute)
        shp = _initializer_shape(model, node.input[1])
        if shp and len(shp) == 2:
            ic = shp[1] if trans_b else shp[0]
            if ic < 16:
                add("warn", f"Gemm '{node.name}': 输入特征数 {ic} < 16 (8bit 权重下限), 该 FC 会回落 CPU")

    # ---- E. 1x1 卷积通道约束 ----
    bad_1x1 = 0
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        shp = _initializer_shape(model, node.input[1])
        if not shp or len(shp) != 4:
            continue
        o_ch, c_ch, kh, kw = shp
        if kh == 1 and kw == 1:
            if c_ch % 4 or o_ch % 4:
                bad_1x1 += 1
                add("warn", f"1x1 卷积 '{node.name}': 输入通道 {c_ch} / 输出通道 {o_ch} 非 4 的倍数 "
                            f"-> PWCONV 被拒回落 CPU")
            else:
                add("info", f"1x1 卷积 '{node.name}': {c_ch}->{o_ch} 满足 4 的倍数约束 (预期上 NPU)")
    if bad_1x1:
        add("info", "提示: 通道数不满足约束的 1x1 卷积可以零填充到 4 的倍数 (权重也填 0), 用少量计算量换 NPU 卸载")

    # ---- 打印 ----
    n_err = sum(1 for f in findings if f["level"] == "error")
    n_warn = sum(1 for f in findings if f["level"] == "warn")
    console(f"[audit ] {onnx_path}")
    for f in findings:
        if f["level"] != "info":
            console(f"[audit ] {f['level'].upper():5s} {f['msg']}")
    console(f"[audit ] 结论: {n_err} error / {n_warn} warning (info 级见 audit_report.md)")
    return findings


def format_report(onnx_path, findings):
    """findings -> markdown 文本."""
    lines = ["# ONNX 审计报告", "", f"- 文件: `{onnx_path}`", ""]
    for lvl, title in (("error", "错误 (阻断性问题)"),
                       ("warn", "警告 (会导致 NPU 回落 / 降级)"),
                       ("info", "信息")):
        items = [f["msg"] for f in findings if f["level"] == lvl]
        if not items:
            continue
        lines.append(f"## {title}")
        lines += [f"- {m}" for m in items]
        lines.append("")
    return "\n".join(lines)
