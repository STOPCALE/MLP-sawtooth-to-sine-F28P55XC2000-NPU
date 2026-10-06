# -*- coding: utf-8 -*-
"""Stage A: PyTorch float 模型 -> TI-QAT / PTQ 量化 -> int8 ONNX (+ 内建验收).

流程 (与 TI 官方示例口径一致, 见 tinyml-tensorlab 的 torchmodelopt/examples):
  1. adapter.get_model() / get_example_input() 取模型与 batch=1 样例输入
  2. TINPUTinyMLQATFxModule / TINPUTinyMLPTQFxModule 包装 (注入 NPU 硬件量化约束)
  3. QAT: 逐 epoch 微调;  PTQ: 逐遍校准 (train 模式 + no_grad, 同官方 calibrate)
  4. convert() 冻结 scale -> export(opset=18) 导出 int8 ONNX
  5. 审计 (audit) + 验收 (onnxruntime float vs int8 对比)

坑位提醒 (全部来自实测, 工具已自动处理):
  - batch 必须 =1: 样例输入强制校验
  - opset 必须 18: 默认值 + 审计复核
  - 输出目录隔离: 产物统一落 <out_dir>/<name>/, 不污染工程目录
"""

import copy
import dataclasses
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from . import ToolError
from . import adapter as adapter_mod
from . import envcheck
from .audit import audit_onnx, format_report

_LOSSES = {"mse": nn.MSELoss, "l1": nn.L1Loss, "huber": nn.HuberLoss}


def _import_tinpu_quant():
    try:
        from tinyml_torchmodelopt.quantization import (TinyMLQConfigType,
                                                       TINPUTinyMLPTQFxModule,
                                                       TINPUTinyMLQATFxModule)
    except ImportError as e:
        raise ToolError(
            "无法导入 TI-QAT 包装器 (tinyml_torchmodelopt)。\n"
            "请在 ti-npu 环境运行: E:\\anaconda\\envs\\ti-npu\\python.exe -m torch2tinpu ...\n"
            f"原始错误: {e}"
        )
    return TinyMLQConfigType, TINPUTinyMLQATFxModule, TINPUTinyMLPTQFxModule


def _resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ToolError(f"quant.device 无效: {name} (可用: auto / cpu / cuda)")
    return name


def _as_float_tensor(x, device):
    return torch.as_tensor(x).detach().float().to(device)


def run_convert(job, console=print):
    """Stage A 主流程; 返回 run_info dict."""
    problems = envcheck.check_python_env()
    if problems:
        raise ToolError("依赖缺失: " + "; ".join(problems) +
                        "\n请在 ti-npu 环境运行: E:\\anaconda\\envs\\ti-npu\\python.exe -m torch2tinpu ...")

    torch.manual_seed(job.seed)
    np.random.seed(job.seed)
    random.seed(job.seed)

    os.makedirs(job.run_dir, exist_ok=True)
    console(f"[adapter] {job.adapter_path}")
    ad = adapter_mod.load_adapter(job.adapter_path)

    # ---- 1. 模型与样例输入 ----
    model = adapter_mod.require(ad, "get_model", "返回已加载权重的 float 模型")()
    if not isinstance(model, nn.Module):
        raise ToolError("adapter.get_model() 必须返回 torch.nn.Module")
    model = model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    console(f"[model ] {type(model).__name__} | 参数量 {n_params:,}")

    example = adapter_mod.require(ad, "get_example_input", "返回 batch=1 的样例输入")()
    if not isinstance(example, torch.Tensor):
        raise ToolError("adapter.get_example_input() 必须返回 torch.Tensor")
    example = example.detach().float().cpu()
    if example.dim() < 1 or example.shape[0] != 1:
        raise ToolError(f"样例输入 batch 必须为 1 (TI NPU 硬约束), 当前 shape={tuple(example.shape)}")
    console(f"[model ] 样例输入 shape={tuple(example.shape)} dtype=float32")

    device = _resolve_device(job.quant.device)
    q = job.quant
    console(f"[quant ] method={q.method} | w{q.weight_bitwidth}/a{q.activation_bitwidth} | "
            f"{q.total_epochs} epochs x {q.iters_per_epoch} iters | batch={q.batch_size} | lr={q.learning_rate} | device={device}")

    float_model = copy.deepcopy(model)  # 纯 float 基准 (验收用)

    # ---- 2. TI 包装 ----
    TinyMLQConfigType, QAT_Fx, PTQ_Fx = _import_tinpu_quant()
    qconfig = TinyMLQConfigType(weight_bitwidth=q.weight_bitwidth,
                                activation_bitwidth=q.activation_bitwidth,
                                auto_quantization=False)
    wrap_kwargs = dict(qconfig_type=qconfig, example_inputs=example,
                       total_epochs=q.total_epochs, output_int=q.output_int,
                       verbose=q.verbose)
    if q.num_observer_update_epochs is not None:
        wrap_kwargs["num_observer_update_epochs"] = q.num_observer_update_epochs
    if q.num_batch_norm_update_epochs is not None:
        wrap_kwargs["num_batch_norm_update_epochs"] = q.num_batch_norm_update_epochs
    wrapper_cls = QAT_Fx if q.method == "qat" else PTQ_Fx
    wrapped = wrapper_cls(model, **wrap_kwargs)
    if device == "cuda":
        wrapped = wrapped.to(device)

    # ---- 3. 训练 / 校准 ----
    t_start = time.time()
    if q.method == "qat":
        _run_qat(job, wrapped, ad, device, console)
    else:
        _run_ptq(job, wrapped, ad, device, console)
    train_secs = time.time() - t_start

    # ---- 4. convert -> export ----
    wrapped.eval()
    wrapped = wrapped.convert()            # 冻结 scale + 转 CPU + 整数运算图
    torch.manual_seed(job.seed)            # 固定导出输入, 保证可复现
    wrapped.export(torch.rand(*example.shape), job.int8_onnx,
                   opset_version=q.opset_version,
                   input_names=["input"], output_names=["output"])
    console(f"[int8  ] 导出完成: {job.int8_onnx}")

    # ---- 5. 审计 + 验收 ----
    findings = audit_onnx(job.int8_onnx, console=console)
    with open(os.path.join(job.run_dir, "audit_report.md"), "w", encoding="utf-8") as f:
        f.write(format_report(job.int8_onnx, findings))

    verify = None
    if job.verify.enabled:
        verify = _verify(job, ad, float_model, console)
        with open(os.path.join(job.run_dir, "verify_report.json"), "w", encoding="utf-8") as f:
            json.dump(verify, f, indent=2, ensure_ascii=False, default=float)

    # ---- 6. 记录 ----
    info = {
        "name": job.name,
        "time": datetime.now().isoformat(timespec="seconds"),
        "adapter": job.adapter_path,
        "int8_onnx": job.int8_onnx,
        "quant_settings": dataclasses.asdict(q),
        "train_seconds": round(train_secs, 1),
        "verify": verify,
        "audit": findings,
    }
    with open(os.path.join(job.run_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)

    console(f"[done  ] 产物目录: {job.run_dir}")
    if job.compile.enabled:
        console(f"[next  ] python -m torch2tinpu compile \"{job.path}\"")
    return info


def _run_qat(job, wrapped, ad, device, console):
    q = job.quant
    sample_batch = adapter_mod.require(ad, "sample_batch",
                                       "在 QAT 模式下生成随机训练批次 (x, y)")
    loss_fn = _LOSSES[q.loss]()
    params = [p for p in wrapped.parameters() if p.requires_grad]
    if q.optimizer == "adam":
        opt = torch.optim.Adam(params, lr=q.learning_rate)
    elif q.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=q.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(params, lr=q.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=q.total_epochs) \
        if q.scheduler == "cosine" else None

    console("[quant ] QAT 微调开始 ...")
    for epoch in range(q.total_epochs):
        wrapped.train()                    # 每 epoch 只调一次 (推进温度/冻结调度)
        epoch_loss, t0 = 0.0, time.time()
        for _ in range(q.iters_per_epoch):
            x, y = adapter_mod.call_supported(sample_batch, q.batch_size)[:2]
            x = _as_float_tensor(x, device)
            y = _as_float_tensor(y, device)
            loss = loss_fn(wrapped(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        if sched is not None:
            sched.step()
        console(f"[quant ] epoch {epoch + 1:3d}/{q.total_epochs} | "
                f"loss {epoch_loss / q.iters_per_epoch:.3e} | {time.time() - t0:.1f}s")


def _run_ptq(job, wrapped, ad, device, console):
    """PTQ: 逐遍校准 (与官方 calibrate 一致: train 模式 + no_grad 前向, observers 收范围)."""
    q = job.quant
    calib_fn = getattr(ad, "get_calibration_batches", None)
    sample_batch = getattr(ad, "sample_batch", None)
    if not callable(calib_fn) and not callable(sample_batch):
        raise ToolError("PTQ 模式需要 adapter 提供 sample_batch() 或 get_calibration_batches()")

    console("[quant ] PTQ 校准开始 ...")
    for epoch in range(q.total_epochs):
        wrapped.train()
        t0 = time.time()
        with torch.no_grad():
            if callable(calib_fn):
                for x in adapter_mod.call_supported(calib_fn, q.iters_per_epoch):
                    wrapped(_as_float_tensor(x, device))
            else:
                for _ in range(q.iters_per_epoch):
                    x = adapter_mod.call_supported(sample_batch, q.batch_size)[0]
                    wrapped(_as_float_tensor(x, device))
        console(f"[quant ] calibration {epoch + 1:3d}/{q.total_epochs} | {time.time() - t0:.1f}s")


def _verify(job, ad, float_model, console):
    """内建验收: float 模型 vs int8 ONNX 输出对比 (+ adapter.evaluate 自定义指标)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess = ort.InferenceSession(job.int8_onnx, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    exp_shape = tuple(sess.get_inputs()[0].shape)      # e.g. (1, 1, 10, 1)

    # ---- 验收输入 ----
    n = job.verify.num_samples
    get_eval = getattr(ad, "get_eval_inputs", None)
    if callable(get_eval):
        xs = adapter_mod.call_supported(get_eval, n)
    else:
        sample_batch = adapter_mod.require(ad, "sample_batch",
                                           "在未提供 get_eval_inputs() 时提供验收输入")
        xs = adapter_mod.call_supported(sample_batch, n)[0]
    if isinstance(xs, torch.Tensor):
        xs = [xs[i] for i in range(xs.shape[0])]
    else:
        xs = list(xs)

    float_model = float_model.eval().cpu()

    def float_predict(x_np):
        with torch.no_grad():
            y = float_model(torch.from_numpy(np.asarray(x_np, dtype=np.float32)))
        return y.cpu().numpy()

    def quant_predict(x_np):
        return sess.run(None, {in_name: np.asarray(x_np, dtype=np.float32)})[0]

    # ---- 逐样本对比 ----
    diffs_max, diffs_mean, skipped = [], [], 0
    for x in xs:
        x = torch.as_tensor(x).detach().float().cpu()
        if tuple(x.shape) == exp_shape:
            xin = x
        elif tuple(x.shape) == exp_shape[1:]:
            xin = x.unsqueeze(0)
        else:
            skipped += 1
            continue
        x_np = xin.numpy()
        yf = np.asarray(float_predict(x_np)).ravel()
        yq = np.asarray(quant_predict(x_np)).ravel()
        m = min(yf.size, yq.size)
        d = np.abs(yf[:m] - yq[:m])
        if d.size:
            diffs_max.append(float(d.max()))
            diffs_mean.append(float(d.mean()))

    result = {
        "num_samples": len(diffs_max),
        "skipped_wrong_shape": skipped,
        "diff_max": max(diffs_max) if diffs_max else None,
        "diff_mean": float(np.mean(diffs_mean)) if diffs_mean else None,
    }
    if diffs_max:
        console(f"[verify] float vs int8 | n={result['num_samples']} | "
                f"max |d|={result['diff_max']:.4e} | mean |d|={result['diff_mean']:.4e}")
    if skipped:
        console(f"[verify] 跳过 {skipped} 个形状不匹配的样本 (预期 {exp_shape} 或 {exp_shape[1:]})")

    # ---- 自定义指标 ----
    ev = getattr(ad, "evaluate", None)
    if callable(ev):
        try:
            fm = {k: float(v) for k, v in dict(ev(float_predict)).items()}
            qm = {k: float(v) for k, v in dict(ev(quant_predict)).items()}
            result["float_metrics"] = fm
            result["quant_metrics"] = qm
            console(f"[verify] 自定义指标 float : {fm}")
            console(f"[verify] 自定义指标 int8  : {qm}")
        except Exception as e:  # 自定义指标不阻断主流程
            console(f"[verify] adapter.evaluate() 调用失败 (忽略): {e}")
    return result
