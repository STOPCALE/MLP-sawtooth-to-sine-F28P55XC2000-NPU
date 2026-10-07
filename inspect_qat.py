# -*- coding: utf-8 -*-
"""
诊断工具(只读): 探查 saw2sin_qat_state.pt 里到底存了什么.
不训练、不改文件, 只打印每个 key 的类型 / 形状 / dtype / 量化参数.

用法(在 Charge_Num 目录, torch 环境):
    conda run --no-capture-output -n torch python inspect_qat.py
"""

import os

import torch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(OUT_DIR, "saw2sin_qat_state.pt")

sd = torch.load(CKPT, map_location="cpu")

print("=" * 78)
print("file :", CKPT)
print("type :", type(sd).__name__, "| n_keys :", len(sd))
print("=" * 78)

for k, v in sd.items():
    head = "%-46s %-14s" % (k, type(v).__name__)
    if torch.is_tensor(v):
        head += " shape=%-12s dtype=%s" % (tuple(v.shape), v.dtype)
    print(head)

    if torch.is_tensor(v) and v.is_quantized:
        # 量化张量: 真身整数在 int_repr() 里
        try:
            print("      int_repr[:6] =", v.int_repr().flatten()[:6].tolist())
        except Exception as e:                       # noqa: BLE001
            print("      int_repr err:", repr(e))
        try:
            print("      q_scale = %.8f | q_zero_point = %d"
                  % (float(v.q_scale()), int(v.q_zero_point())))
        except Exception:
            # per-channel 时 q_scale() 会报错, 需走 _scale / buffer
            print("      (per-channel: q_scale() 不可用, 见下面 *_scale 键)")
    elif torch.is_tensor(v) and v.numel() <= 8:
        print("      value =", v.tolist())
    elif torch.is_tensor(v):
        print("      first4 =", v.flatten()[:4].tolist())

    # fbgemm 把真正的权重/bias 打包进 tuple: (int8 权重, float bias)
    if isinstance(v, tuple):
        print("      tuple len =", len(v), "|", [type(t).__name__ for t in v])
        wt = v[0]
        if torch.is_tensor(wt) and wt.is_quantized:
            print("      W_q.int_repr[:6] =", wt.int_repr().flatten()[:6].tolist())
            print("      W_q shape        =", tuple(wt.shape), "dtype", wt.dtype)
            try:
                print("      W  per-channel scale[:4] =",
                      wt.q_per_channel_scales().flatten()[:4].tolist())
                print("      W  per-channel zp[:4]    =",
                      wt.q_per_channel_zero_points().flatten()[:4].tolist())
            except Exception as e:                    # noqa: BLE001
                print("      per-channel scale n/a:", repr(e))
        if len(v) > 1 and torch.is_tensor(v[1]):
            print("      bias dtype =", v[1].dtype, "first4 =", v[1].flatten()[:4].tolist())

print("=" * 78)

# ---------- 附: PTQ 版 QDQ ONNX 的量化参数(通常更干净, 便于手写时对照) ----------
try:
    import onnx
    from onnx import numpy_helper

    onnx_path = os.path.join(OUT_DIR, "saw2sin_int8.onnx")
    m = onnx.load(onnx_path)
    print("QDQ ONNX:", os.path.basename(onnx_path))
    print("  initializers:")
    for init in m.graph.initializer:
        low = init.name.lower()
        if any(s in low for s in ("scale", "zero_point", "quantized", "weight", "bias")):
            arr = numpy_helper.to_array(init)
            flat = arr.ravel()
            print("    %-46s shape=%-12s dtype=%-8s head=%s"
                  % (init.name, tuple(arr.shape), arr.dtype, flat[:4].tolist()))
    print("  nodes:")
    for nd in m.graph.node:
        print("    %-18s in=%-38s out=%s"
              % (nd.op_type, list(nd.input), list(nd.output)))
except Exception as e:                                # noqa: BLE001
    print("onnx dump skipped:", repr(e))

print("=" * 78)
