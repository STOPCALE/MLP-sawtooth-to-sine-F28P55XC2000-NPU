# -*- coding: utf-8 -*-
"""
工具脚本(非教学本体): 在 PC 上"预演"MCU 的 float32 版本, 预测板端会差多少.

MCU 版与 PC 版的唯一差别:
  - M 用 float32 (编译期预计算)
  - acc*M 的乘法在 float32 域完成
  - 其余 (int32 整数累加 / 就近取偶 / clip) 完全相同

用途: 上板前先知道"预期会差几个点", 板端结果一出来即可对照判断移植是否成功.
"""

import os

import numpy as np
import onnx
from onnx import numpy_helper

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
P = {init.name: numpy_helper.to_array(init)
     for init in onnx.load(os.path.join(OUT_DIR, "saw2sin_int8.onnx")).graph.initializer}

DEFS = [
    ("net.0", "s_scale", "s_zero_point",
     "/net/net.1/Relu_output_0_scale", "/net/net.1/Relu_output_0_zero_point"),
    ("net.2", "/net/net.1/Relu_output_0_scale", "/net/net.1/Relu_output_0_zero_point",
     "/net/net.3/Relu_output_0_scale", "/net/net.3/Relu_output_0_zero_point"),
    ("net.4", "/net/net.3/Relu_output_0_scale", "/net/net.3/Relu_output_0_zero_point",
     "y_scale", "y_zero_point"),
]

layers = []
for w, sn, zn, so, zo in DEFS:
    layers.append(dict(
        W=P["%s.weight_quantized" % w].astype(np.int64),          # 提升到 int64 模拟 int32 累加
        b=P["%s.bias_quantized" % w].astype(np.int64),
        S_w=P["%s.weight_scale" % w].astype(np.float32),
        S_in=float(P[sn]), z_in=int(P[zn]),
        S_out=float(P[so]), z_out=int(P[zo])))


def infer_f64(s):
    """PC 参考版: 全 float64 (与 numpy 版逐位一致, 已知与 ORT 相同)."""
    q = np.clip(np.rint(s.astype(np.float64) / layers[0]["S_in"]).astype(np.int64)
                + layers[0]["z_in"], -128, 127).astype(np.int64).reshape(-1, 1)
    mid = []
    for L in layers:
        acc = (q - L["z_in"]) @ L["W"].T + L["b"]
        M = L["S_in"] * L["S_w"].astype(np.float64) / L["S_out"]
        q = np.clip(np.rint(acc.astype(np.float64) * M).astype(np.int64)
                    + L["z_out"], -128, 127).astype(np.int64)
        mid.append(q.astype(np.int8))
    return mid, (np.float32(L["S_out"]) * (q.astype(np.float32) - L["z_out"])).astype(np.float64)


def infer_f32(s):
    """MCU 预演版: M/S_in/S_out 为 float32, acc*M 在 float32 域完成.
       额外返回: max|prod32 - prod64| 与 min(.5 边界余量), 用来量化"为什么没翻车"."""
    si = np.float32(layers[0]["S_in"])
    q = np.clip(np.rint(s.astype(np.float32) / si).astype(np.int64)
                + layers[0]["z_in"], -128, 127).astype(np.int64).reshape(-1, 1)
    mid = []
    d_prod_max = 0.0        # float32 vs float64 的乘积差异上界
    margin_min = 1.0        # 乘积距离 .5 边界的最小余量
    d_M_max = 0.0
    for L in layers:
        acc = (q - L["z_in"]) @ L["W"].T + L["b"]
        M32 = (np.float32(L["S_in"]) * L["S_w"] / np.float32(L["S_out"])).astype(np.float32)
        M64 = L["S_in"] * L["S_w"].astype(np.float64) / L["S_out"]
        d_M_max = max(d_M_max, float(np.abs(M32.astype(np.float64) - M64).max()))

        prod32 = acc.astype(np.float32) * M32                 # <-- float32 乘法
        prod64 = acc.astype(np.float64) * M64
        d_prod_max = max(d_prod_max,
                         float(np.abs(prod32.astype(np.float64) - prod64).max()))
        frac = np.abs(prod64 - np.rint(prod64))               # 到最近整数的距离, [0, 0.5]
        margin_min = min(margin_min, float((0.5 - frac).min()))

        q = np.clip(np.rint(prod32).astype(np.int64) + L["z_out"], -128, 127).astype(np.int64)
        mid.append(q.astype(np.int8))
    y = np.float32(L["S_out"]) * (q.astype(np.float32) - np.float32(L["z_out"]))
    return mid, y.astype(np.float64), d_M_max, d_prod_max, margin_min


N = 1000
s = np.linspace(-1.0, 1.0, N, dtype=np.float32)

mid64, y64 = infer_f64(s)
mid32, y32, d_M, d_prod, margin = infer_f32(s)

print("=" * 72)
print("MCU float32 preview vs PC float64 reference  (N = %d points)" % N)
print("=" * 72)
for k, name in enumerate(["q1", "q2"]):
    d = np.abs(mid32[k].astype(np.int64) - mid64[k].astype(np.int64))
    n_bad = int((d != 0).sum())
    tot = d.size
    print("%s: max diff = %d | mismatching values = %d / %d (%.3f%%)"
          % (name, d.max(), n_bad, tot, 100.0 * n_bad / tot))

dy = np.abs(y32 - y64)
print("y : max diff = %.3e | mismatching values = %d / %d"
      % (dy.max(), int((dy != 0).sum()), N))
print("-" * 72)
print("why the rounding survived float32:")
print("  max |M_f32 - M_f64|          = %.3e" % d_M)
print("  max |prod_f32 - prod_f64|    = %.3e   <- perturbation" % d_prod)
print("  min distance to .5 boundary  = %.3e   <- headroom" % margin)
print("  headroom / perturbation      = %.0fx" % (margin / max(d_prod, 1e-30)))
print("=" * 72)
print("Board acceptance: results must match this run (same float32 arithmetic).")
print("  max diff <= 1 and mismatching ~0.0x%%  =>  port is correct")
print("=" * 72)
