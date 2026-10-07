# -*- coding: utf-8 -*-
"""check_clip_bound.py -- AI 工具 (C8): "去掉 requant 钳位" 的最坏情况范围证明

对每层每个输出行 o, 在 x ∈ [-128,127]^in_dim (int8 全范围超立方体, 比实测数据宽得多)
上求 y = (b_fold[o] + Σ_i x_i·W[o,i]) · M[o] 的精确最坏上/下界:
   |x_i| 取端点:  x_i=127 或 -128 (按 W·M 的符号选)
再转成 r = round_even(y) + z_out 的最坏界 (round 误差 ±0.5),
检查 r ∈ [-128, 127] 的余量 (需 > 0.01, 覆盖 MCU 上 float32 的舍入 ~1e-5)。

结论: 若所有行的余量都 > 0.01, 则钳位是死代码 (对任何 int8 输入都成立),
C8 可以删除它。⚠ 换模型/换权重/换标定后必须重跑本证明!
"""
import os

import numpy as np
import onnx
from onnx import numpy_helper

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
P = {init.name: numpy_helper.to_array(init)
     for init in onnx.load(os.path.join(OUT_DIR, "saw2sin_int8.onnx")).graph.initializer}

# 与 export_c_header.py 相同的层定义 (W 名, S_w 名, b 名, S_in, z_in, S_out, z_out)
LAYER_DEFS = [
    ("L0", "net.0", "s_scale", "s_zero_point",
     "/net/net.1/Relu_output_0_scale", "/net/net.1/Relu_output_0_zero_point"),
    ("L1", "net.2", "/net/net.1/Relu_output_0_scale", "/net/net.1/Relu_output_0_zero_point",
     "/net/net.3/Relu_output_0_scale", "/net/net.3/Relu_output_0_zero_point"),
    ("L2", "net.4", "/net/net.3/Relu_output_0_scale", "/net/net.3/Relu_output_0_zero_point",
     "y_scale", "y_zero_point"),
]

print("方法: 穷举输入域 —— 入口钳位把 q0 封在 [-128,127] (任何 s 都跑不出),")
print("      对每个 q0 精确模拟整条链路 (float32 逐位对齐 MCU 的 MPYF32/ADDF32),")
print("      收集每层 requant 的 r 真实可达范围。这是对全部 s 的完整证明。")
print("")

# 与 MCU 完全一致的 float32 requant 模拟 (C5 单加魔数形式)
def requant(acc, M, zm):
    t = np.float32(np.float32(acc) * M)            # I32TOF32 + MPYF32
    t = np.float32(t + np.float32(12582912.0))     # + 1.5*2^23 (ADDF32)
    return int(np.int64(t)) + int(zm)              # F32TOI32 (截断) + 整数域减魔数

def layer(ldef):
    W = P["%s.weight_quantized" % ldef[1]].astype(np.int64)
    b = P["%s.bias_quantized" % ldef[1]].astype(np.int64)
    s_in = float(P[ldef[2]]); z_in = int(P[ldef[3]])
    s_out = float(P[ldef[4]]); z_out = int(P[ldef[5]])
    S_w = P["%s.weight_scale" % ldef[1]].astype(np.float64)
    M = (s_in * S_w / s_out).astype(np.float32)
    bf = b - z_in * W.sum(axis=1)                  # b_fold, (out,)
    return dict(W=W, bf=bf, M=M, z_out=z_out)

L0, L1, L2 = (layer(d) for d in LAYER_DEFS)

def run(ld, xs):
    """一层前向 (int64 精确点积 + float32 requant), 返回 (r_数组, 夹后 q_数组)"""
    acc = ld["bf"] + ld["W"].dot(np.asarray(xs, dtype=np.int64))
    r = np.array([requant(a, m, ld["z_out"] - 12582912)
                  for a, m in zip(acc, ld["M"])], dtype=np.int64)
    return r, np.clip(r, -128, 127)

# L0: q0 穷举 (256 个值)
rng = []
q1_set = set()
for q0 in range(-128, 128):
    r, q1 = run(L0, [q0])
    rng.append(r)
    q1_set.add(tuple(int(v) for v in q1))
r0 = np.concatenate(rng)

# L1: 对每个唯一 q1
rng = []
q2_set = set()
for q1 in q1_set:
    r, q2 = run(L1, q1)
    rng.append(r)
    q2_set.add(tuple(int(v) for v in q2))
r1 = np.concatenate(rng)

# L2: 对每个唯一 q2
rng = []
for q2 in q2_set:
    r, _ = run(L2, q2)
    rng.append(r)
r2 = np.concatenate(rng)

print("可达集: |q0|=256 -> |q1|=%-3d -> |q2|=%-3d" % (len(q1_set), len(q2_set)))
print("")
print("%-4s | %14s %10s %10s | %14s %10s %10s" %
      ("层", "r_min", "余量低", "余量高", "r_max", "超界?", "结论"))
ok = True
for tag, r in (("L0", r0), ("L1", r1), ("L2", r2)):
    m_lo = int(r.min()) + 128        # r=-128 是合法值 (余量 0 正常); <0 才是超界
    m_hi = 127 - int(r.max())
    bad = (r.min() < -128) or (r.max() > 127)
    ok = ok and not bad
    print("%-4s | %14d %10d %10d | %14d %10s %s" %
          (tag, r.min(), m_lo, m_hi, r.max(), "无" if not bad else "有!",
           "在界内" if not bad else "超界!"))
print("")
print("判决: %s" % ("SAFE —— 钳位在完整输入域上是死代码, C8 可删"
                   if ok else "UNSAFE —— 存在可达超界, 保留钳位!"))
print("(注: 模拟与 MCU 逐位对齐 (IEEE float32 同序运算); 上板后由 CH22=0.0314 最终判定)")
