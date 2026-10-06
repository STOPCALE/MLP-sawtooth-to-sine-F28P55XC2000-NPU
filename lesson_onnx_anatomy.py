# -*- coding: utf-8 -*-
"""
微课 1 & 2 的解剖教具:
  课 1: 打开 saw2sin.onnx, 看到模型在磁盘上"真实的模样"
  课 2: 亲手把一个权重数字量化成 int8, 再还原, 量出误差

运行(在该目录, 已激活 torch 环境):
  python lesson_onnx_anatomy.py

你会看到:
  [1.1] 图的"接线" -> 每个算子吃什么、吐什么
  [1.2] 6 张权重表 -> 模型的全部"知识" (核对总数 = 4353)
  [1.3] 输入/输出张量的名字和形状
  [1.4] W1 的前 8 个真实数值
  [2.1] 为什么输入 scale = 1/127.5
  [2.2] 逐通道量化权重: 手算 scale -> q -> 反量化 -> 误差(<= 半格)
  [2.3] "一个数字的旅程"
"""

import os

import numpy as np
import onnx
from onnx import numpy_helper

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
m = onnx.load(os.path.join(OUT_DIR, "saw2sin.onnx"))
g = m.graph

print("=" * 66)
print("Lesson 1: anatomy of saw2sin.onnx (what a model file REALLY is)")
print("=" * 66)

print("\n[1.1] Graph nodes: the wiring (who eats what, who spits what)")
for i, node in enumerate(g.node):
    print("  #%d  %-6s  in=%s  out=%s"
          % (i, node.op_type, list(node.input), list(node.output)))

print("\n[1.2] Initializers: ALL the 'knowledge' of this model")
total = 0
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    total += arr.size
    print("  %-16s shape=%-10s  params=%d" % (init.name, str(arr.shape), arr.size))
print("  TOTAL = %d    (expect 128 + 4160 + 65 = 4353)" % total)


def dims_of(v):
    return [d.dim_param if d.dim_param else d.dim_value
            for d in v.type.tensor_type.shape.dim]


print("\n[1.3] Input / output tensors:")
for v in g.input:
    print("  input :", v.name, dims_of(v))
for v in g.output:
    print("  output:", v.name, dims_of(v))


def get_init(name):
    for init in g.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


W1 = get_init("net.0.weight")
if W1 is None:                       # 名字兜底: 找形状 (64,1) 的那一张
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.shape == (64, 1):
            W1 = arr
            break

print("\n[1.4] First 8 numbers of W1 (trained knowledge, raw on disk):")
print(" ", np.round(W1.ravel()[:8], 6))

print()
print("=" * 66)
print("Lesson 2: quantize numbers by hand (int8, the deep data view)")
print("=" * 66)

print("\n[2.1] Input s spans [-1, 1].  Symmetric int8 uses q in [-127, 127]")
print("      scale = (1 - (-1)) / 255 = %.6f" % (2.0 / 255))
print("      -> matches the s_scale = 0.007843 printed by onnxruntime earlier")

print("\n[2.2] Weights: per-channel quantization (one ruler per output channel)")
print("      case A: W1 (64 x 1) -- every channel holds just ONE number,")
print("              its ruler is |w|/127, so the round-trip is EXACT (err = 0).")
print("              (that is why onnxruntime showed odd scales like 0.000162!)")
S1 = np.abs(W1).max(axis=1) / 127.0
print("              channel 0: w=%+.6f  S=%.6f  q=%+d  back=%+.6f  err=%.2e"
      % (W1[0, 0], S1[0], int(np.round(W1[0, 0] / S1[0])), W1[0, 0], 0.0))

print("\n      case B: W2 (64 x 64) -- 64 numbers per channel share ONE ruler:")
W2 = get_init("net.2.weight")
S2 = np.abs(W2).max(axis=1) / 127.0
q2 = np.clip(np.round(W2 / S2[:, None]), -127, 127)
W2_dq = q2 * S2[:, None]
err2 = np.abs(W2 - W2_dq)
print("              channel 0 ruler S = %.6f" % S2[0])
for j in range(3):
    print("              w=%+.6f -> q=%+4d -> back=%+.6f   err=%.2e"
          % (W2[0, j], int(q2[0, j]), W2_dq[0, j], err2[0, j]))
print("      max error over ALL 4096 weights = %.2e" % err2.max())
print("      upper bound = largest half-step  = %.2e" % (S2.max() / 2))
print("      -> error NEVER exceeds half a step: the fundamental rule of quantization")

print("\n[2.3] The life of one number (taken from W2):")
i, j = 10, 5
w = float(W2[i, j])
s_i = float(S2[i])
qi = int(np.clip(np.round(w / s_i), -127, 127))
print("      w=%+.6f -> divide by ruler S=%.6f -> %+.3f -> round -> q=%+d"
      % (w, s_i, w / s_i, qi))
print("      q=%+d -> multiply back -> %.6f   (error = %.2e, half-step = %.2e)"
      % (qi, qi * s_i, abs(w - qi * s_i), s_i / 2))

print()
print("DONE. Now answer the check questions in the chat.")
