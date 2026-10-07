# -*- coding: utf-8 -*-
"""
诊断/工具脚本(非教学本体): int8 模型的全量精度体检 + 金标准导出

做什么:
  1. 用 1000 点均匀覆盖 s ∈ [-1,1], 分别跑 float 模型与 int8 模型 (onnxruntime)
  2. 与解析目标 y = -sin(pi*s) 对比 -> 看清"量化到底损失了多少"
  3. 保存金标准向量 golden_vectors.npz (供 C 版对拍用)
  4. 出图 batch_bench.png

用法(在 Charge_Num 目录, torch 环境):
  python batch_bench.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FLOAT_ONNX = os.path.join(OUT_DIR, "saw2sin.onnx")
INT8_ONNX = os.path.join(OUT_DIR, "saw2sin_int8.onnx")
N = 1000

# ---------- 1. 造数据 + 两个模型各跑一遍 ----------
s = np.linspace(-1.0, 1.0, N, dtype=np.float32).reshape(-1, 1)
y_true = -np.sin(np.pi * s)

sess_f = ort.InferenceSession(FLOAT_ONNX, providers=["CPUExecutionProvider"])
sess_q = ort.InferenceSession(INT8_ONNX, providers=["CPUExecutionProvider"])
y_f = sess_f.run(["y"], {"s": s})[0]
y_q = sess_q.run(["y"], {"s": s})[0]

# ---------- 2. 取中间张量(开后门), 给 C 版当"中间结果"对拍的标准 ----------
m = onnx.load(INT8_ONNX)
extra = []
for name in ["/net/net.1/Relu_output_0_QuantizeLinear_Output",
             "/net/net.3/Relu_output_0_QuantizeLinear_Output"]:
    vi = onnx.helper.ValueInfoProto()
    vi.name = name
    m.graph.output.append(vi)
    extra.append(name)
sess_q2 = ort.InferenceSession(m.SerializeToString(),
                               providers=["CPUExecutionProvider"])
q1, q2 = sess_q2.run(extra, {"s": s})
print("intermediate shapes: q1=%s q2=%s" % (q1.shape, q2.shape))


def report(tag, y):
    d = np.abs(y - y_true)
    print("%-14s vs target: max=%.3e  mean=%.3e" % (tag, d.max(), d.mean()))
    return d


print("=" * 66)
d_f = report("float", y_f)
d_q = report("int8 (ORT)", y_q)
print("quantization cost: error grows %.1fx  (%.2e -> %.2e)"
      % (d_q.max() / d_f.max(), d_f.max(), d_q.max()))
print("=" * 66)

# ---------- 3. 导出金标准 ----------
npz = os.path.join(OUT_DIR, "golden_vectors.npz")
np.savez(npz,
         s=s.astype(np.float32),
         y_true=y_true.astype(np.float32),
         y_float=y_f.astype(np.float32),
         y_int8=y_q.astype(np.float32),
         q1_int8=q1,          # (1000,64) int8
         q2_int8=q2)          # (1000,64) int8
print("saved:", npz)
print("  keys: s, y_true, y_float, y_int8, q1_int8, q2_int8")

# ---------- 4. 出图 ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

axes[0].plot(s.ravel(), y_true.ravel(), lw=2, label="target  -sin(pi s)")
axes[0].plot(s.ravel(), y_f.ravel(), "--", lw=1.5, label="float")
axes[0].plot(s.ravel(), y_q.ravel(), "--", lw=1.5, label="int8 (PTQ)")
axes[0].legend()
axes[0].set_title("waveforms over full input range")
axes[0].set_ylabel("y")

axes[1].semilogy(s.ravel(), d_f.ravel() + 1e-12, label="|float - target|")
axes[1].semilogy(s.ravel(), d_q.ravel() + 1e-12, label="|int8  - target|")
axes[1].legend()
axes[1].set_title("absolute error (log scale) - the quantization staircase")
axes[1].set_xlabel("saw value s")
axes[1].set_ylabel("|error|")

plt.tight_layout()
png = os.path.join(OUT_DIR, "batch_bench.png")
plt.savefig(png, dpi=120)
print("saved:", png)
plt.show()
