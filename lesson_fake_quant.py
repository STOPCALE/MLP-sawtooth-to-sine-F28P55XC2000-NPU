# -*- coding: utf-8 -*-
"""
微课 3 的解剖教具: 伪量化 (fake quant) —— QAT "适应训练"的真相

运行(在该目录, 已激活 torch 环境):
  python lesson_fake_quant.py

你会看到:
  [3.1] 光滑曲线被 int8 量化后的"阶梯"长相 + 误差不超过半格
  [3.2] 对照实验: 裸 round 写法 -> 梯度全 0 (无法训练)
                 STE  写法 -> 梯度直通 (QAT 能训练的原因)
  [3.3] 用这套工具重新解读你的 QAT 日志
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------- 伪量化函数: QAT 内部用的就是这种结构 ----------
def fake_quant_naive(x, S):
    """'天真'写法: round 在 PyTorch 里梯度处处为 0 -> 反向传不回去"""
    q = torch.clamp(torch.round(x / S), -127.0, 127.0)
    return q * S


def fake_quant_ste(x, S):
    """STE 写法 (straight-through estimator):
    前向: 返回量化值 (和上面一模一样); 反向: 让梯度原样通过."""
    q = torch.clamp(torch.round(x / S), -127.0, 127.0)
    y = q * S
    return x + (y - x).detach()      # <- 这一行就是 STE 的全部秘密


print("=" * 66)
print("Lesson 3: fake quant anatomy (what QAT is really doing)")
print("=" * 66)

# ---------- 3.1 阶梯效应 ----------
print("\n[3.1] Smooth curve through int8 quantization: the staircase")
x = torch.linspace(-1, 1, 4001)
y = torch.sin(np.pi * x)              # 和目标同形状
S = y.abs().max() / 127.0             # 对称量化的尺子
yq = fake_quant_naive(y, S)
err = (y - yq).abs()
print("      step size S    = %.6f" % S)
print("      max |y - yq|   = %.6f   (theoretical limit S/2 = %.6f)"
      % (err.max(), S / 2))

fig, axes = plt.subplots(3, 1, figsize=(10, 9))
axes[0].plot(x, y, lw=2, label="original")
axes[0].plot(x, yq, "--", lw=1.5, label="after int8 quantization")
axes[0].legend()
axes[0].set_title("staircase: a smooth curve becomes discrete levels")

zoom = (x > -0.05) & (x < 0.05)
axes[1].plot(x[zoom], y[zoom], lw=2, label="original (zoom)")
axes[1].plot(x[zoom], yq[zoom], "-o", ms=3, lw=1, label="quantized (zoom)")
axes[1].legend()
axes[1].set_title("zoom in: each step is exactly S wide")

axes[2].plot(x, err, label="|error|")
axes[2].axhline(float(S / 2), color="r", ls="--", label="limit = S/2")
axes[2].set_yscale("log")
axes[2].legend()
axes[2].set_title("error is always <= half a step")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lesson_fake_quant.png"), dpi=120)
print("      figure saved: lesson_fake_quant.png")

# ---------- 3.2 梯度对照: 为什么 QAT 能训练 ----------
print("\n[3.2] Backward test: round() has zero gradient - so how can QAT train?")
S2 = 0.01

x1 = torch.linspace(-0.5, 0.5, 9).requires_grad_()
fake_quant_naive(x1, S2).sum().backward()
print("      naive(round) backward -> x.grad =", [round(v, 1) for v in x1.grad.tolist()])

x2 = torch.linspace(-0.5, 0.5, 9).requires_grad_()
fake_quant_ste(x2, S2).sum().backward()
print("      STE(detach)  backward -> x.grad =", [round(v, 1) for v in x2.grad.tolist()])
print("      naive: all-zero gradients -> weights never update -> cannot train")
print("      STE  : gradients pass through as if the quantizer did not exist")
print("      -> that is the trick that makes QAT possible")

# ---------- 3.3 回到你的 QAT 日志 ----------
print("\n[3.3] Now re-read your QAT log:")
print("      - QAT training loss converged to ~5e-7")
print("        -> the network adapted to the 'staircase world'")
print("      - fake-quant model vs float  = 1.77e-3   (adaptation worked)")
print("      - REAL int8 vs float         = 3.72e-2   (reality is harsher!)")
print("      Open question: where does this fake-vs-real gap come from?")
print("      Candidates: extra requantize steps between ops /")
print("                  different int8 kernels and rounding /")
print("                  input & output scales that training never sees.")
print("      -> verifying on the REAL inference chain (finally: on-chip) is mandatory.")

print()
print("DONE. Now answer the check questions in the chat.")
