# -*- coding: utf-8 -*-
"""
验证实验: 网络能否应对"频率不同"的输入?

两部分:
  Part A (相位映射的频率不变性):
      在 FREQ=1 上训练的网络, 直接用于 1/3/7.5/20 倍频率的锯齿波输入.
      预期: 全部正确 —— 因为映射 s -> y 与频率无关.

  Part B (把频率作为输入, 条件生成):
      训练时频率随机采样 U[1, 8], 网络输入变成 2 维 (s, f).
      预期: 训练范围内的频率全部学对; 训练范围外 (f=12) 出现外推退化.

运行注意: 用 conda run 或先激活 torch 环境 (见 saw2sin_mlp.py 顶部说明).
输出: freq_invariance_partA.png / cond_freq_partB.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=64, n_layers=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==================== Part A: 相位映射的频率不变性 ====================
# 训练: 与例程完全一致 (FREQ=1, 输入只有 s, 1 维)
def sample_batch_A(n):
    t = torch.rand(n, 1)
    s = 2.0 * t - 1.0
    y = torch.sin(2.0 * np.pi * t)
    return s, y


modelA = MLP(1, 64, 2)
optA = torch.optim.Adam(modelA.parameters(), lr=1e-3)
lossf = nn.MSELoss()
for it in range(4000):
    s, y = sample_batch_A(256)
    loss = lossf(modelA(s), y)
    optA.zero_grad()
    loss.backward()
    optA.step()
print("Part A trained (freq=1 only), final loss = %.3e" % loss.item())

# 测试: 同一个网络, 换成不同频率的锯齿波输入 (每行统一显示 5 个周期)
modelA.eval()
freqs = [1.0, 3.0, 7.5, 20.0]
fig, axes = plt.subplots(len(freqs), 1, figsize=(11, 2.1 * len(freqs)))
with torch.no_grad():
    for ax, f in zip(axes, freqs):
        t = torch.linspace(0, 5.0 / f, 4000).unsqueeze(1)
        phase = torch.frac(f * t)            # 周期内相位 [0,1)
        s = 2.0 * phase - 1.0                # 锯齿波瞬时值 (网络输入)
        y_true = torch.sin(2.0 * np.pi * phase)
        y_pred = modelA(s)
        err = (y_pred - y_true).abs().max().item()
        ax.plot(t[:, 0], s[:, 0], alpha=0.25, label="saw input")
        ax.plot(t[:, 0], y_true[:, 0], lw=2, label="target sin")
        ax.plot(t[:, 0], y_pred[:, 0], "--", lw=2, label="MLP output")
        ax.set_title("Part A: net trained at freq=1 ONLY, now input freq = %.1f | max err = %.1e"
                     % (f, err))
        ax.legend(loc="upper right", fontsize=8, ncol=3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "freq_invariance_partA.png"), dpi=120)
print("saved: freq_invariance_partA.png")

# ==================== Part B: 把频率作为第二个输入 ====================
# 训练: 频率随机 U[1, 8], 输入 2 维 (s, f)
F_MIN, F_MAX = 1.0, 8.0


def sample_batch_B(n):
    t = torch.rand(n, 1)
    f = F_MIN + (F_MAX - F_MIN) * torch.rand(n, 1)   # 随机频率
    s = 2.0 * t - 1.0
    y = torch.sin(2.0 * np.pi * f * t)
    x = torch.cat([s, f / F_MAX], dim=1)              # 输入: (s, f 归一化到 [0.125, 1])
    return x, y


torch.manual_seed(SEED)
modelB = MLP(2, 128, 3)
optB = torch.optim.Adam(modelB.parameters(), lr=1e-3)
for it in range(1, 30001):
    x, y = sample_batch_B(256)
    loss = lossf(modelB(x), y)
    optB.zero_grad()
    loss.backward()
    optB.step()
    if it % 2500 == 0:
        print("Part B iter %6d | loss = %.3e" % (it, loss.item()))

# 测试: f=1..8 (训练范围内) + f=12 (训练范围外)
modelB.eval()
test_freqs = [1.0, 3.0, 5.5, 8.0, 12.0]
fig, axes = plt.subplots(len(test_freqs), 1, figsize=(11, 2.1 * len(test_freqs)))
with torch.no_grad():
    for ax, f in zip(axes, test_freqs):
        t = torch.linspace(0, 3.0, 3000).unsqueeze(1)     # 3 个归一化周期
        s = 2.0 * torch.frac(t) - 1.0                     # 锯齿瞬时值 (始终在 [-1,1])
        fv = torch.full_like(s, f / F_MAX)                # 频率列 (同样要归一化!)
        x = torch.cat([s, fv], dim=1)
        y_true = torch.sin(2.0 * np.pi * f * t)
        y_pred = modelB(x)
        err = (y_pred - y_true).abs().max().item()
        tag = "in range" if f <= F_MAX else "OUT of range!"
        ax.plot(t[:, 0], y_true[:, 0], lw=2, label="target sin")
        ax.plot(t[:, 0], y_pred[:, 0], "--", lw=2, label="MLP output")
        ax.set_title("Part B: conditioned net (s, f), test freq = %.1f (%s) | max err = %.1e"
                     % (f, tag, err))
        ax.legend(loc="upper right", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cond_freq_partB.png"), dpi=120)
print("saved: cond_freq_partB.png")
plt.show()
