# -*- coding: utf-8 -*-
"""
部署模拟实验: 训练 -> 提取权重 -> "单片机侧"逐点推理 -> 频率扫描

场景: 输入锯齿波频率 100-500 Hz 不固定, 验证输出的正弦是否正确.
要点: 网络是"逐点波形整形器"(输入瞬时值 -> 输出瞬时值),
      映射与频率无关; 真正限制输出波形质量的是"每周期采样/推理点数".

输出:
  mlp_weights.h  - 导出的 C 语言权重数组 (单片机可直接使用, 注释为英文防乱码)
  deploy_sim.png - 频率扫描结果图
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
HIDDEN = 64
N_LAYERS = 2


# ---------- 1. 训练: FREQ=1, 学成 1:1 波形整形器 ----------
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=HIDDEN, n_layers=N_LAYERS):
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


def sample_batch(n):
    t = torch.rand(n, 1)
    s = 2.0 * t - 1.0
    y = torch.sin(2.0 * np.pi * t)
    return s, y


model = MLP(1, HIDDEN, N_LAYERS)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.MSELoss()
for it in range(4000):
    s, y = sample_batch(256)
    loss = lossf(model(s), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
print("trained (freq=1), final loss = %.3e" % loss.item())

# ---------- 2. 提取权重 -> C 数组文件 (模型的"内容") ----------
sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
W1 = sd["net.0.weight"].astype(np.float32).reshape(-1)
b1 = sd["net.0.bias"].astype(np.float32).reshape(-1)
W2 = sd["net.2.weight"].astype(np.float32).reshape(-1)
b2 = sd["net.2.bias"].astype(np.float32).reshape(-1)
W3 = sd["net.4.weight"].astype(np.float32).reshape(-1)
b3 = sd["net.4.bias"].astype(np.float32).reshape(-1)

h_path = os.path.join(OUT_DIR, "mlp_weights.h")
with open(h_path, "w", encoding="utf-8") as fo:
    fo.write("#ifndef MLP_WEIGHTS_H\n#define MLP_WEIGHTS_H\n")
    fo.write("/* Exported from PyTorch. Network: 1 -> 64 -> 64 -> 1, SiLU.\n")
    fo.write("   Task: sawtooth(instantaneous value) -> sine value (1:1 waveshaper).\n")
    fo.write("   Input s in [-1, 1] (normalized saw value), output y.\n")
    fo.write("   Inference per sample:\n")
    fo.write("     h1[i] = silu(W1[i]*s + b1[i]);\n")
    fo.write("     h2[i] = silu(sum_j W2[i*64+j]*h1[j] + b2[i]);\n")
    fo.write("     y     = sum_j W3[j]*h2[j] + b3[0];   silu(x) = x / (1 + expf(-x))\n")
    fo.write("   W2 is row-major: W2[i*64+j], i = output unit, j = input unit. */\n\n")

    def write_arr(name, vals):
        fo.write("static const float %s[%d] = {\n" % (name, len(vals)))
        for i in range(0, len(vals), 6):
            chunk = ", ".join("%.9gf" % float(v) for v in vals[i:i + 6])
            fo.write("    " + chunk + ",\n")
        fo.write("};\n\n")

    write_arr("MLP_W1", W1)
    write_arr("MLP_B1", b1)
    write_arr("MLP_W2", W2)
    write_arr("MLP_B2", b2)
    write_arr("MLP_W3", W3)
    write_arr("MLP_B3", b3)
    fo.write("#endif\n")
print("saved:", h_path)

# ---------- 3. "单片机侧"推理函数 (纯 numpy, 不依赖 PyTorch) ----------
W2m = W2.reshape(HIDDEN, HIDDEN)


def silu(x):
    return x / (1.0 + np.exp(-x))


def infer_one(s):
    # 模拟单片机: 每来一个采样点, 做一次前向计算
    h1 = silu(W1 * s + b1)
    h2 = silu(W2m @ h1 + b2)
    return float(W3 @ h2 + b3[0])


# ---------- 4. 一致性验证: numpy 推理 vs PyTorch ----------
with torch.no_grad():
    s_chk = torch.linspace(-1, 1, 1000).unsqueeze(1)
    y_torch = model(s_chk).numpy().ravel()
y_np = np.array([infer_one(float(x)) for x in np.linspace(-1, 1, 1000)], dtype=np.float32)
print("consistency check |numpy - pytorch| max = %.2e" % np.abs(y_np - y_torch).max())

# ---------- 5. 频率扫描: 模拟不同频率的现场输入 ----------
fs = 25000.0   # 假设: 采样+推理速度 25 kHz  (500 Hz x 50 点/周期)
cases = [
    (100.0, fs),
    (300.0, fs),
    (500.0, fs),
    (500.0, 6000.0),   # 对比: 推理/采样速度不足时 (每周期仅 12 点)
]
fig, axes = plt.subplots(len(cases), 1, figsize=(11, 2.2 * len(cases)))
for ax, (f, fs_c) in zip(axes, cases):
    n_cycle = 4
    t = np.arange(0.0, n_cycle / f, 1.0 / fs_c)
    phase = np.mod(f * t, 1.0)
    s_in = 2.0 * phase - 1.0
    y_mc = np.array([infer_one(v) for v in s_in], dtype=np.float32)
    y_true = np.sin(2.0 * np.pi * f * t)
    err = np.abs(y_mc - y_true).max()
    ax.plot(t * 1e3, y_true, lw=2, alpha=0.65, label="expected sin")
    ax.plot(t * 1e3, y_mc, "o--", ms=3, lw=1.2, label="MCU output (pointwise)")
    ax.set_title("input %.0f Hz | fs=%.0f kHz | %.0f samples/period | max err = %.1e"
                 % (f, fs_c / 1e3, fs_c / f, err))
    ax.set_xlabel("time (ms)")
    ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "deploy_sim.png"), dpi=120)
print("saved: deploy_sim.png")
plt.show()
