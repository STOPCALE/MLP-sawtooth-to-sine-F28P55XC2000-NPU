# -*- coding: utf-8 -*-
"""
实验 01: 用 MLP 学习 "锯齿波 -> 正弦波" 的逐点映射

任务定义 (取一个周期, t 属于 [0, 1)):
    锯齿波 (网络输入): s(t) = 2t - 1          # 从 -1 线性升到 +1
    正弦波 (训练标签): y(t) = sin(2*pi*t)     # 与锯齿波同周期、同相位
    网络要学的映射:    s  ->  y               # 单值函数: y = -sin(pi*s)

运行 (两种方式任选):
    1) VS Code 里直接按运行按钮
    2) 命令行:  E:\\anaconda\\envs\\torch\\python.exe saw2sin_mlp.py

输出:
    1) 控制台: 训练过程中的 loss 变化
    2) saw2sin_result.png: 损失曲线 + 波形对比 + 输入输出映射对比
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------- 0. 超参数 -----------------
SEED = 0            # 随机种子, 保证每次运行结果可复现
HIDDEN = 64         # 隐藏层宽度 (每层神经元数)
N_LAYERS = 2        # 隐藏层数量
BATCH = 256         # 每次迭代随机采样的样本数
ITERS = 16000        # 训练迭代次数
LR = 1e-3           # 学习率
FREQ = 1.0          # 正弦波频率 (含几个完整周期); 改成 2.0 / 4.0 观察高频更难学

torch.manual_seed(SEED)


# ----------------- 1. 数据发生器 -----------------
# 不预先构建数据集, 而是每次迭代现场随机采一批 (s, y) 数据对.
# 这是"无限数据"的思想: 数据由公式生成, 要多少有多少.
def sample_batch(n):
    t = torch.rand(n, 1)                        # t ~ U[0, 1) 的随机时刻
    s = 2.0 * t - 1.0                           # 锯齿波瞬时值 (输入)
    y = torch.sin(2.0 * np.pi * FREQ * t)       # 正弦波瞬时值 (标签)
    return s, y


# ----------------- 2. 模型: 多层感知机 MLP -----------------
class MLP(nn.Module):   #继承自nn.Module
    def __init__(self, hidden=HIDDEN, n_layers=N_LAYERS):   #构造函数的默认参数
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))   # 全连接层
            layers.append(nn.ReLU())                   # 激活函数 (可换 ReLU / Tanh 做对比)
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))            # 输出层: 1 个标量
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------- 3. 训练 -----------------
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # model.parameters() 自动收集所有层的权重（训练器靠它更新参数）
loss_fn = nn.MSELoss()                                    # 均方误差损失

print("training start ...")
loss_log = []
for it in range(1, ITERS + 1):
    s, y = sample_batch(BATCH)   # 1) 采一批数据
    pred = model(s)              # 2) 前向传播: 网络预测
    loss = loss_fn(pred, y)      # 3) 计算损失
    optimizer.zero_grad()        # 4) 清空上一轮残留的梯度
    loss.backward()              # 5) 反向传播: 自动求导
    optimizer.step()             # 6) 更新网络参数
    loss_log.append(loss.item())
    if it % 500 == 0:
        print(f"iter {it:5d} | loss = {loss.item():.3e}")

# ----------------- 4. 评估与可视化 -----------------
model.eval()                     # 推理模式 (本实验无 BatchNorm/Dropout, 属于好习惯)
with torch.no_grad():            # 推理不需要梯度, 省内存
    t = torch.linspace(0, 1, 1000).unsqueeze(1)   # [0,1) 上的均匀密网格
    s = 2.0 * t - 1.0
    y_true = torch.sin(2.0 * np.pi * FREQ * t)
    y_pred = model(s)

err = (y_pred - y_true).abs().max().item()
print(f"max abs error on grid: {err:.3e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) 训练损失曲线
axes[0].plot(loss_log)
axes[0].set_yscale("log")
axes[0].set_title("Train loss (MSE)")
axes[0].set_xlabel("iteration")

# (b) 波形对比: 锯齿波(输入) vs 正弦波(目标) vs MLP 输出
axes[1].plot(t[:, 0], s[:, 0], alpha=0.35, label="saw (input)")
axes[1].plot(t[:, 0], y_true[:, 0], lw=2, label="sin (target)")
axes[1].plot(t[:, 0], y_pred[:, 0], "--", lw=2, label="MLP output")
axes[1].set_title("waveforms vs time")
axes[1].legend()

# (c) 映射对比: 输入 s -> 输出 y
axes[2].plot(s[:, 0], y_true[:, 0], lw=2, label="true map")
axes[2].plot(s[:, 0], y_pred[:, 0], "--", lw=2, label="MLP map")
axes[2].set_title("map: saw value -> sin value")
axes[2].set_xlabel("saw value s")
axes[2].legend()

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saw2sin_result.png")
plt.savefig(fig_path, dpi=120)
print("figure saved:", fig_path)
plt.show()
