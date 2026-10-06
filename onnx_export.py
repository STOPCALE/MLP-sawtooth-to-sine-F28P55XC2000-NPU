# -*- coding: utf-8 -*-
"""
部署路线 A · 第 1 步: 导出 ONNX (float32)

流程:
  1. 训练 float 模型 (与例程一致, 可复现)
  2. 保存权重 saw2sin_model.pt (给后面的 QAT 微调用)
  3. 导出 saw2sin.onnx (batch 维动态: 单点 N=1 和批量 N=1000 都支持)
  4. onnxruntime 数值验证: ONNX 输出必须与 PyTorch 一致 (导出必须无损!)
  5. 打印 ONNX 计算图信息 (看看这个"模型文件"里到底装了什么)

运行: conda run --no-capture-output -n torch python onnx_export.py
"""

import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

SEED = 0
torch.manual_seed(SEED)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
HIDDEN = 64
N_LAYERS = 2


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


# ---------- 1. 训练 float 模型 ----------
model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.MSELoss()
for it in range(4000):
    s, y = sample_batch(256)
    loss = lossf(model(s), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
print("trained, final loss = %.3e" % loss.item())

model.eval()
pt_path = os.path.join(OUT_DIR, "saw2sin_model.pt")
torch.save(model.state_dict(), pt_path)
print("saved:", pt_path)

# ---------- 2. 导出 ONNX ----------
onnx_path = os.path.join(OUT_DIR, "saw2sin.onnx")
torch.onnx.export(
    model,
    torch.zeros(2, 1),                    # dummy input, 形状 (batch, 1)
    onnx_path,
    input_names=["s"],                    # 给输入命名: 部署时官方 API 要按名字喂数据
    output_names=["y"],
    dynamic_axes={"s": {0: "batch"},      # batch 维可变: N=1 单点 / N=1000 批量都行
                  "y": {0: "batch"}},
    opset_version=17,                     # 算子集版本: 给工具链用的"语言版本"
)
print("saved:", onnx_path)

# ---------- 3. 数值验证 (导出必须无损) ----------
import onnxruntime as ort

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
s_np = np.linspace(-1, 1, 1000, dtype=np.float32).reshape(-1, 1)
with torch.no_grad():
    y_torch = model(torch.from_numpy(s_np)).numpy()
y_onnx = sess.run(None, {"s": s_np})[0]
print("onnx vs pytorch, max diff = %.2e" % np.abs(y_torch - y_onnx).max())

# 部署形态验证: 单点推理 (batch=1, 单片机未来就是这种调用方式)
single = np.array([[-0.5]], dtype=np.float32)
y_single = sess.run(None, {"s": single})[0]
print("single-point test: s=-0.5 -> y=%.4f (expect ~+1.0)" % y_single[0, 0])

# ---------- 4. 看 ONNX 计算图 ----------
import onnx

m = onnx.load(onnx_path)
ops = Counter(n.op_type for n in m.graph.node)
print("graph operators:", dict(ops))
print("graph inputs :", [i.name for i in m.graph.input])
print("graph outputs:", [o.name for o in m.graph.output])
