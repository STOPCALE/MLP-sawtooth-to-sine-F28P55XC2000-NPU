# -*- coding: utf-8 -*-
"""
部署路线 A · 第 3 步: 量化感知训练 QAT (让网络"适应" int8)

原理:
  prepare_qat 把每个 Linear 层换成"伪量化"版本:
  前向时模拟 int8 的舍入误差 (round + clamp), 反向时梯度直通 (STE).
  于是网络在训练中就能"看到"量化误差并学会补偿它.

流程:
  1. 加载 float 权重 saw2sin_model.pt
  2. prepare_qat -> 微调 2000 步 -> convert_fx (冻结 scale, 变真 int8 模型)
  3. 三方对比: float vs PTQ vs QAT -> quant_compare.png
  4. 尝试导出 QAT int8 ONNX; 保存量化权重 saw2sin_qat_state.pt (手写推理阶段要用)

运行: conda run --no-capture-output -n torch python qat_train.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.observer import (MovingAverageMinMaxObserver,
                                            MovingAveragePerChannelMinMaxObserver)
import onnxruntime as ort
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
HIDDEN = 64
N_LAYERS = 2

print("supported quantized engines:", torch.backends.quantized.supported_engines)
if "fbgemm" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "fbgemm"


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


# ---------- 0. 加载 float 模型 ----------
model = MLP()
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "saw2sin_model.pt")))
model.eval()

s_grid = torch.linspace(-1, 1, 1000).unsqueeze(1)  # 评估用网格
with torch.no_grad():
    y_float = model(s_grid).numpy().ravel()

# ---------- 1. prepare_qat (插入伪量化节点) ----------
# qconfig = "量化方案说明书": 这里显式使用 全8bit (quant_min/quant_max),
# 避开默认 fbgemm qconfig 的 reduce_range (只用到 7bit, 精度损失大)
act_obs = MovingAverageMinMaxObserver.with_args(
    dtype=torch.quint8, qscheme=torch.per_tensor_affine,
    quant_min=0, quant_max=255)
wt_obs = MovingAveragePerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
    quant_min=-128, quant_max=127)
qconfig = QConfig(activation=act_obs, weight=wt_obs)
model.train()
qconfig_mapping = QConfigMapping().set_global(qconfig)
example_inputs = (torch.rand(8, 1),)
model_prepared = quantize_fx.prepare_qat_fx(model, qconfig_mapping,
                                            example_inputs=example_inputs)
print("QAT prepared, fine-tuning ...")

# ---------- 2. 微调 (让网络适应量化误差; lr 逐步衰减到小值) ----------
opt = torch.optim.Adam(model_prepared.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.1)
lossf = nn.MSELoss()
for it in range(1, 4001):
    s, y = sample_batch(256)
    loss = lossf(model_prepared(s), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()
    if it % 500 == 0:
        print("QAT iter %4d | lr = %.1e | loss = %.3e"
              % (it, opt.param_groups[0]["lr"], loss.item()))

# ---------- 3. convert -> 真 int8 模型 ----------
# 冻结观察器, 让最后的 scale 定稿 (遇到不支持的环境就跳过)
try:
    from torch.ao.quantization import disable_observer
    model_prepared.apply(disable_observer)
    print("observers disabled (scale frozen)")
except Exception as e:
    print("disable_observer skipped:", repr(e))
model_prepared.eval()
with torch.no_grad():
    y_fake = model_prepared(s_grid).numpy().ravel()   # 伪量化模型输出 (训练时"看到"的)
qmodel = quantize_fx.convert_fx(model_prepared)
q_path = os.path.join(OUT_DIR, "saw2sin_qat_state.pt")
torch.save(qmodel.state_dict(), q_path)
print("saved:", q_path)

qmodel.eval()
with torch.no_grad():
    y_qat_pt = qmodel(s_grid).numpy().ravel()

print("fake-quant vs real-int8 max diff = %.2e" % np.abs(y_fake - y_qat_pt).max())
print("fake-quant vs float     max diff = %.2e" % np.abs(y_fake - y_float).max())

# ---------- 4. 尝试导出 QAT int8 ONNX ----------
qat_onnx = os.path.join(OUT_DIR, "saw2sin_qat_int8.onnx")
y_qat = y_qat_pt   # 默认用 PyTorch 量化模型输出; onnx 导出成功则用 onnxruntime 输出
try:
    torch.onnx.export(
        qmodel, torch.zeros(2, 1), qat_onnx,
        input_names=["s"], output_names=["y"],
        dynamic_axes={"s": {0: "batch"}, "y": {0: "batch"}},
        opset_version=17,
    )
    sess = ort.InferenceSession(qat_onnx, providers=["CPUExecutionProvider"])
    y_qat_onnx = sess.run(None, {"s": s_grid.numpy()})[0].ravel()
    print("qat onnx ok, onnxruntime vs pytorch-int8 max diff = %.2e"
          % np.abs(y_qat_onnx - y_qat_pt).max())
    y_qat = y_qat_onnx
except Exception as e:
    print("QAT ONNX export failed:", repr(e))

# ---------- 5. 三方对比 ----------
s_np = s_grid.numpy()
sess_q = ort.InferenceSession(os.path.join(OUT_DIR, "saw2sin_int8.onnx"),
                              providers=["CPUExecutionProvider"])
y_ptq = sess_q.run(None, {"s": s_np})[0].ravel()
y_true = -np.sin(np.pi * s_np.ravel())

print("float: vs target max = %.2e" % np.abs(y_float - y_true).max())
print("PTQ  : vs target max = %.2e | vs float max = %.2e mean = %.2e"
      % (np.abs(y_ptq - y_true).max(), np.abs(y_ptq - y_float).max(),
         np.abs(y_ptq - y_float).mean()))
print("QAT  : vs target max = %.2e | vs float max = %.2e mean = %.2e"
      % (np.abs(y_qat - y_true).max(), np.abs(y_qat - y_float).max(),
         np.abs(y_qat - y_float).mean()))

sx = s_np.ravel()
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(sx, y_true, lw=2, label="target (-sin(pi s))")
axes[0].plot(sx, y_float, "--", lw=1.5, label="float")
axes[0].plot(sx, y_ptq, "--", lw=1.5, label="int8 PTQ")
axes[0].plot(sx, y_qat, "--", lw=1.5, label="int8 QAT")
axes[0].legend()
axes[0].set_title("output waveform")

axes[1].semilogy(sx, np.abs(y_float - y_true) + 1e-9, label="float vs target")
axes[1].semilogy(sx, np.abs(y_ptq - y_true) + 1e-9, label="PTQ vs target")
axes[1].semilogy(sx, np.abs(y_qat - y_true) + 1e-9, label="QAT vs target")
axes[1].legend()
axes[1].set_title("absolute error (log scale)")
axes[1].set_xlabel("saw value s")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "quant_compare.png"), dpi=120)
print("saved: quant_compare.png")
plt.show()
