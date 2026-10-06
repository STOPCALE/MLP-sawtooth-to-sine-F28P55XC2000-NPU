# -*- coding: utf-8 -*-
"""
[saw2sin -> TI-NPU] Stage A (卷积版): TI-QAT 重量化 -> 导出 NPU 可用的 int8 ONNX

为什么用 1x1 卷积表达 (实测结论):
  - TI NPU 编译器只把 "Gemm" 形态的全连接认作 FC 模式, 卸载到 NPU;
    PyTorch 导出的逐点 MLP 呈 "MatMul + 独立 Add" 形态 (3/4 维张量无法被
    onnxruntime 融合成 Gemm) -> FC 全部回落 CPU (编译报告实测).
  - 而 Conv2d(1->64, 1x1) 与 Linear(1->64) 数学完全等价 (逐点线性映射):
        y[o,h,w] = sum_c W[o,c,0,0]*x[c,h,w] + b[o]
    卷积形态正好匹配 TI NPU 的 FCONV/GCONV 支持路径 (官方模型全是这种结构).
  - 权重完全复用 saw2sin_model.pt (Linear 权重 reshape 成 1x1 卷积核).

输入/输出约定 (TI 时序流):
  (1, 1, frame, 1)  ->  (1, 1, frame, 1)
  NHWC 直觉: 一次推理 = frame 个连续采样点 -> frame 个正弦值 (逐点独立映射).

运行:
  E:\\anaconda\\envs\\ti-npu\\python.exe E:\\desk\\ti-npu\\saw2sin_npu\\case_saw2sin\\saw2sin_tinpu_qat_conv.py [frame]
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# ----------------- 0. 路径与超参数 -----------------
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
CHARGE_DIR = r"E:\desk\DeepStudy\Charge_Num"
FLOAT_CKPT = os.path.join(CHARGE_DIR, "saw2sin_model.pt")
OUT_ONNX = os.path.join(WORK_DIR, "saw2sin_tinpu_int8.onnx")

HIDDEN = 64
EPOCHS = 15
ITERS_PER_EPOCH = 200
BATCH = 256
LR = 2e-4
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 10

SEED = 0
torch.manual_seed(SEED)
print(f"[info ] FRAME = {FRAME}")


# ----------------- 1. 模型 -----------------
class MLP(nn.Module):
    """原始结构 (只用于读 saw2sin_model.pt 的 state_dict 键名, 与 Charge_Num 一致)"""
    def __init__(self, in_dim=1, hidden=HIDDEN, n_layers=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvMLP(nn.Module):
    """与 MLP(1->64->64->1) 数学等价的 1x1 卷积网络"""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1), nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)          # x: (B, 1, frame, 1)


# ----------------- 2. 数据与工具 -----------------
def sample_batch(n):
    """随机窗口: (n, 1, frame, 1) 的锯齿输入 -> 正弦目标"""
    t = torch.rand(n, FRAME, 1)
    s = 2.0 * t - 1.0
    y = torch.sin(2.0 * np.pi * t)
    return s.permute(0, 2, 1).unsqueeze(3).contiguous(), y.permute(0, 2, 1).unsqueeze(3).contiguous()


def window_pack(s):
    """(N,1) 逐点序列 -> (N//frame, 1, frame, 1) 窗口"""
    n = (s.shape[0] // FRAME) * FRAME
    return s[:n].reshape(-1, FRAME, 1).permute(0, 2, 1).unsqueeze(3).contiguous()


def grid_eval(fn):
    s = torch.linspace(-1, 1, 2000).unsqueeze(1)
    y_true = torch.sin(np.pi * (s + 1.0))
    with torch.no_grad():
        y_pred = fn(window_pack(s)).permute(0, 2, 1, 3).reshape(-1, 1)
    err = (y_pred - y_true[:y_pred.shape[0]]).abs()
    return err.max().item(), err.mean().item()


# ----------------- 3. 载入 float 权重 (Linear -> 1x1 Conv) -----------------
mlp = MLP()
mlp.load_state_dict(torch.load(FLOAT_CKPT, map_location="cpu"))
mlp.eval()

convnet = ConvMLP()
with torch.no_grad():
    convnet.net[0].weight.copy_(mlp.net[0].weight.view(HIDDEN, 1, 1, 1))
    convnet.net[0].bias.copy_(mlp.net[0].bias)
    convnet.net[2].weight.copy_(mlp.net[2].weight.view(HIDDEN, HIDDEN, 1, 1))
    convnet.net[2].bias.copy_(mlp.net[2].bias)
    convnet.net[4].weight.copy_(mlp.net[4].weight.view(1, HIDDEN, 1, 1))
    convnet.net[4].bias.copy_(mlp.net[4].bias)
convnet.eval()

# 等价性验证 (float 层面应该几乎完全一致)
s_grid = torch.linspace(-1, 1, 2000).unsqueeze(1)
with torch.no_grad():
    y_mlp = mlp(s_grid).reshape(-1)
    y_conv = convnet(window_pack(s_grid)).permute(0, 2, 1, 3).reshape(-1)
equiv = (y_mlp - y_conv).abs().max().item()
print(f"[equiv] MLP vs ConvMLP max diff = {equiv:.3e}")
assert equiv < 1e-5, "conv 转换不等价, 请检查权重对应!"

err_max_f, err_mean_f = grid_eval(convnet)
print(f"[float ] max err = {err_max_f:.3e}   mean err = {err_mean_f:.3e}")

# ----------------- 4. TINPUT QAT 包装 -----------------
from tinyml_torchmodelopt.quantization import TinyMLQConfigType, TINPUTinyMLQATFxModule

example_inputs = torch.rand(1, 1, FRAME, 1)           # (1, 1, frame, 1): batch=1 静态
qconfig_type = TinyMLQConfigType(weight_bitwidth=8, activation_bitwidth=8,
                                 auto_quantization=False)
qat = TINPUTinyMLQATFxModule(convnet, example_inputs=example_inputs,
                             qconfig_type=qconfig_type,
                             total_epochs=EPOCHS,
                             output_int=False)

# ----------------- 5. QAT 微调 -----------------
optimizer = torch.optim.Adam(qat.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = nn.MSELoss()

print("QAT fine-tuning ...")
for epoch in range(EPOCHS):
    qat.train()                                       # 每 epoch 只调一次 (推进温度/冻结调度)
    for _ in range(ITERS_PER_EPOCH):
        s, y = sample_batch(BATCH)
        loss = loss_fn(qat(s), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    qat.eval()
    err_max, err_mean = grid_eval(qat)
    print(f"epoch {epoch + 1:2d}/{EPOCHS} | loss {loss.item():.3e} | fake-quant max err {err_max:.3e}")

# ----------------- 6. convert -> export -----------------
qat.eval()
qat = qat.convert()
# opset 18 = 官方默认 (train_base.py --opset-version default=18)
qat.export(torch.rand(1, 1, FRAME, 1), OUT_ONNX, opset_version=18,
           input_names=["input"], output_names=["output"])
print("exported:", OUT_ONNX)

# ----------------- 7. 验收: onnxruntime 推理导出模型 -----------------
import onnx
import onnxruntime as ort

onnx_model = onnx.load(OUT_ONNX)
in_names = [i.name for i in onnx_model.graph.input]
for vi in list(onnx_model.graph.input) + list(onnx_model.graph.output):
    dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in vi.type.tensor_type.shape.dim]
    print(f"onnx {'input ' if vi.name in in_names else 'output'}: {vi.name} shape={dims}")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess = ort.InferenceSession(OUT_ONNX, so, providers=["CPUExecutionProvider"])

s = np.linspace(-1, 1, 2000).astype(np.float32).reshape(-1, 1)
y_true = np.sin(np.pi * (s + 1.0))
outs = []
for i in range(0, 2000 - FRAME + 1, FRAME):
    win = s[i:i + FRAME].reshape(1, 1, FRAME, 1)
    out = sess.run(None, {"input": win})[0]
    outs.append(np.asarray(out).ravel())
y_int8 = np.concatenate(outs)
err = np.abs(y_int8 - y_true.ravel()[0:y_int8.size])
print(f"[int8  ] max err = {err.max():.3e}   mean err = {err.mean():.3e}")

print()
print("done. next: 用 saw2sin_compile.yaml 编译到 F28P55 (Stage B)")
