# -*- coding: utf-8 -*-
"""
[saw2sin -> TI-NPU] Stage A: TI-QAT 重量化 -> 导出 NPU 可用的 int8 ONNX

为什么旧文件不行:
  saw2sin.onnx (float) 只能 CPU;
  saw2sin_int8.onnx / saw2sin_qat_int8.onnx 是"通用 CPU 量化"(ORT QDQ / fbgemm),
  不符合 TI NPU 硬件约束 -> 都上不了 NPU.

本脚本做什么:
  1. 复现 MLP(1->64->64->1) 结构, 载入 saw2sin_model.pt 的 float 权重
  2. 用 TINPUTinyMLQATFxModule 包装 -> 注入 TI NPU 量化约束(per-channel 对称权重、
     per-tensor 对称激活、2 的幂 scale 等, 这是 NPU 硬件的硬性要求)
  3. QAT 微调: 前向模拟 int8 舍入误差(STE), 让网络"学会补偿"量化误差
  4. convert() 冻结所有 scale -> export() 导出 QDQ 形式 int8 ONNX (batch=1 静态)
  5. 用 onnxruntime 对导出结果做精度验收

易错点:
  - batch 必须 =1 (TI NPU 硬约束), 所以 example_inputs 用 torch.rand(1,1)
  - wrapper.train() 每次调用会推进"温度/冻结"调度(内部 num_epochs_tracked+1),
    所以每个 epoch 只调一次 train(), 不要在 epoch 内反复调
  - total_epochs 要和实际训练 epoch 数一致(温度调度表按它生成)

运行:
  E:\\anaconda\\envs\\ti-npu\\python.exe E:\\desk\\ti-npu\\saw2sin_npu\\case_saw2sin\\saw2sin_tinpu_qat.py
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
N_LAYERS = 2
EPOCHS = 15              # QAT 微调轮数
ITERS_PER_EPOCH = 200    # 每轮迭代数 (共 3000 步)
BATCH = 256
LR = 2e-4
# TI 时序流按 (1, frame, 1) 窗口处理输入; frame 可用命令行参数覆盖:
#   python saw2sin_tinpu_qat.py 1     <- 单点 (1,1,1)
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 10

SEED = 0
torch.manual_seed(SEED)
print(f"[info ] FRAME = {FRAME}")


# ----------------- 1. 模型与数据 (与 saw2sin_mlp.py 完全一致) -----------------
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
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


def grid_eval(fn, n=2001):
    """在 s 网格上评估网络输出与理想正弦的误差"""
    s = torch.linspace(-1, 1, n).unsqueeze(1)
    y_true = torch.sin(np.pi * (s + 1.0))     # s=2t-1 -> y=sin(2*pi*t)=sin(pi*(s+1))
    with torch.no_grad():
        y_pred = fn(s)
    err = (y_pred - y_true).abs()
    return err.max().item(), err.mean().item()


# ----------------- 2. 载入 float 权重并记录基准误差 -----------------
model = MLP()
state = torch.load(FLOAT_CKPT, map_location="cpu")
model.load_state_dict(state)
model.eval()
err_max_f, err_mean_f = grid_eval(model)
print(f"[float ] max err = {err_max_f:.3e}   mean err = {err_mean_f:.3e}")

# ----------------- 3. 用 TINPUT QAT 包装 -----------------
from tinyml_torchmodelopt.quantization import TinyMLQConfigType, TINPUTinyMLQATFxModule

example_inputs = torch.rand(1, FRAME, 1)              # (1, frame, 1): batch=1 静态, TI 时序流按窗口处理
qconfig_type = TinyMLQConfigType(weight_bitwidth=8, activation_bitwidth=8,
                                 auto_quantization=False)
qat = TINPUTinyMLQATFxModule(model, example_inputs=example_inputs,
                             qconfig_type=qconfig_type,
                             total_epochs=EPOCHS,
                             output_int=False)        # False = 输出反量化为 float (回归任务官方默认)

# ----------------- 4. QAT 微调 -----------------
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
    qat.eval()                                        # 伪量化(未 convert)模型的量化误差观察
    err_max, err_mean = grid_eval(qat)
    print(f"epoch {epoch + 1:2d}/{EPOCHS} | loss {loss.item():.3e} | fake-quant max err {err_max:.3e}")

# ----------------- 5. convert -> export -----------------
qat.eval()
qat = qat.convert()                                   # 冻结 scale, 转成整数运算图
# 重要: opset 必须与官方流程一致 (train_base.py 默认 18), 否则 ORT 融合结果不同,
# 全连接会以 MatMul+Add 形态出现, TI 编译器的 FC 模式匹配器不识别 -> 全落在 CPU
torch.manual_seed(SEED)                               # 重新固定随机种子 -> 导出输入确定
qat.export(torch.rand(1, FRAME, 1), OUT_ONNX, opset_version=18,
           input_names=["input"], output_names=["output"])
print("exported:", OUT_ONNX)

# ----------------- 6. 验收: onnxruntime 推理导出模型 -----------------
import onnx
import onnxruntime as ort

onnx_model = onnx.load(OUT_ONNX)
for vi in list(onnx_model.graph.input) + list(onnx_model.graph.output):
    dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in vi.type.tensor_type.shape.dim]
    print(f"onnx {('input ' if vi.name in [i.name for i in onnx_model.graph.input] else 'output')}: {vi.name} shape={dims}")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess = ort.InferenceSession(OUT_ONNX, so, providers=["CPUExecutionProvider"])

s = np.linspace(-1, 1, 2000).astype(np.float32).reshape(-1, 1)
y_true = np.sin(np.pi * (s + 1.0))
# 模型输入是 (1, frame, 1) 静态窗口, 按窗口喂入 (400 个点 = 40 个窗口)
outs = []
for i in range(0, 2000 - FRAME + 1, FRAME):
    out = sess.run(None, {"input": s[i:i + FRAME][None, :, :]})[0]
    outs.append(np.asarray(out).ravel())
y_int8 = np.concatenate(outs)
err = np.abs(y_int8 - y_true.ravel()[0:y_int8.size])
print(f"[int8  ] max err = {err.max():.3e}   mean err = {err.mean():.3e}")

print()
print("done. next: 用 saw2sin_compile.yaml 编译到 F28P55 (Stage B)")
