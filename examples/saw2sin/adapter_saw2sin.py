# -*- coding: utf-8 -*-
"""saw2sin 示例 adapter: 已训练 MLP(1->64->64->1) 的 1x1 卷积等价表达.

数据来源: E:\\desk\\DeepStudy\\Charge_Num\\saw2sin_model.pt (float 权重)
任务: 逐点锯齿波瞬时值 s ∈ [-1,1] -> 正弦值 sin(pi*(s+1))

为什么写成 1x1 卷积:
  PyTorch 导出的全连接是 MatMul+独立 Add 形态, TI 编译器只识别 Gemm 形态
  -> FC 会全部回落 CPU; 而 Linear 与 Conv2d(k=1x1) 数学等价, 卷积形态能走
  FCONV/GCONV/PWCONV 的 NPU 路径。(详见 docs/saw2sin_case.md §3)
"""
import numpy as np
import torch
import torch.nn as nn

FLOAT_CKPT = r"E:\desk\DeepStudy\Charge_Num\saw2sin_model.pt"
HIDDEN = 64
FRAME = 10                     # 窗口长度 (部署静态形状 (1,1,FRAME,1))


class _MLP(nn.Module):
    """原始结构 (只用于读 checkpoint 的 state_dict 键名)."""

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


class _ConvMLP(nn.Module):
    """与 MLP(1->64->64->1) 数学等价的 1x1 卷积网络."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1), nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)          # x: (B, 1, FRAME, 1)


def _window_pack(s):
    """(N,1) 逐点序列 -> (N//FRAME, 1, FRAME, 1) 窗口."""
    n = (s.shape[0] // FRAME) * FRAME
    return s[:n].reshape(-1, FRAME, 1).permute(0, 2, 1).unsqueeze(3).contiguous()

#这里开始要加入

def get_model():
    mlp = _MLP()
    mlp.load_state_dict(torch.load(FLOAT_CKPT, map_location="cpu"))
    mlp.eval()

    conv = _ConvMLP()
    with torch.no_grad():
        conv.net[0].weight.copy_(mlp.net[0].weight.view(HIDDEN, 1, 1, 1))
        conv.net[0].bias.copy_(mlp.net[0].bias)
        conv.net[2].weight.copy_(mlp.net[2].weight.view(HIDDEN, HIDDEN, 1, 1))
        conv.net[2].bias.copy_(mlp.net[2].bias)
        conv.net[4].weight.copy_(mlp.net[4].weight.view(1, HIDDEN, 1, 1))
        conv.net[4].bias.copy_(mlp.net[4].bias)
    conv.eval()

    # 等价性自检 (float 层面): 不等价直接报错, 绝不带病导出
    s = torch.linspace(-1, 1, 991).unsqueeze(1)
    n = (s.shape[0] // FRAME) * FRAME
    with torch.no_grad():
        y_mlp = mlp(s[:n]).reshape(-1)
        y_conv = conv(_window_pack(s)).permute(0, 2, 1, 3).reshape(-1)
    diff = (y_mlp - y_conv).abs().max().item()
    assert diff < 1e-5, f"Linear->Conv(1x1) 权重映射不等价! max diff = {diff}"
    return conv


def get_example_input():
    return torch.rand(1, 1, FRAME, 1)




def get_eval_inputs(num_samples):
    """验收输入: s 网格打包成窗口 -> (num_windows, 1, FRAME, 1)."""
    n = num_samples * FRAME
    s = torch.linspace(-1, 1, n).unsqueeze(1)
    return _window_pack(s)


def evaluate(predict):
    """网格验收: predict(np (1,1,FRAME,1)) -> np; 返回相对理论正弦的误差."""
    s = np.linspace(-1, 1, 2000).astype(np.float32).reshape(-1, 1)
    y_true = np.sin(np.pi * (s + 1.0)).ravel()
    outs = []
    for i in range(0, 2000 - FRAME + 1, FRAME):
        win = s[i:i + FRAME].reshape(1, 1, FRAME, 1)
        outs.append(np.asarray(predict(win)).ravel())
    y = np.concatenate(outs)
    err = np.abs(y - y_true[:y.size])
    return {"max_err": float(err.max()), "mean_err": float(err.mean())}

#这里开始结束

def sample_batch(batch_size):
    """QAT 微调批次: 随机窗锯齿 (x) -> 正弦 (y)."""
    t = torch.rand(batch_size, FRAME, 1)
    s = 2.0 * t - 1.0
    y = torch.sin(2.0 * np.pi * t)
    return (s.permute(0, 2, 1).unsqueeze(3).contiguous(),
            y.permute(0, 2, 1).unsqueeze(3).contiguous())
