# -*- coding: utf-8 -*-
"""
部署路线 A · 第 2 步: 训练后量化 PTQ (float32 -> int8)

流程:
  1. 标定 (calibration): 用与训练同分布的输入喂 float ONNX, 统计每层激活范围
  2. quantize_static: 生成 int8 模型 saw2sin_int8.onnx (QDQ 格式)
  3. 验证: int8 输出 vs float 输出 vs 解析目标, 量化误差一目了然
  4. 打印真实的量化 scale (标定结果), 与理论值对照

运行: conda run --no-capture-output -n torch python onnx_int8_quant.py
"""

import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FLOAT_ONNX = os.path.join(OUT_DIR, "saw2sin.onnx")
INT8_ONNX = os.path.join(OUT_DIR, "saw2sin_int8.onnx")


class SawCalib(CalibrationDataReader):
    """标定数据器: 提供有代表性的输入样本 (这里: s 均匀覆盖 [-1,1], 与训练分布一致)"""

    def __init__(self, n=4096, batch=256):
        s = np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(n, 1)
        self._batches = iter([{"s": s[i:i + batch]} for i in range(0, n, batch)])

    def get_next(self):
        return next(self._batches, None)


# ---------- 量化 ----------
quantize_static(
    FLOAT_ONNX,
    INT8_ONNX,
    SawCalib(),
    quant_format=QuantFormat.QDQ,   # QDQ 格式: 图里显式出现 QuantizeLinear/DequantizeLinear
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,               # 权重逐输出通道各用一个 scale (更精确)
)
print("saved:", INT8_ONNX)

# 文件大小对比 (部署体积的第一手数据)
sz_f = os.path.getsize(FLOAT_ONNX) / 1024
sz_q = os.path.getsize(INT8_ONNX) / 1024
print("file size: float %.1f KB -> int8 %.1f KB (x%.2f smaller)" % (sz_f, sz_q, sz_f / sz_q))

# ---------- 验证 ----------
s_np = np.linspace(-1, 1, 1000, dtype=np.float32).reshape(-1, 1)
sess_f = ort.InferenceSession(FLOAT_ONNX, providers=["CPUExecutionProvider"])
sess_q = ort.InferenceSession(INT8_ONNX, providers=["CPUExecutionProvider"])
y_float = sess_f.run(None, {"s": s_np})[0]
y_int8 = sess_q.run(None, {"s": s_np})[0]
y_true = -np.sin(np.pi * s_np)


def report(tag, y):
    print("%s | vs float: max=%.2e mean=%.2e | vs target: max=%.2e"
          % (tag, np.abs(y - y_float).max(), np.abs(y - y_float).mean(),
             np.abs(y - y_true).max()))


report("float", y_float)
report("int8 ", y_int8)

# ---------- 查看量化参数 (教学: scale 长什么样) ----------
m = onnx.load(INT8_ONNX)
scales = [init for init in m.graph.initializer if "scale" in init.name.lower()]
print("num tensors with scale:", len(scales))
for init in scales[:8]:
    arr = numpy_helper.to_array(init).ravel()
    print("   %-44s %s" % (init.name, np.round(arr[:3], 6)))
