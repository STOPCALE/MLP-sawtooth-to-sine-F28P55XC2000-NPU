# 编译报告 — saw2sin

- 输入模型: `E:\desk\ti-npu\saw2sin_npu\examples\saw2sin\out\saw2sin\saw2sin_int8.onnx`
- 目标器件: F28P55 | task: generic_timeseries_regression
- 编译模式: type=hard (启用 NPU)
- 产物: `E:\desk\ti-npu\saw2sin_npu\examples\saw2sin\out\saw2sin\compile\work\data\projects\saw2sin\run\saw2sin\compilation\artifacts\mod.a` | `E:\desk\ti-npu\saw2sin_npu\examples\saw2sin\out\saw2sin\compile\work\data\projects\saw2sin\run\saw2sin\compilation\artifacts\tvmgen_default.h` | 打包 `E:\desk\ti-npu\saw2sin_npu\examples\saw2sin\out\saw2sin\compile\work\data\projects\saw2sin\run\saw2sin\compilation\saw2sin_F28P55.zip`

## NPU 卸载 (Layer Patterns Offloaded)

- PWCONV x 1

## 未卸载算子 (Operators not Offloaded)

- add x 1
- multiply x 5
- floor x 1
- clip x 3
- cast x 4
- reshape x 2
- qnn.conv2d x 2
- nn.bias_add x 2
- right_shift x 2
- qnn.dequantize x 1

## mod.a 内存占用

- Code: 1054 B (1.03 KB)
- RO Data: 3442 B (3.36 KB)
- RW Data: 1398 B (1.37 KB)
- Total: 5894 B (5.76 KB)

## 警告 (NPU 回落原因等)

- PWCONV layer NOT offloaded to TI-NPU
- PWCONV layer config: 8-bit input [10, 1, 1], 8-bit output [10, 1, 64], 8-bit kernel [1, 1], strides [1, 1], padding [(0, 0), (0, 0)], data layout NHWC, kernel layout HWIO
- PWCONV does not support 1 input channels. Please change to multiples of 4.

> 完整日志: `compile/run.log`
