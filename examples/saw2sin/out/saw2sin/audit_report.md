# ONNX 审计报告

- 文件: `E:\desk\ti-npu\saw2sin_npu\examples\saw2sin\out\saw2sin\saw2sin_int8.onnx`

## 警告 (会导致 NPU 回落 / 降级)
- 1x1 卷积 '/net.0/net.0.0/Conv': 输入通道 1 / 输出通道 64 非 4 的倍数 -> PWCONV 被拒回落 CPU
- 1x1 卷积 '/net.4/net.4.0/Conv': 输入通道 64 / 输出通道 1 非 4 的倍数 -> PWCONV 被拒回落 CPU

## 信息
- opset=18 (与 TI 官方流程一致)
- 输入 input 形状 [1, 1, 10, 1]
- 输出 output 形状 [1, 1, 10, 1]
- 算子统计: {'Mul': 9, 'Clip': 6, 'Add': 4, 'Floor': 4, 'Conv': 3, 'Relu': 2}
- 量化格式: TI-NPU 风格 (Add/Mul/Floor/Clip 整数表达), 可作为编译输入
- 1x1 卷积 '/net.2/net.2.0/Conv': 64->64 满足 4 的倍数约束 (预期上 NPU)
- 提示: 通道数不满足约束的 1x1 卷积可以零填充到 4 的倍数 (权重也填 0), 用少量计算量换 NPU 卸载
