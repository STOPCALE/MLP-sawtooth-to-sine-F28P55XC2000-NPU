# saw2sin — F28P55 部署集成说明

> 由 torch2tinpu deploy 自动生成; 接口细节以 `artifacts/tvmgen_default.h` 实际内容为准。

## 1. 文件清单

| 文件 | 说明 |
|---|---|
| `artifacts/mod.a` | 模型库 (NPU 微码 + CPU 回退算子), 加入 CCS 工程参与链接 |
| `artifacts/tvmgen_default.h` | 模型接口头文件 (I/O 结构体 / 运行函数 / NPU 初始化) |

- 模型 I/O: 输入 (1, 1, 10, 1), float32; 输出 (1, 1, 10, 1), float32
- 运行位置: TI NPU 硬件加速器 (TVMGEN_DEFAULT_TI_NPU)

## 2. CCS 集成步骤

参考例程: `E:\ccs\C2000Ware_26_01_00_00\libraries\ai\examples\arc_fault\f28p55x`

1. 把 `artifacts/` 加入 CCS 工程 (参考例程做法: `mod.a` 作为库文件参与链接)
2. 在工程 `lnk.cmd` 中增加模型数据段 (段名固定; 放置位置参考对应例程):

   ```
   .rodata.tvm      : > FLASH_BANK0      /* 模型常量 -> Flash */
   .bss.noinit.tvm  : > RAMGS3           /* NPU 数据 -> 全局共享 RAM (实测 RAMGS3 可行; 名称按器件/工程调整) */
   ```

3. main 中调用 (字段名来自头文件: `input` / `output`):

   ```c
   TI_NPU_init();                     /* 使能 NPU (含中断配置), 上电后一次 */
   in.input  = (void *)input_buf;
   out.output = (void *)output_buf;
   tvmgen_default_run(&in, &out);
   while (!tvmgen_default_finished) { }   /* 轮询等待 NPU 完成 */
   ```

4. 用 golden vectors 做上板对比验证 (参考例程 `test_vector.c` 的组织方式)

## 3. 编译报告摘要

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


## 4. 常见问题

- 编译报 `Cross Compiler path is invalid`: 运行 torch2tinpu compile 的终端缺 C2000_CG_ROOT (本工具已自动注入, 属历史坑位)
- 内存不足: 对照上方 mod.a 内存报告调整模型规模或段放置
- 板端结果和 PC 端不一致: 先确认 golden vector 与 Python int8 推理(onnxruntime)一致, 再查板端数据布局/归一化
- NPU 未启用: 确认头文件里有 `#define TVMGEN_DEFAULT_TI_NPU` (没有则表示编译成了 CPU 软库)
