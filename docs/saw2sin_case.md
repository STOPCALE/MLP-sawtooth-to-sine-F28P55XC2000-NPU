# saw2sin 模型 → TI F28P55 NPU 部署转换记录

> 2026-10-06 实测。从 `E:\desk\DeepStudy\Charge_Num\saw2sin_model.pt`（float 权重）
> 到 F28P55 NPU 硬件可执行的 `mod.a` 的完整过程。
> `case_saw2sin/` 目录内的脚本 + 配置可以 100% 复现（前提：conda 环境 `ti-npu` 已搭好）。
>
> **归档说明**：本文档是 saw2sin 案例的原始转换记录（工程重组后移入 `docs/`，
> 脚本与编译产物在 `case_saw2sin/`）。通用工具 `torch2tinpu` 的用法见仓库根 `README.md`。

---

## 0. 成果速览

| 项目 | 结果 |
|---|---|
| 输入 | `saw2sin_model.pt`（float，MLP：1→64→64→1，逐点锯齿→正弦映射）|
| 转换产物 | `saw2sin_tinpu_int8.onnx`（TI-QAT int8，静态 `(1,1,10,1) → (1,1,10,1)`）|
| 编译产物 | `case_saw2sin\data\projects\saw2sin\run\saw2sin_tinpu\compilation\artifacts\mod.a` + `tvmgen_default.h` |
| NPU 卸载 | 中间层 64→64（≈97% 计算量）✅ 上 NPU；首层/末层回落 CPU（通道数硬约束，见 §4）|
| int8 精度 | max err ≈ 4.4e-2（QAT 15 轮，可调优）|
| 头文件标志 | `#define TVMGEN_DEFAULT_TI_NPU`（硬件版而非软件版）|
| **板端实测** | ✅ F28P55 LaunchPad 实机跑通：推理 **637.6µs/次**、输出与理想正弦 ≤1 LSB、VOFA+ 波形正常（见 §9）|

---

## 使用流程（速查）

> 全流程总览 + 三个常用场景。想看"为什么"跳到 §1~§3；想直接跑，看这里。

### 流程图

```mermaid
flowchart LR
    A[saw2sin_model.pt<br/>float 权重] -->|Stage A: TI-QAT 重量化| B[saw2sin_tinpu_int8.onnx<br/>int8, (1,1,10,1) 静态]
    B -->|Stage B: BYOM 编译| C[mod.a + tvmgen_default.h<br/>+ *_packed.bin]
    C -->|CCS 集成| D[F28P55 板端运行]
```

### 场景 ①：复现当前结果（两条命令）

| 步骤 | 操作 | 耗时 |
|---|---|---|
| 前提 | 新终端先设 `$env:C2000_CG_ROOT = 'E:\ccs\ccs\tools\compiler\ti-cgt-c2000_25.11.1.LTS'`（重启过 VS Code 的终端可省略）| - |
| Stage A | `E:\anaconda\envs\ti-npu\python.exe E:\desk\ti-npu\saw2sin_npu\case_saw2sin\saw2sin_tinpu_qat_conv.py` | ~1 分钟 |
| Stage B | （先 `cd` 到 `case_saw2sin`）`E:\anaconda\envs\ti-npu\python.exe -m tinyml_modelmaker.run_tinyml_modelmaker saw2sin_compile.yaml` | ~1 分钟 |

产物路径：`case_saw2sin\data\projects\saw2sin\run\saw2sin_tinpu\compilation\`（检查点见 §6.3）

### 场景 ②：改了模型 / 重训了权重后重新转换

1. **只是权重更新**（结构不变）：直接重跑场景 ① 的两条命令即可（权重路径写死在脚本里）
2. **网络结构变了**（层数 / 宽度）：改 `saw2sin_tinpu_qat_conv.py` 两处——
   - a) `MLP` 类（读权重用）和 `ConvMLP` 类（网络本体）的结构定义
   - b) 权重映射块 `convnet.net[i].weight.copy_(mlp.net[j].weight.view(...))` 的层号与 reshape 尺寸
   - 脚本自带**等价性断言**（MLP vs ConvMLP 必须 < 1e-5），改错了会直接报错，不会带病导出
3. **编译配置 `saw2sin_compile.yaml` 不用改**（除非换芯片或换 ONNX 路径）

### 场景 ③：产物上板（CCS 集成）

见 §9（含最小 C 代码骨架；"接口以头文件为准"）。

### 环境速查

| 项 | 值 |
|---|---|
| Python | `E:\anaconda\envs\ti-npu\python.exe`（conda env `ti-npu`，工具链为源码 editable 安装）|
| 环境自检 | `E:\anaconda\envs\ti-npu\python.exe E:\desk\ti-npu\check_ti_npu_env.py` |
| 编译器变量 | `C2000_CG_ROOT`（用户级已持久设置；运行中的 VS Code 需重启后新终端才会继承）|
| 工具链源码 | `E:\desk\ti-npu\tinyml-tensorlab-main\tinyml-tensorlab-main\` |
| 环境搭建全流程 | 见 `docs/env_setup.md`（从零搭建步骤 / 逐项自检 / 环境类踩坑清单）|

---

## 1. 背景：为什么旧的三个 ONNX 全都不能用

`Charge_Num` 里已有的三个模型（实测验证过）：

| 文件 | 不能用上 NPU 的原因 |
|---|---|
| `saw2sin.onnx` | float 模型 → NPU 只能跑 CPU |
| `saw2sin_int8.onnx` | 通用 CPU 量化（ORT QDQ）→ 激活/权重的量化格式不满足 TI NPU 硬件约束 |
| `saw2sin_qat_int8.onnx` | fbgemm QAT 导出 → 同样是通用量化，且导出图很脏 |
| 公共问题 | 三者都是**动态 batch**（NPU 只支持 batch=1 静态）|

结论：**必须用 TI 自家工具重新做 QAT 量化**，不能复用任何旧产物。

---

## 2. 原理：TI 链路做了什么

### 2.1 第一步：TI-QAT 重量化（`TINPUTinyMLQATFxModule`）

TI 的 QAT 包装器把普通 PyTorch 模型"包装"成符合 NPU 硬件约束的形态：

- 权重量化：per-channel 对称 + **2 的幂 scale**（硬件乘法器要求）
- 激活量化：per-tensor 对称
- 把量化图替换成 TINPU 专用算子形态（`right_shift`、幂次 scale 等）
- QAT 微调：前向模拟 int8 舍入误差（STE），网络学会补偿量化损失
- `convert()` 冻结 scale → `export()` 导出 QDQ 形式 int8 ONNX

### 2.2 第二步：编译（`ti_mcu_nnc` / tvmc）

TVM 后端把 int8 ONNX 拆成两部分：

- 能匹配硬件模式的层（FCONV/GCONV/PWCONV/FC）→ 生成 NPU 微码 + 数据（`*_packed.bin`、`*_mmr.bin`）
- 其余算子 → 生成 C2000 软件库（用 `cl2000` 交叉编译进 `mod.a`）
- 报告 `TI NPU Offloading Report`：`[X]` = 上 NPU，`[ ]` = 回落 CPU

### 2.3 数据形态约定

TI 时序流按**窗口**处理：`(1, 1, frame, 1)`（batch=1，channel=1，frame 个连续采样点）。
本模型逐点独立映射，frame 内各点互不影响——所以任何窗口长度语义都成立（本次取 10）。

---

## 3. 核心发现（原理级，值得反复看）

### 3.1 发现一：全连接层的"形态之争"——MatMul+Add 不被识别

**现象**：纯 MLP 直导（Linear 版）编译后 **NPU 零卸载**，全在 CPU。

**排查**（对照实验法）：把官方成功案例（`generic_timeseries_regression` 的 REGR_2k int8 模型）
用**同一个 BYOM 配置**编译 → FCONV/GCONV/FC 全部正常卸载 ⇒ 流程没问题，是模型图结构问题。

**根因**（直接对比两个 ONNX 的图结构得到）：

```
官方模型的全连接:  Gemm(A, B, bias)              ← 三输入融合形态 → 被识别为 FC → 上 NPU
我们的 MLP 直导:   MatMul(A, B) + Add(bias)      ← 偏置分离      → 不被识别   → 回落 CPU
```

- PyTorch 导出时，全连接是否融合成 `Gemm` 取决于 **ORT 的图优化**；
- ORT 的 `MatMul+Add → Gemm` 融合**只对 2 维输入生效**；
- TI 时序流张量是 3/4 维（如 `(1,10,1)`）→ 融合不了 → 永远拿不到 Gemm → FC 永远上不了 NPU。

（试过改 opset 17→18，无效；opset 18 是官方默认值，改了只是"对齐官方"，不解决此问题。）

### 3.2 发现二：用 1×1 卷积等价表达（解法）

`Linear(1→64)` 与 `Conv2d(1→64, kernel=1×1)` **数学完全等价**：

$$y[o,h,w] = \sum_c W[o,c,0,0]\cdot x[c,h,w] + b[o] \qquad (\text{in}=1 \Rightarrow y = W[:,0]\cdot x + b)$$

卷积形态正好匹配 TI NPU 官方支持的 FCONV/GCONV/PWCONV 路径（官方模型全是这种结构）。

**权重映射**（零损失复用原 checkpoint）：

```python
convnet.net[0].weight.copy_(mlp.net[0].weight.view(64, 1, 1, 1))   # Linear(1->64) -> Conv(1->64,1x1)
convnet.net[2].weight.copy_(mlp.net[2].weight.view(64, 64, 1, 1))  # Linear(64->64)
convnet.net[4].weight.copy_(mlp.net[4].weight.view(1, 64, 1, 1))   # Linear(64->1)
# bias 一一对应直接 copy
```

脚本里带自动断言：等价性实测 max diff = 3.6e-07（浮点误差级别）✅

---

## 4. 实测卸载结果解读

```
Layer Patterns Offloaded:
  PWCONV          1        ← 中间层 64→64 ✅
```

| 层 | 为什么这个结果 |
|---|---|
| 中间层 64→64 | 通道数 64（4 的倍数）✓ 卸载成功 |
| 首层 1→64 | 报错原文：*"PWCONV does not support 1 input channels. Please change to multiples of 4."*（1×1 卷积要求输入通道为 4 的倍数）|
| 末层 64→1 | 输出通道=1，静默回落（占计算量仅 ~1.5%，可忽略）|

> 注意：BYOM 文档说"首层输入通道 = 1"是针对 **kernel>1×1 的 FCONV** 的规则；
> 1×1 卷积会被归类为 PWCONV，通道必须 4 的倍数。两条规则不矛盾，看层被分到哪个模式。

**读报告指南**：`[X]` = 该算子跑在 NPU 硬件上；warnings 会写明不满足的硬件约束。

---

## 5. 踩坑清单（按时间顺序）

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | 编译脚本读 YAML 崩溃 `UnicodeDecodeError: gbk` | TI 脚本用系统默认编码（GBK）打开文件，YAML 里有中文注释 | YAML 只写 ASCII |
| 2 | `KeyError: 'data_processing_feature_extraction'` | 官方文档写的是 `feature_extraction:`（不准），源码找的是全名 | 用全名 `data_processing_feature_extraction` |
| 3 | 编译变 `type=soft`（CPU 软件库），无 NPU 报告 | BYOM 配置没写 `training.quantization`，流程把 `type=hard` 自动降级 soft（源码 `tinyml_benchmark.py:201`）| config 里加 `quantization: 2` |
| 4 | hard 模式仍零卸载 | 全连接是 `MatMul+Add` 形态非 `Gemm`（§3.1）| 改用 1×1 卷积表达（§3.2）|
| 5 | 新终端编译报 `Cross Compiler path is invalid: C:\Users\HP\ti\...` | `C2000_CG_ROOT` 已是用户级环境变量，但**运行中的 VS Code 进程缓存的是旧环境**，它开的新终端不继承 | 重启 VS Code，或当前会话手动 `$env:C2000_CG_ROOT='E:\ccs\ccs\tools\compiler\ti-cgt-c2000_25.11.1.LTS'` |
| 6 | 导出模型 ORT 融合行为与官方不一致 | 官方默认 opset=18，wrapper 默认 opset=17 | `qat.export(..., opset_version=18)` |

---

## 6. 复现步骤

### 6.0 前提

- conda 环境 `ti-npu` 已搭好（自检：`E:\anaconda\envs\ti-npu\python.exe E:\desk\ti-npu\check_ti_npu_env.py`）
- 终端有 `C2000_CG_ROOT`（新终端需重启过 VS Code；否则先手动设）

### 6.1 Stage A —— TI-QAT 重量化（约 1 分钟）

```powershell
E:\anaconda\envs\ti-npu\python.exe E:\desk\ti-npu\saw2sin_npu\case_saw2sin\saw2sin_tinpu_qat_conv.py
# 可选: 末尾加参数指定窗口长度, 如 `... 1` 表示 (1,1,1,1)
```

期望输出：等价性 diff < 1e-5；15 轮 QAT loss ~1e-4 量级；导出 `saw2sin_tinpu_int8.onnx`；int8 max err ~4e-2

### 6.2 Stage B —— 编译到 F28P55（约 1 分钟）

```powershell
$env:C2000_CG_ROOT = 'E:\ccs\ccs\tools\compiler\ti-cgt-c2000_25.11.1.LTS'   # 新终端需要
Set-Location 'E:\desk\ti-npu\saw2sin_npu\case_saw2sin'
E:\anaconda\envs\ti-npu\python.exe -m tinyml_modelmaker.run_tinyml_modelmaker saw2sin_compile.yaml
```

### 6.3 验证检查点

报告里应该看到：
- `Target: c, ti-npu type=hard ...`（是 hard 不是 soft）
- `Layer Patterns Offloaded: PWCONV 1`（中间层 [X]）
- `TI Model Library Memory Usage (mod.a): Total ≈ 5.9 KB`

产物路径：`case_saw2sin\data\projects\saw2sin\run\saw2sin_tinpu\compilation\`

---

## 7. 文件清单

> 下表路径均在 `case_saw2sin\` 目录下。

| 文件 | 作用 |
|---|---|
| `saw2sin_tinpu_qat_conv.py` | **主脚本**：卷积版 QAT 重量化 + 导出（Stage A）|
| `saw2sin_compile.yaml` | 编译配置（Stage B）|
| `saw2sin_tinpu_qat.py` | 对照版（Linear/MLP 直导）——留作 Gemm 问题的实验证据，实际不用 |
| `_test_e2e_model.yaml` / `ref_regr2k_model.onnx` | 对照实验：官方模型走同一 BYOM 流程（证明流程没问题）|
| `_onnx_diff.py` | 诊断脚本：对比两个 ONNX 图结构（发现 MatMul+Add vs Gemm 的功臣）|
| `saw2sin_tinpu_int8.onnx` | Stage A 产物（TI-QAT 量化模型）|
| `data\projects\...` | Stage B 产物（mod.a / tvmgen_default.h / F28P55.zip）|
| `README.md` | 本文档（原理 / 坑 / 复现 / 使用流程；工程重组前的 README）|

---

## 8. 已知限制与可选优化

| 项 | 现状 | 可选优化 |
|---|---|---|
| int8 精度 | max err ≈ 4.4e-2 | QAT 加 epoch（15 → 30~50）、调 LR；改 frame 长度影响不大 |
| 首层 CPU | 输入通道=1 不满足 4 倍数 | 输入零填充到 4 通道（3 通道填 0，权重也填 0）→ 可尝试，但只值 ~1.5% 计算量 |
| 末层 CPU | 输出通道=1 | 同上，收益小 |
| NPU 占比 | 中间层 ≈97% MACs 已上 NPU | 实际加速效果建议板端实测 |

---

## 9. 板端集成（✅ 2026-10-06 已完成并在实机验证）

### 9.1 集成内容（CCS 工程 `E:\desk\CCS\uart_and_npu`）

| 项 | 内容 |
|---|---|
| 模型库 | `artifacts\mod.a` + `artifacts\tvmgen_default.h`（从编译产物复制）|
| NPU 支撑源文件 | `device\f28p55x_npu.c`（从 C2000Ware device_support 复制；mod.a 依赖它提供 `NPU_setInstrMem/ParamMem/InterruptHandler/clearInterruptACK`）|
| 链接脚本 | `28p55x_generic_{flash,ram}_lnk.cmd` 新增：`.rodata.tvm → FLASH_BANK0 / RAMGS3`；`.bss.noinit.tvm → RAMGS3` |
| 构建文件 | `makefile` + `device\subdir_vars.mk`：加入 `device\f28p55x_npu.obj` 和 `../artifacts/mod.a` |
| 主程序 | `empty_driverlib_main.c` 重写为验证程序（原回显程序备份为 `.bak`）|

### 9.2 验证程序功能（对应板端验证三条需求）

1. **片内自行产生输入**：锯齿波 `s = 2*phase-1`，步进 1/2000（幅值 ±1 精确）
2. **每次推理计时**：CPUTimer1（150MHz 递减计数），从 `tvmgen_default_run()` 到完成标志置位
3. **串口帧（VOFA JustFloat）**：10 输入 → 10 输出 → 1 耗时(us) → 帧尾 `00 00 80 7F`

### 9.3 VOFA+ 配置

- 串口：**COM6**（XDS110 Application/User UART），115200
- 协议：**JustFloat**
- 通道：CH0–CH9 输入(锯齿) ‖ CH10–CH19 输出(正弦) ‖ CH20 推理耗时(us)

### 9.4 实测结果（真实闭环数据）

| 项 | 结果 |
|---|---|
| 输入锯齿波 | 逐点步进 +0.001，与设定相位完全吻合 ✓ |
| 输出 vs 理想正弦 | 全部 ≤1 LSB（输出量化步长 1/64≈0.0156）✓ |
| 单次推理耗时 | **637.6 µs**（含 CPU 回退层 + NPU 层 + 轮询，非常稳定）|
| 数据速率 | ≈120 帧/秒（637µs 推理 + 88B@115200）；锯齿周期 ≈1.7s，VOFA 显示清晰 |

### 9.5 注意事项

- **同一时刻只有一个程序能占用 COM6**：VOFA 打开时其它程序读取会被拒绝（正常现象）
- 命令行构建/烧录：`gmake -C CPU1_FLASH all` → `DSLite.exe flash --config=TMS320F28P550SJ9_LaunchPad.ccxml CPU1_FLASH\uart_and_npu.out`
- 若以后用 CCS GUI 重新生成 makefile，需要把 `f28p55x_npu.c` 与 `mod.a` 重新加入工程

**C 侧调用骨架**（实际完整实现见工程 `empty_driverlib_main.c`，函数/结构体名以 `tvmgen_default.h` 为准）：

```c
#include "tvmgen_default.h"   /* 提供: TI_NPU_init / tvmgen_default_run / tvmgen_default_finished */

static struct tvmgen_default_inputs  npu_in;
static struct tvmgen_default_outputs npu_out;

void saw2sin_npu_init(void)
{
    TI_NPU_init();                  /* 使能 NPU 硬件（含中断配置）*/
}

/* 输入/输出均为 (1,1,10,1) float32 —— C 侧按一维数组传指针即可 */
void saw2sin_npu_run(const float *saw10, float *sin10)
{
    npu_in.input    = (void *)saw10;
    npu_out.output  = (void *)sin10;
    tvmgen_default_run(&npu_in, &npu_out);
    while (!tvmgen_default_finished) { }     /* 轮询等待 NPU 完成（也可改成中断里等）*/
}
```
