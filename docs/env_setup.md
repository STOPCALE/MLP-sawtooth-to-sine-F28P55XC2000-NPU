# TI NPU 开发环境搭建与自检（ti-npu / Windows）

> 本文档记录 TI NPU 工具链的**完整搭建流程、自检方法与踩过的坑**（本机实测，2026-10）。
> 目标：换一台机器能照着复现；出问题时知道先查哪一环。
> 相关文档：工具用法见仓库根 `README.md`；模型/编译类坑位见 `docs/saw2sin_case.md`。

---

## 0. 环境总览（本机实测状态）

| 组件 | 本机值 | 作用 |
|---|---|---|
| Python 环境 | conda env `ti-npu`（Python 3.10.22）<br>`E:\anaconda\envs\ti-npu\python.exe` | 跑量化 / 编译 / 部署全流程 |
| 工具链源码 | `E:\desk\ti-npu\tinyml-tensorlab-main\tinyml-tensorlab-main\`（v1.5.0，2026-Sep 发布）| 4 个子仓（见 §1.3）|
| NPU 编译器 | `ti-mcu-nnc 2.1.2`（TVM 系）| int8 ONNX → `mod.a` + `tvmgen_default.h` |
| C2000 交叉编译器 | `E:\ccs\ccs\tools\compiler\ti-cgt-c2000_25.11.1.LTS`（`cl2000`）| 编译 CPU 回退算子并打包进 mod.a |
| 板端库/例程 | `E:\ccs\C2000Ware_26_01_00_00` | `libraries\ai\examples\arc_fault\f28p55x`（集成模板）、`device_support\...\f28p55x_npu.c` |
| 关键环境变量 | `C2000_CG_ROOT` | 编译流程定位交叉编译器（**必须**）|

**Python 包版本明细（实测）**：

| 包 | 版本 | 安装方式 |
|---|---|---|
| tinyml_modelmaker / tinyml_tinyverse / tinyml_torchmodelopt / tinyml_modelzoo | 1.5.0 | 本地源码 editable |
| ti-mcu-nnc | 2.1.2 | TI CDN wheel |
| torch | 2.11.0+cpu | pip |
| onnx / onnxruntime / onnxsim | 1.22.0 / 1.23.2 / 0.4.36 | pip |
| numpy / pandas / scikit-learn / matplotlib / PyYAML | 2.2.6 / 2.3.3 / 1.7.2 / 3.10.9 / 6.0.3 | pip |

---

## 1. 从零搭建（6 步）

### 1.1 建 conda 环境

```powershell
conda create -n ti-npu python=3.10 -y
```

- 工具链 1.5.0 兼容 Python 3.10 ~ 3.14；本机 3.10.22 实测全流程通过。
- ⚠ **本机经验**：VS Code 终端里 conda 常未初始化，`conda activate ti-npu` 会失败/不生效。
  所以本仓库所有命令**一律用绝对路径解释器**：`E:\anaconda\envs\ti-npu\python.exe`（照抄即可）。

### 1.2 获取工具链源码

- 官方方式：跑 `tinyml-modelmaker\setup_all.ps1`，自动 clone 4 个仓到父目录（详见 §5）。
- 本机方式：下载整包 `tinyml-tensorlab-main.zip` 解压到 `E:\desk\ti-npu\`。
  注意解压后是**双层同名目录**：`E:\desk\ti-npu\tinyml-tensorlab-main\tinyml-tensorlab-main\`
  （下文称它为 `$PARENT`，4 个子仓都在里面）。

### 1.3 editable 安装工具链 4 个包

> 这 4 个包 **PyPI / 镜像源里没有**，只能从源码装。

```powershell
$PARENT = 'E:\desk\ti-npu\tinyml-tensorlab-main\tinyml-tensorlab-main'
$PY     = 'E:\anaconda\envs\ti-npu\python.exe'

& $PY -m pip install --upgrade pip setuptools wheel
git config --global --add safe.directory $PARENT     # ← PowerShell 下 editable 安装必需（见 §3-③）

& $PY -m pip install -e "$PARENT\tinyml-modeloptimization\torchmodelopt"   # TI-QAT/PTQ 包装器
& $PY -m pip install -e "$PARENT\tinyml-tinyverse"                        # 训练框架
& $PY -m pip install -e "$PARENT\tinyml-modelmaker"                       # 编译/部署流程
& $PY -m pip install -e "$PARENT\tinyml-modelzoo"                         # 官方模型库
```

torch / onnx / onnxruntime / onnxsim 等依赖会随上面安装自动带入。
本机 pip 源是清华镜像（PyPI 加速）；TI 自己的包不走 PyPI（见 1.4）。

验证：`& $PY -m pip list | Select-String "tinyml"` —— 4 个包都应显示 1.5.0 + 本地源码路径。

### 1.4 安装 NPU 编译器 ti-mcu-nnc

> **不在 PyPI / 镜像里**（`pip install ti-mcu-nnc` 直接失败），必须用 TI 官方 CDN 的直链 wheel。

```powershell
# 本机实测安装命令（注意 wheel 名要匹配你的 Python 版本与平台: cp310 + win_amd64）
& $PY -m pip install https://software-dl.ti.com/mctools/esd/tvm/mcu/ti_mcu_nnc-2.1.2-cp310-cp310-win_amd64.whl

& $PY -c "import ti_mcu_nnc; print('ti_mcu_nnc OK')"
```

- 换 Python 版本/平台时，把 URL 里的 `cp310-cp310-win_amd64` 换成对应值（wheel 列表见 NNC 用户指南，§5）。
- 它是 TVM 系编译器，**不需要**另外安装 `tvm` 包。
- 查看已装来源：`& $PY -m pip show ti-mcu-nnc`。

### 1.5 C2000 交叉编译器 + `C2000_CG_ROOT`

| 平台 | 获取方式 |
|---|---|
| Linux / macOS | 官方脚本 `tinyml-modelmaker\setup_cg_tools.sh`（自动下载 `ti_cgt_c2000_25.11.1.LTS` 静默安装到 `~/ti`）|
| Windows（本机）| 随 CCS 安装：`E:\ccs\ccs\tools\compiler\ti-cgt-c2000_25.11.1.LTS` |

设置用户级环境变量并验证：

```powershell
[Environment]::SetEnvironmentVariable('C2000_CG_ROOT',
    'E:\ccs\ccs\tools\compiler\ti-cgt-c2000_25.11.1.LTS', 'User')

Test-Path "$env:C2000_CG_ROOT\bin\cl2000.exe"      # 期望 True
```

- ⚠ 这是编译期**最容易踩的坑**（见 §3-②）。用 `torch2tinpu compile` 时工具会自动探测并注入，无需手工设置；
  直接手跑 `tinyml_modelmaker` 时则必须自己保证这个变量生效。
- 版本要匹配：工具链 1.5.0 默认用 **25.11.1.LTS**。

### 1.6 C2000Ware（板端库）

- 本机：`E:\ccs\C2000Ware_26_01_00_00`（随 CCS 安装或从 TI 官网单独下载）。
- 用途：板端集成时对照 `libraries\ai\examples\arc_fault\f28p55x`；
  `device_support` 里的 `f28p55x_npu.c` 是 `mod.a` 的运行时依赖。

---

## 2. 环境自检（配好后必跑）

### 2.1 工具链自检脚本

```powershell
E:\anaconda\envs\ti-npu\python.exe E:\desk\ti-npu\check_ti_npu_env.py
```

逐项打印 `[OK]` / `[FAIL]`，共 9 项：
tinyml_modelmaker、tinyml_tinyverse、tinyml_torchmodelopt、tinyml_modelzoo、
torch、onnx、onnxruntime、TI-NPU QAT 入口（`TINPUTinyMLQATFxModule` 可导入）、ti_mcu_nnc。

全部通过 → 输出「全部组件就绪」；任一 `[FAIL]` → 回到 §1 对应步骤重装。

### 2.2 工具端到端自检

```powershell
cd E:\desk\ti-npu\saw2sin_npu
E:\anaconda\envs\ti-npu\python.exe -m torch2tinpu check examples\saw2sin\job_saw2sin.yaml
```

一次同时验证：依赖齐全 / adapter 与模型可实例化 / 样例输入形状 / **交叉编译器自动探测结果**
（输出里的 `[env ] 自动探测编译器: ...` 一行）。

---

## 3. 环境类踩坑清单

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| ① | `conda activate ti-npu` 无效，`python` 找不到/torch 缺失 | VS Code 终端里 conda 未初始化 | **一律用绝对路径** `E:\anaconda\envs\ti-npu\python.exe`，不做 activate |
| ② | 编译报 `Cross Compiler path is invalid: C:\Users\HP\ti\...` | `C2000_CG_ROOT` 未生效——即使已设成用户级变量，**运行中的 VS Code 缓存了旧环境**，它开的新终端不继承 | 重启 VS Code；或当前会话手动 `$env:C2000_CG_ROOT='E:\ccs\...\ti-cgt-c2000_25.11.1.LTS'`；用 `torch2tinpu compile` 则自动处理 |
| ③ | `pip install -e` 报 `dubious ownership` / safe.directory 提示 | git 认为仓库目录归属异常（PowerShell 环境常见） | `git config --global --add safe.directory <仓库路径>` |
| ④ | `pip install tinyml-modelmaker` 等报「找不到包」 | 这 4 个包不在 PyPI / 镜像源 | 按 §1.3 用本地源码 editable 安装 |
| ⑤ | `pip install ti-mcu-nnc` 报「No matching distribution」 | 该包也不在 PyPI / 镜像源 | 按 §1.4 用 TI CDN 直链 wheel 安装 |
| ⑥ | 下载/安装 TI 资源失败或极慢 | 需访问 TI 官方域 | 确认 `software-dl.ti.com`、`dr-download.ti.com` 可直连（本机可；受限网络需换源/代理）|
| ⑦ | 编译脚本崩溃 `UnicodeDecodeError: 'gbk' codec can't decode ...` | 中文 Windows 默认 GBK，而编译 YAML 含非 ASCII 字符 | 编译 YAML 必须**纯 ASCII**（`torch2tinpu` 生成时已强制保证）|

---

## 4. 速查表

| 项 | 值 |
|---|---|
| conda 环境 | `ti-npu`（Python 3.10.22）|
| Python 解释器 | `E:\anaconda\envs\ti-npu\python.exe` |
| 工具链源码（$PARENT）| `E:\desk\ti-npu\tinyml-tensorlab-main\tinyml-tensorlab-main\` |
| 环境自检脚本 | `E:\desk\ti-npu\check_ti_npu_env.py` |
| NPU 编译器 | `ti-mcu-nnc 2.1.2`（TI CDN wheel）|
| 交叉编译器 | `E:\ccs\ccs\tools\compiler\ti-cgt-c2000_25.11.1.LTS` |
| C2000Ware | `E:\ccs\C2000Ware_26_01_00_00` |
| 关键环境变量 | `C2000_CG_ROOT` |
| 工具入口 | `python -m torch2tinpu <check/convert/compile/deploy/all> <job.yaml>` |
| pip 源 | 清华镜像（PyPI 加速；TI 包除外）|

---

## 5. 参考（官方文件与链接）

| 文件 / 链接 | 说明 |
|---|---|
| `tinyml-modelmaker\setup_all.ps1` | 官方 Windows 安装脚本（clone 4 仓 + editable 安装）|
| `tinyml-modelmaker\setup_all.sh` | 官方 Linux/macOS 安装脚本 |
| `tinyml-modelmaker\setup_cg_tools.sh` | 官方 C2000 编译器自动安装脚本（Linux/macOS）|
| `tinyml-modelmaker\run_tinyml_modelmaker.sh` | 官方编译运行脚本（含 WORK_DIR / DATA_DIR / TOOLS_PATH 约定）|
| `tinyml-modelmaker\DEVICE_TASK_SUPPORT.md` | 器件 / 任务支持矩阵 |
| `tinyml-modelzoo\docs\NPU_CONFIGURATION_GUIDELINES.md` | **官方 NPU 层约束**（FCONV / GCONV / DWCONV / PWCONV / FC 的通道与内核规则）|
| NNC 用户指南 | https://software-dl.ti.com/mctools/nnc/mcu/users_guide/ |
| Tiny ML Tensorlab 用户指南 | https://software-dl.ti.com/C2000/esd/mcu_ai/user_guide/index.html |
