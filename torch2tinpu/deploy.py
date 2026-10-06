# -*- coding: utf-8 -*-
"""Stage C: 生成部署集成包 (产物拷贝 + CCS 集成说明 + C 调用模板).

产物结构:
  <run>/deploy/
    ├── artifacts/mod.a             (编译后的模型库)
    ├── artifacts/tvmgen_default.h  (接口头文件)
    ├── INTEGRATION.md              (CCS 集成步骤 / 段配置 / golden vector 建议)
    └── <name>_npu.h / <name>_npu.c (可选: 调用模板, 单入单出模型才生成)

说明:
  - I/O 形状与 struct 字段名全部从 tvmgen_default.h 实际内容解析, 不靠猜测
  - .rodata.tvm / .bss.noinit.tvm 段配置为 F28P55 实测 (参考 C2000Ware arc_fault 例程);
    不同器件段名/RAM 名称可能不同, 以对应例程为准
"""

import glob
import os
import re
import shutil

from . import ToolError

_C2000WARE_GLOBS = [
    r"E:\ccs\C2000Ware_*\libraries\ai\examples\arc_fault\*",
    r"C:\ti\c2000\C2000Ware_*\libraries\ai\examples\arc_fault\*",
]


def _parse_header(header_path):
    """从 tvmgen_default.h 解析 I/O 结构体字段与形状."""
    with open(header_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    def fields(struct_name):
        m = re.search(r"struct\s+" + struct_name + r"\s*\{(.*?)\}", text, re.S)
        return re.findall(r"void\s*\*\s*(\w+)\s*;", m.group(1)) if m else []

    shapes_in, shapes_out = [], []
    m = re.search(r"expects the following inputs/outputs:(.*?)\*/", text, re.S)
    if m:
        parts = m.group(1).split("Outputs:")
        shapes_in = re.findall(r"Tensor\[([^\]]+)\]", parts[0])
        if len(parts) > 1:
            shapes_out = re.findall(r"Tensor\[([^\]]+)\]", parts[1])

    def elems(shape_str):
        m2 = re.search(r"\(([^)]*)\)", shape_str)
        if not m2:
            return None
        n = 1
        try:
            for tok in m2.group(1).split(","):
                tok = tok.strip()
                if tok:
                    n *= int(tok)
        except ValueError:
            return None
        return n

    return {
        "in_fields": fields("tvmgen_default_inputs"),
        "out_fields": fields("tvmgen_default_outputs"),
        "shapes_in": shapes_in,
        "shapes_out": shapes_out,
        "elems_in": [elems(s) for s in shapes_in],
        "elems_out": [elems(s) for s in shapes_out],
        "is_tinpu": "TVMGEN_DEFAULT_TI_NPU" in text,
    }


def _find_reference_example(device):
    """找 C2000Ware 里对应器件的参考例程目录."""
    sub = {"F28P55": "f28p55", "F28P65": "f28p65"}
    want = sub.get(device)
    hits = []
    for pattern in _C2000WARE_GLOBS:
        hits += [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if want:
        for d in hits:
            if want in os.path.basename(d).lower():
                return d
    return hits[0] if hits else None


def run_deploy(job, console=print):
    compile_dir = os.path.join(job.run_dir, "compile")
    work_dir = os.path.join(compile_dir, "work")
    comp = os.path.join(work_dir, "data", "projects", job.compile.dataset_name,
                        "run", job.name, "compilation")
    mod_a = os.path.join(comp, "artifacts", "mod.a")
    header = os.path.join(comp, "artifacts", "tvmgen_default.h")
    if not (os.path.isfile(mod_a) and os.path.isfile(header)):
        raise ToolError(f"未找到编译产物 ({mod_a})\n"
                        f"先运行: python -m torch2tinpu compile \"{job.path}\"")

    deploy_dir = os.path.join(job.run_dir, "deploy")
    art_dir = os.path.join(deploy_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    shutil.copyfile(mod_a, os.path.join(art_dir, "mod.a"))
    shutil.copyfile(header, os.path.join(art_dir, "tvmgen_default.h"))

    hdr = _parse_header(header)
    ref = _find_reference_example(job.compile.target_device)

    summary_md = ""
    summary_path = os.path.join(compile_dir, "summary.md")
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_md = f.read()

    md = _build_integration_md(job, hdr, ref, mod_a, header, summary_md)
    with open(os.path.join(deploy_dir, "INTEGRATION.md"), "w", encoding="utf-8") as f:
        f.write(md)

    c_files = None
    if job.deploy.generate_c_template:
        if len(hdr["in_fields"]) == 1 and len(hdr["out_fields"]) == 1 and \
                hdr["elems_in"] and hdr["elems_out"] and hdr["elems_in"][0] and hdr["elems_out"][0]:
            c_files = _write_c_template(job, deploy_dir, hdr)
        else:
            console("[deploy] 模型为多入/多出, 跳过 C 模板生成 (参考 INTEGRATION.md 的调用骨架)")

    console(f"[deploy] 集成包: {deploy_dir}")
    console(f"[deploy]   artifacts/mod.a + tvmgen_default.h")
    console(f"[deploy]   INTEGRATION.md (CCS 集成步骤)")
    if c_files:
        console(f"[deploy]   {os.path.basename(c_files[0])} + {os.path.basename(c_files[1])} (调用模板)")
    console("[done  ] 将 deploy/artifacts 与 xxx_npu.c/.h 加入 CCS 工程即可上板 (详见 INTEGRATION.md)")
    return deploy_dir


def _build_integration_md(job, hdr, ref, mod_a, header, summary_md):
    in_f = hdr["in_fields"][0] if hdr["in_fields"] else "input"
    out_f = hdr["out_fields"][0] if hdr["out_fields"] else "output"
    shapes_in = "; ".join(hdr["shapes_in"]) or "(未解析)"
    shapes_out = "; ".join(hdr["shapes_out"]) or "(未解析)"
    dev = job.compile.target_device

    lines = [
        f"# {job.name} — {dev} 部署集成说明",
        "",
        "> 由 torch2tinpu deploy 自动生成; 接口细节以 `artifacts/tvmgen_default.h` 实际内容为准。",
        "",
        "## 1. 文件清单",
        "",
        "| 文件 | 说明 |",
        "|---|---|",
        "| `artifacts/mod.a` | 模型库 (NPU 微码 + CPU 回退算子), 加入 CCS 工程参与链接 |",
        "| `artifacts/tvmgen_default.h` | 模型接口头文件 (I/O 结构体 / 运行函数 / NPU 初始化) |",
        "",
        f"- 模型 I/O: 输入 {shapes_in}; 输出 {shapes_out}",
        f"- 运行位置: {'TI NPU 硬件加速器 (TVMGEN_DEFAULT_TI_NPU)' if hdr['is_tinpu'] else 'CPU (未启用 NPU)'}",
        "",
        "## 2. CCS 集成步骤",
        "",
        ("参考例程: `" + ref + "`") if ref else
        "(未自动找到 C2000Ware 参考例程; 见 C2000Ware 的 `libraries/ai/examples/` 对应器件目录)",
        "",
        "1. 把 `artifacts/` 加入 CCS 工程 (参考例程做法: `mod.a` 作为库文件参与链接)",
        "2. 在工程 `lnk.cmd` 中增加模型数据段 (段名固定; 放置位置参考对应例程):",
        "",
        "   ```",
        "   .rodata.tvm      : > FLASH_BANK0      /* 模型常量 -> Flash */",
        "   .bss.noinit.tvm  : > RAMGS3           /* NPU 数据 -> 全局共享 RAM (实测 RAMGS3 可行; 名称按器件/工程调整) */",
        "   ```",
        "",
        f"3. main 中调用 (字段名来自头文件: `{in_f}` / `{out_f}`):",
        "",
        "   ```c",
        "   TI_NPU_init();                     /* 使能 NPU (含中断配置), 上电后一次 */",
        "   in." + in_f + "  = (void *)input_buf;",
        "   out." + out_f + " = (void *)output_buf;",
        "   tvmgen_default_run(&in, &out);",
        "   while (!tvmgen_default_finished) { }   /* 轮询等待 NPU 完成 */",
        "   ```",
        "",
        "4. 用 golden vectors 做上板对比验证 (参考例程 `test_vector.c` 的组织方式)",
        "",
        "## 3. 编译报告摘要",
        "",
        summary_md if summary_md else "(未找到 compile/summary.md)",
        "",
        "## 4. 常见问题",
        "",
        "- 编译报 `Cross Compiler path is invalid`: 运行 torch2tinpu compile 的终端缺 C2000_CG_ROOT (本工具已自动注入, 属历史坑位)",
        "- 内存不足: 对照上方 mod.a 内存报告调整模型规模或段放置",
        "- 板端结果和 PC 端不一致: 先确认 golden vector 与 Python int8 推理(onnxruntime)一致, 再查板端数据布局/归一化",
        "- NPU 未启用: 确认头文件里有 `#define TVMGEN_DEFAULT_TI_NPU` (没有则表示编译成了 CPU 软库)",
        "",
    ]
    return "\n".join(lines)


def _write_c_template(job, deploy_dir, hdr):
    """生成 <name>_npu.h/.c 调用模板 (单入单出模型). C 文件保持纯 ASCII 注释 (编译器安全)."""
    name_c = re.sub(r"\W", "_", job.name)
    if not name_c or name_c[0].isdigit():
        name_c = "m_" + name_c
    guard = name_c.upper() + "_NPU_H_"
    in_f = hdr["in_fields"][0]
    out_f = hdr["out_fields"][0]
    n_in = hdr["elems_in"][0]
    n_out = hdr["elems_out"][0]

    h_path = os.path.join(deploy_dir, f"{name_c}_npu.h")
    c_path = os.path.join(deploy_dir, f"{name_c}_npu.c")

    h_text = f"""/* Auto-generated by torch2tinpu: {job.name} NPU wrapper template (edit as needed). */
#ifndef {guard}
#define {guard}

#include "tvmgen_default.h"

#ifdef __cplusplus
extern "C" {{
#endif

/* tensor element counts (derived from shapes declared in tvmgen_default.h) */
#define {name_c.upper()}_INPUT_LEN  ({n_in})
#define {name_c.upper()}_OUTPUT_LEN ({n_out})

/* initialize the NPU (call once after power-up) */
void {name_c}_npu_init(void);

/* single inference: float32 input/output buffers, lengths above; returns 0 on success */
int {name_c}_npu_run(const float *in, float *out);

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
    c_text = f"""/* Auto-generated by torch2tinpu: {job.name} NPU wrapper template (edit as needed). */
#include "{name_c}_npu.h"

static struct tvmgen_default_inputs  s_npu_in;
static struct tvmgen_default_outputs s_npu_out;

void {name_c}_npu_init(void)
{{
    TI_NPU_init();
}}

int {name_c}_npu_run(const float *in, float *out)
{{
    s_npu_in.{in_f}  = (void *)in;
    s_npu_out.{out_f} = (void *)out;
    if (tvmgen_default_run(&s_npu_in, &s_npu_out) != 0) {{
        return -1;
    }}
    while (!tvmgen_default_finished) {{ }}
    return 0;
}}
"""
    with open(h_path, "w", encoding="ascii") as f:
        f.write(h_text)
    with open(c_path, "w", encoding="ascii") as f:
        f.write(c_text)
    return h_path, c_path
