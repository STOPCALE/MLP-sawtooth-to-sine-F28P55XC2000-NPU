# -*- coding: utf-8 -*-
"""诊断: 对比 mine vs ref 两个 int8 ONNX 的图结构 (找 NPU 无法识别的差异)"""
import os
import sys
from collections import Counter

import onnx

BASE = os.path.dirname(os.path.abspath(__file__))
FILES = [
    ("MINE (saw2sin)", BASE + r"\saw2sin_tinpu_int8.onnx"),
    ("REF  (regr2k)", BASE + r"\ref_regr2k_model.onnx"),
]

for tag, path in FILES:
    print("=" * 25, tag, "=" * 25)
    m = onnx.load(path)
    print("ir_version:", m.ir_version, " opsets:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    for vi in m.graph.input:
        print("  input :", vi.name, [d.dim_value for d in vi.type.tensor_type.shape.dim])
    for vi in m.graph.output:
        print("  output:", vi.name, [d.dim_value for d in vi.type.tensor_type.shape.dim])
    print("  op counts:", dict(Counter(n.op_type for n in m.graph.node)))
    print("  domains  :", dict(Counter((n.domain or "<default>") for n in m.graph.node)))
    inits = {i.name for i in m.graph.initializer}
    print("  #initializers:", len(inits))
    for n in m.graph.node:
        if n.op_type in ("MatMul", "Gemm", "Conv"):
            print("   ", n.op_type, "-> inputs are initializer:", [i in inits for i in n.input], "|", n.name[:40])
    print("  -- nodes --")
    for n in m.graph.node:
        attrs = []
        for a in n.attribute:
            if a.name in ("scale", "zero_point", "axis", "transB", "perm"):
                v = list(a.floats) if a.floats else (list(a.ints) if a.ints else a.i)
                attrs.append(f"{a.name}={v}")
        print(f"    {n.op_type:24s} {n.name[:40]:42s} in={[i[:20] for i in n.input]} {','.join(attrs)}")
    print()
