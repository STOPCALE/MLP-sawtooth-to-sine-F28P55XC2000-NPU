# -*- coding: utf-8 -*-
"""job.yaml 的加载 / 默认值 / 校验.

job.yaml 是每个模型的"转换任务书", 全部字段都有默认值,
最小配置只需要 name + adapter.path (其余按默认即可跑通 QAT->F28P55 全流程).

字段速查 (完整参考见仓库 README):
  name            输出名 / 编译 run 名 (必须纯 ASCII: TI 工具链按 GBK 读文件)
  out_dir         输出根目录 (相对 job.yaml 所在目录), 实际产物在 <out_dir>/<name>/
  seed            全局随机种子
  adapter.path    adapter.py 路径 (相对 job.yaml 所在目录)
  quant.*         量化参数 (method/bitwidth/epochs/learning_rate/...)
  verify.*        内建验收 (float vs int8 对比)
  compile.*       编译参数 (target_device/task_type/...)
  deploy.*        部署包生成 (说明文档 + C 模板)
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml

from . import ToolError

VALID_METHODS = ("qat", "ptq")
VALID_BITWIDTHS = (2, 4, 8)
VALID_OPTIMIZERS = ("adam", "sgd", "adamw")
VALID_SCHEDULERS = ("cosine", "none")
VALID_LOSSES = ("mse", "l1", "huber")

_ROOT_KEYS = {"name", "out_dir", "seed", "adapter", "quant", "verify", "compile", "deploy"}


def _abspath(path, base):
    if path is None or path == "":
        return path
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return os.path.abspath(os.path.normpath(path))


def _ascii_ok(s):
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _fill(dc_cls, data, section):
    """把 YAML 字典填进 dataclass; 未知字段直接报错 (防拼写错误静默失效)."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ToolError(f"job 配置段 '{section}' 必须是字典")
    known = set(dc_cls.__dataclass_fields__.keys())
    unknown = sorted(set(data) - known)
    if unknown:
        raise ToolError(f"job 配置段 '{section}' 存在未知字段 {unknown}; 可用字段: {sorted(known)}")
    return dc_cls(**data)


@dataclass
class QuantConfig:
    method: str = "qat"              # qat (带微调) | ptq (仅校准)
    weight_bitwidth: int = 8         # 2 / 4 / 8
    activation_bitwidth: int = 8     # 2 / 4 / 8
    output_int: bool = False         # True=模型输出保留 int8; False=输出反量化为 float
    opset_version: int = 18          # 勿改: 与 TI 官方流程一致 (影响 FC 的 Gemm 融合)
    total_epochs: int = 15           # QAT 微调轮数 / PTQ 校准遍数
    iters_per_epoch: int = 200       # 每轮迭代批次数
    batch_size: int = 256
    learning_rate: float = 2e-4
    optimizer: str = "adam"          # adam / sgd / adamw
    scheduler: str = "cosine"        # cosine / none
    loss: str = "mse"                # mse / l1 / huber
    device: str = "auto"             # auto / cpu / cuda (QAT 训练设备)
    verbose: bool = True             # TI 包装器的详细日志
    num_observer_update_epochs: Optional[int] = None   # None=包装器默认 (半程冻结)
    num_batch_norm_update_epochs: Optional[int] = None


@dataclass
class VerifyConfig:
    enabled: bool = True
    num_samples: int = 200           # 验收样本数 (float vs int8 输出对比)


@dataclass
class CompileConfig:
    enabled: bool = True
    target_device: str = "F28P55"
    target_module: str = "timeseries"            # timeseries / image / audio / radar
    task_type: str = "generic_timeseries_regression"
    model_path: Optional[str] = None             # 默认本 job 的 int8 产物; 也可指向外部 TI-NPU 格式 ONNX
    dataset_name: Optional[str] = None           # modelmaker 工程目录名; 默认 = name
    quantization_version: int = 2                # 2 = TI-NPU 格式 (hard 模式必需, 勿改)
    toolchain_root: Optional[str] = None         # C2000 编译器根目录; 默认自动探测
    extra_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeployConfig:
    enabled: bool = True
    generate_c_template: bool = True             # 生成 <name>_npu.h/.c 调用模板


@dataclass
class JobConfig:
    path: str                                    # job.yaml 绝对路径
    job_dir: str                                 # job.yaml 所在目录
    name: str
    out_dir: str                                 # 输出根目录 (绝对)
    seed: int = 0
    adapter_path: str = ""
    quant: QuantConfig = field(default_factory=QuantConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)

    @property
    def run_dir(self):
        """本任务的产物目录: <out_dir>/<name>/"""
        return os.path.join(self.out_dir, self.name)

    @property
    def int8_onnx(self):
        return os.path.join(self.run_dir, self.name + "_int8.onnx")

    def to_dict(self):
        d = dataclasses.asdict(self)
        d.pop("path", None)
        d.pop("job_dir", None)
        return d

    @classmethod
    def load(cls, path):
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise ToolError(f"job 文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ToolError(f"job 文件顶层必须是字典: {path}")
        unknown = sorted(set(raw) - _ROOT_KEYS)
        if unknown:
            raise ToolError(f"job 顶层存在未知字段 {unknown}; 可用字段: {sorted(_ROOT_KEYS)}")

        job_dir = os.path.dirname(path)

        name = raw.get("name")
        if not name or not isinstance(name, str):
            raise ToolError("job 缺少 'name' 字段 (输出目录名 / 编译 run 名)")
        if not _ascii_ok(name):
            raise ToolError(f"name 必须为纯 ASCII (TI 工具链按 GBK 读文件, 中文会崩): {name!r}")

        adapter = raw.get("adapter")
        if not isinstance(adapter, dict) or not adapter.get("path"):
            raise ToolError("job 缺少 'adapter.path' 字段 (adapter.py 路径)")
        unknown = sorted(set(adapter) - {"path"})
        if unknown:
            raise ToolError(f"job 配置段 'adapter' 存在未知字段 {unknown}; 可用字段: ['path']")
        adapter_path = _abspath(adapter["path"], job_dir)
        if not os.path.isfile(adapter_path):
            raise ToolError(f"adapter.path 指向的文件不存在: {adapter_path}")

        out_dir = _abspath(raw.get("out_dir") or "out", job_dir)

        cfg = cls(path=path, job_dir=job_dir, name=name, out_dir=out_dir,
                  seed=int(raw.get("seed", 0)), adapter_path=adapter_path)
        cfg.quant = _fill(QuantConfig, raw.get("quant"), "quant")
        cfg.verify = _fill(VerifyConfig, raw.get("verify"), "verify")
        cfg.compile = _fill(CompileConfig, raw.get("compile"), "compile")
        cfg.deploy = _fill(DeployConfig, raw.get("deploy"), "deploy")

        cls._validate(cfg)
        return cfg

    @staticmethod
    def _validate(cfg):
        q = cfg.quant
        if q.method not in VALID_METHODS:
            raise ToolError(f"quant.method 无效: {q.method!r} (可用: {VALID_METHODS})")
        for key in ("weight_bitwidth", "activation_bitwidth"):
            if getattr(q, key) not in VALID_BITWIDTHS:
                raise ToolError(f"quant.{key} 无效: {getattr(q, key)} (可用: {VALID_BITWIDTHS})")
        if q.optimizer not in VALID_OPTIMIZERS:
            raise ToolError(f"quant.optimizer 无效: {q.optimizer!r} (可用: {VALID_OPTIMIZERS})")
        if q.scheduler not in VALID_SCHEDULERS:
            raise ToolError(f"quant.scheduler 无效: {q.scheduler!r} (可用: {VALID_SCHEDULERS})")
        if q.loss not in VALID_LOSSES:
            raise ToolError(f"quant.loss 无效: {q.loss!r} (可用: {VALID_LOSSES})")
        if q.total_epochs < 1 or q.iters_per_epoch < 1 or q.batch_size < 1:
            raise ToolError("quant.total_epochs / iters_per_epoch / batch_size 必须 >= 1")
        if q.opset_version < 17:
            raise ToolError(f"quant.opset_version 无效: {q.opset_version} (官方流程为 18)")

        c = cfg.compile
        if c.quantization_version != 2:
            raise ToolError(
                "compile.quantization_version 必须为 2 (= TI-NPU 格式)。\n"
                "这不是可调参数: 其它值会让编译流程把 target 从 type=hard 降级为 type=soft "
                "(CPU 软库, 没有 NPU 加速), 属于流程防呆, 请直接删除该字段使用默认值。"
            )
        if c.dataset_name is None:
            c.dataset_name = cfg.name
        if not _ascii_ok(c.dataset_name):
            raise ToolError(f"compile.dataset_name 必须为纯 ASCII: {c.dataset_name!r}")
        if c.model_path is not None:
            c.model_path = _abspath(c.model_path, cfg.job_dir)
        if c.toolchain_root is not None:
            c.toolchain_root = _abspath(c.toolchain_root, cfg.job_dir)
