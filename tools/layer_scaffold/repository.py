# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import re
from pathlib import Path


class Repository:
    """Read-only queries on the SADL source tree."""

    LAYER_ENUM_FILES = ("sadl/layer.h", "sadl_cuda/layer.h")
    _ENUM_ENTRY = re.compile(r"^\s*(\w+)\s*=\s*(\d+)\s*,", re.M)
    _ONNX_OPS = re.compile(r"onnx_ops\s*=\s*\(([^)]*)\)", re.S)
    _NETRON_ID = re.compile(r"^\s*\w+:\s*\{\s*id:\s*(\d+)\s*,", re.M)
    _PY_ENUM = re.compile(r"^\s+(\w+)\s*=\s*(\d+)\s*$", re.M)

    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root / "sadl" / "layer.h").is_file():
            raise ValueError(f"{self.root} is not a SADL repository (no sadl/layer.h)")

    def path(self, rel):
        return self.root / rel

    def exists(self, rel):
        return self.path(rel).exists()

    def read(self, rel):
        return self.path(rel).read_text(encoding="utf-8")

    # --- layer ids ------------------------------------------------------------
    def operation_type_count(self, rel="sadl/layer.h"):
        """Value of OperationTypeCount, i.e. the next free layer id."""
        if not self.exists(rel):
            return None
        m = re.search(r"OperationTypeCount\s*=\s*(\d+)", self.read(rel))
        return int(m.group(1)) if m else None

    def enum_entries(self, rel):
        """{name: id} of OperationType::Type in a C++ layer.h."""
        if not self.exists(rel):
            return {}
        text = self.read(rel)
        start = text.find("enum Type")
        end = text.find("};", start)
        entries = {m.group(1): int(m.group(2)) for m in self._ENUM_ENTRY.finditer(text[start:end])}
        return {k: v for k, v in entries.items()
                if k != "OperationTypeCount" and not k.startswith("OperationExperimental")}

    def python_op_types(self):
        rel = "converter/onnx2sadl/types/op_type.py"
        return {m.group(1): int(m.group(2)) for m in self._PY_ENUM.finditer(self.read(rel))}

    def netron_ids(self):
        rel = "tools/netron/sadl.js"
        if not self.exists(rel):
            return set()
        text = self.read(rel)
        end = text.find("};", text.find("sadl.OpSpec"))
        return {int(m.group(1)) for m in self._NETRON_ID.finditer(text[:end])}

    # --- converters -----------------------------------------------------------
    def converted_onnx_ops(self):
        """{onnx op type: converter file} of the existing ONNX -> SADL converters."""
        result = {}
        for path in sorted(self.path("converter/onnx2sadl/ops").glob("*_converter.py")):
            m = self._ONNX_OPS.search(path.read_text(encoding="utf-8"))
            if m:
                for op in re.findall(r"[\"']([^\"']+)[\"']", m.group(1)):
                    result[op] = path.name
        return result

    # --- conflicts ------------------------------------------------------------
    def conflicts(self, spec):
        """Messages explaining why ``spec`` cannot be generated (empty if none)."""
        errors = []
        cpu = self.enum_entries("sadl/layer.h")
        if spec.name in cpu:
            errors.append(f"{spec.name} already exists in sadl/layer.h (id {cpu[spec.name]})")
        count = self.operation_type_count()
        if spec.sadl_id != count:
            errors.append(f"the SADL id must be the next free one ({count}, OperationTypeCount in sadl/layer.h)")
        if spec.sadl_id in cpu.values():
            errors.append(f"id {spec.sadl_id} is already used in sadl/layer.h")
        py = self.python_op_types()
        if spec.name in py:
            errors.append(f"{spec.name} already exists in converter/onnx2sadl/types/op_type.py")
        if spec.sadl_id in py.values():
            errors.append(f"id {spec.sadl_id} is already used in converter/onnx2sadl/types/op_type.py")
        if spec.cuda != "skip":
            cuda = self.enum_entries("sadl_cuda/layer.h")
            if spec.name in cuda or spec.sadl_id in cuda.values():
                errors.append(f"{spec.name} or id {spec.sadl_id} already exists in sadl_cuda/layer.h")
            cuda_count = self.operation_type_count("sadl_cuda/layer.h")
            if cuda_count != spec.sadl_id:
                errors.append(f"sadl_cuda/layer.h is out of sync (OperationTypeCount = {cuda_count}, "
                              f"expected {spec.sadl_id}): fix it or answer 'skip' for CUDA")
        if spec.netron and spec.sadl_id in self.netron_ids():
            errors.append(f"id {spec.sadl_id} is already used in tools/netron/sadl.js")
        handled = self.converted_onnx_ops()
        for op in spec.onnx_ops:
            if op in handled:
                errors.append(f"ONNX op {op} is already converted by ops/{handled[op]}")
        return errors

    # --- licence header -------------------------------------------------------
    def cpp_license_header(self):
        """The BSD comment block of an existing layer header (reused for new ones)."""
        for rel in ("sadl/layer_abs.h", "sadl/layer_floor.h"):
            if self.exists(rel):
                text = self.read(rel).replace("\r\n", "\n")
                end = text.find("*/")
                if text.startswith("/*") and end > 0:
                    return text[:end + 2] + "\n"
        return "// SPDX-License-Identifier: BSD-3-Clause\n"
