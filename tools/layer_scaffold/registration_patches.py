# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import re

from .patch import Patch


class RegistrationPatches:
    """Every insertion into existing files needed to register a new layer.

    Each patch finds an anchor (a regular expression on a line) and inserts the
    new code next to the entries of the same kind. A patch whose change is
    already there does nothing; a patch whose anchor is missing becomes a manual
    step (see Patch).
    """

    def __init__(self, spec, cpp, py, netron):
        self.spec = spec
        self.cpp = cpp
        self.py = py
        self.netron = netron

    def build(self):
        s = self.spec
        patches = [
            self._enum("sadl/layer.h", "1.1"),
            self._include("sadl/layers.h", "1.3"),
            self._case_print("sadl/layers.h", "1.3"),
            self._model_case(),
            self._copy_case(),
        ]
        if s.has_payload:
            patches.append(self._dump())
        patches.append(self._quantization())
        if s.cuda != "skip":
            patches += [
                self._enum("sadl_cuda/layer.h", "3.1"),
                self._include("sadl_cuda/layers.h", "3.1"),
                self._case_print("sadl_cuda/layers.h", "3.1"),
                self._cuda_model_case(),
                self._cuda_readme(),
            ]
        if s.netron:
            patches.append(self._netron_op_spec())
            if s.has_payload:
                patches.append(self._netron_decoder())
        patches += [
            self._op_type(),
            self._catalog_import(),
            self._catalog_entry(),
        ]
        if s.serializer_kind != "none":
            if s.serializer_kind == "custom":
                patches.append(self._import(
                    "converter/onnx2sadl/writer/layer_serializer_registry.py",
                    f"from .{s.py_name}_serializer import {s.name}Serializer", "4.5"))
            patches.append(self._dict_entry(
                "converter/onnx2sadl/writer/layer_serializer_registry.py", self.py.serializer_entry(), "4.5"))
            patches.append(self._dict_entry(
                "converter/sadl2onnx/reader/layer_params_reader_registry.py", self.py.reader_entry(), "5.1"))
        if s.needs_emitter_file:
            patches.append(self._import(
                "converter/sadl2onnx/emitters/emitter_registry.py",
                f"from .{s.py_name}_emitter import {s.name}Emitter", "5.2"))
        patches.append(self._emitter_entry())
        if s.test_model:
            patches.append(self._check_sh())
        return patches

    # --- helpers --------------------------------------------------------------
    @staticmethod
    def _make(rel, description, present_pattern, apply, manual, section):
        return Patch(rel, description, lambda sf: sf.contains(present_pattern), apply, manual, section)

    @staticmethod
    def _indent(line):
        return re.match(r"^(\s*)", line).group(1)

    # --- C++ ------------------------------------------------------------------
    def _enum(self, rel, section):
        name, sid = self.spec.name, self.spec.sadl_id

        def apply(sf):
            i = sf.find_first(r"^\s*OperationTypeCount\s*=\s*\d+")
            if i is None:
                return False
            m = re.match(r"^(\s*)(OperationTypeCount\s*)=\s*(\d+)(.*)$", sf.line(i))
            if not m or int(m.group(3)) != sid:
                return False
            width = len(m.group(2))
            label = name.ljust(width) if len(name) < width else name + " "
            sf.replace(i, f"{m.group(1)}{m.group(2)}= {sid + 1}{m.group(4)}")
            sf.insert(i, [f"{m.group(1)}{label}= {sid},"])
            return True

        return self._make(rel, f"OperationType::{name} = {sid}, OperationTypeCount = {sid + 1}",
                          rf"^\s*{name}\s*=\s*\d+", apply,
                          [f"    {name} = {sid},", f"    OperationTypeCount = {sid + 1},"], section)

    def _include(self, rel, section):
        line = f'#include "{self.spec.cpp_header}"'

        def apply(sf):
            i = sf.find_last(r'^#include "layer_')
            if i is None:
                return False
            sf.insert(i + 1, [line])
            return True

        return self._make(rel, line, re.escape(line), apply, [line], section)

    def _case_print(self, rel, section):
        name = self.spec.name

        def apply(sf):
            i = sf.find_last(r"^\s*DIRTYCASEPRINT\((?!FusedMatMulAdaptQuant\))\w+\);")
            if i is None:
                return False
            sf.insert(i + 1, [f"{self._indent(sf.line(i))}DIRTYCASEPRINT({name});"])
            return True

        return self._make(rel, f"DIRTYCASEPRINT({name}) in opName()", rf"DIRTYCASEPRINT\({name}\)",
                          apply, [f"    DIRTYCASEPRINT({name});"], section)

    def _model_case(self):
        name = self.spec.name

        def apply(sf):
            i = sf.find_first(r"^\s*case layers::OperationType::OperationTypeCount:")
            if i is None:
                return False
            sf.insert(i, self.cpp.model_case(self._indent(sf.line(i))))
            return True

        return self._make("sadl/model.h", "layer factory case in createLayer()",
                          rf"OperationType::{name}:", apply, self.cpp.model_case("  "), "1.3")

    def _cuda_model_case(self):
        name = self.spec.name

        def apply(sf):
            i = sf.find_first(r"^\s*case OperationType::OperationTypeCount:")
            if i is None:
                return False
            width = 38
            j = sf.find_last(r"^\s*case OperationType::\w+:\s*return", 0, i)
            if j is not None:
                width = len(re.match(r"^\s*(case OperationType::\w+:\s*)return", sf.line(j)).group(1))
            label_len = len(f"case OperationType::{name}:")
            sf.insert(i, self.cpp.cuda_model_case(self._indent(sf.line(i)), max(width, label_len + 1)))
            return True

        return self._make("sadl_cuda/model.h", "layer factory case in createLayer()",
                          rf"OperationType::{name}:", apply, self.cpp.cuda_model_case("  ", 38), "3.1")

    def _copy_case(self):
        name = self.spec.name

        def apply(sf):
            i = sf.find_first(r"^\s*case sadl::layers::OperationType::OperationExperimentalStart:")
            if i is None:
                return False
            sf.insert(i, self.cpp.copy_case())
            return True

        return self._make("sample/copy.h", "case copying the parameters", rf"OperationType::{name}:",
                          apply, self.cpp.copy_case(), "2.1")

    def _dump(self):
        name = self.spec.name

        def apply(sf):
            i = sf.find_first(r"^template<typename T> bool sadl::layers::Layer<T>::dump")
            if i is None:
                return False
            sf.insert(i, self.cpp.dump_function())
            return True

        return self._make("sample/dumper.h", f"{name}<T>::dump() writing the parameters",
                          rf"sadl::layers::{name}<T>::dump", apply, self.cpp.dump_function(), "2.2")

    def _quantization(self):
        name = self.spec.name
        rel = "sample/naive_quantization.cpp"
        if self.spec.internal_quantizer:
            def apply(sf):
                i = sf.find_first(r"else if \(layerQ\.op\(\) == sadl::layers::OperationType::Placeholder\)")
                if i is None:
                    return False
                indent = self._indent(sf.line(i))
                sf.insert(i, [
                    f"{indent}else if (layerQ.op() == sadl::layers::OperationType::{name})",
                    f"{indent}  dynamic_cast<sadl::layers::{name}<T> &>(layerQ).m_q = quantizer;",
                ])
                return True

            manual = [f"  else if (layerQ.op() == sadl::layers::OperationType::{name})",
                      f"    dynamic_cast<sadl::layers::{name}<T> &>(layerQ).m_q = quantizer;"]
            return self._make(rel, "quantize(): set the internal quantizer m_q",
                              rf"OperationType::{name}\b", apply, manual, "2.3")

        def apply_no_q(sf):
            start = sf.find_first(r"bool toQuantize\(")
            if start is None:
                return False
            end = sf.find_first(r"^}", start)
            i = sf.find_last(r"type != sadl::layers::OperationType::\w+;\s*$", start, end)
            if i is None:
                return False
            sf.replace(i, re.sub(r";\s*$", "", sf.line(i)))
            sf.insert(i + 1, [f"         && type != sadl::layers::OperationType::{name};"])
            return True

        return self._make(rel, "toQuantize(): no quantizer needed", rf"OperationType::{name}\b",
                          apply_no_q, [f"         && type != sadl::layers::OperationType::{name}"], "2.3")

    def _cuda_readme(self):
        s = self.spec
        execution = "✓" if s.cuda == "gpu" else "CPU"
        row = f"| {s.name} | {execution} | {s.marker}: describe |"

        def apply(sf):
            start = sf.find_first(r"^## Layer support")
            if start is None:
                return False
            head = sf.find_first(r"^\| Layer", start)
            if head is None:
                return False
            rows = []
            i = head + 2  # header + separator
            while i < len(sf.lines) and sf.line(i).startswith("|"):
                rows.append(i)
                i += 1
            if not rows:
                return False
            same = [r for r in rows if sf.line(r).split("|")[2].strip() == execution]
            sf.insert((same or rows)[-1] + 1, [row])
            return True

        return self._make("sadl_cuda/README.md", "row of the Layer support table", rf"^\| {s.name} \|",
                          apply, [row], "3.1")

    # --- Netron ---------------------------------------------------------------
    def _netron_op_spec(self):
        s = self.spec

        def apply(sf):
            i = sf.find_first(rf"^\s*\w+:\s*\{{\s*id:\s*{s.sadl_id - 1}\s*,")
            if i is None:
                return False
            sf.insert(i + 1, [self.netron.op_spec_entry(self._indent(sf.line(i)))])
            return True

        return self._make("tools/netron/sadl.js", "sadl.OpSpec entry", rf"^\s*{s.name}:\s*\{{",
                          apply, [self.netron.op_spec_entry()], "3.2")

    def _netron_decoder(self):
        s = self.spec

        def apply(sf):
            i = sf.find_first(r"^// passthrough")
            if i is None:
                return False
            sf.insert(i, self.netron.decoder())
            return True

        return self._make("tools/netron/sadl.js", "payload decoder", rf"dec\[sadl\.OpCode\.{s.name}\]",
                          apply, self.netron.decoder(), "3.2")

    # --- Python ---------------------------------------------------------------
    def _op_type(self):
        s = self.spec

        def apply(sf):
            i = sf.find_first(rf"^\s+\w+ = {s.sadl_id - 1}\s*$")
            if i is None:
                return False
            sf.insert(i + 1, [f"{self._indent(sf.line(i))}{s.name} = {s.sadl_id}"])
            return True

        return self._make("converter/onnx2sadl/types/op_type.py", f"OpType.{s.name} = {s.sadl_id}",
                          rf"^\s+{s.name} = \d+", apply, [f"    {s.name} = {s.sadl_id}"], "4.1")

    def _catalog_import(self):
        s = self.spec
        return self._import("converter/onnx2sadl/ops/op_catalog.py",
                            f"from .{s.py_name}_converter import {s.name}Converter", "4.4")

    def _catalog_entry(self):
        s = self.spec
        line = f"        {s.name}Converter,"

        def apply(sf):
            start = sf.find_first(r"^\s*CONVERTERS\s*=\s*\(")
            if start is None:
                return False
            end = sf.find_first(r"^\s*\)\s*$", start)
            return sf.insert_sorted(r"^\s+\w+Converter,\s*$", line, lambda x: x.strip(), start, end)

        return self._make("converter/onnx2sadl/ops/op_catalog.py", "OpCatalog.CONVERTERS entry",
                          rf"^\s+{s.name}Converter,", apply, [line], "4.4")

    def _import(self, rel, line, section):
        def apply(sf):
            return sf.insert_sorted(r"^from \.\w+ import \w+\s*$", line, lambda x: x.split()[1])

        return self._make(rel, line, rf"^{re.escape(line)}\s*$", apply, [line], section)

    def _dict_entry(self, rel, entry, section):
        s = self.spec

        def apply(sf):
            start = sf.find_first(r"self\._by_op\s*=\s*\{")
            if start is None:
                return False
            end = sf.find_first(r"^\s*\}\s*$", start)
            if end is None or end == start + 1:
                return False
            sf.insert(end, [self._indent(sf.line(start + 1)) + entry])
            return True

        return self._make(rel, entry, rf"OpType\.{s.name}:", apply, ["            " + entry], section)

    def _emitter_entry(self):
        s = self.spec
        entry = self.py.emitter_entry()

        def apply(sf):
            start = sf.find_first(r"^\s*emitters\s*=\s*\[")
            if start is None:
                return False
            end = sf.find_first(r"^\s*\]\s*$", start)
            if end is None or end == start + 1:
                return False
            sf.insert(end, [self._indent(sf.line(start + 1)) + entry])
            return True

        return self._make("converter/sadl2onnx/emitters/emitter_registry.py", entry,
                          rf"OpType\.{s.name}\b|\b{s.name}Emitter\(\)", apply, ["            " + entry], "5.2")

    # --- tests ----------------------------------------------------------------
    def _check_sh(self):
        s = self.spec
        line = (f"# {s.marker}: add {s.py_name} to L once utests/models/{s.py_name}.onnx "
                "exists (CONTRIBUTE.md 6.2)")

        def apply(sf):
            i = sf.find_last(r"^for F in \$L; do")  # the last list: models converted with transposes
            if i is None:
                return False
            sf.insert(i, [line])
            return True

        return self._make("utests/check.sh", "reminder to list the test model", re.escape(s.marker),
                          apply, [line], "6.2")
