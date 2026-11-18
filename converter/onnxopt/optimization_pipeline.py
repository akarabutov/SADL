# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import copy
from collections import Counter

import onnx

from onnx2sadl.errors.conversion_error import ConversionError

from .function_pass import FunctionPass
from .optimization_pass import OptimizationPass
from .pass_registry import PassRegistry


class OptimizationPipeline:
    """Runs a list of OptimizationPass in order on an ONNX model.

    After each pass the model is validated with ``onnx.checker``; a failing
    pass raises, or is skipped with a warning when ``continue_on_error``.
    """

    def __init__(self, passes=None, check=True, continue_on_error=False, verbose=0):
        self.passes = []
        self.check = check
        self.continue_on_error = continue_on_error
        self.verbose = verbose
        for p in passes or []:
            self.add(p)

    @classmethod
    def from_names(cls, names=None, registry=None, pass_options=None, **kwargs):
        """Build from pass names (default: PassRegistry.DEFAULT_PASSES).

        ``pass_options``: {pass name: {option: value}} given to the pass constructors.
        """
        registry = registry or PassRegistry()
        names = PassRegistry.DEFAULT_PASSES if names is None else names
        pass_options = pass_options or {}
        return cls([registry.create(n, **pass_options.get(n, {})) for n in names], **kwargs)

    def add(self, optimization_pass):
        """Append a pass (an OptimizationPass or a function ``model -> model``)."""
        if not isinstance(optimization_pass, OptimizationPass):
            if not callable(optimization_pass):
                raise ConversionError("[ERROR] not an optimization pass: {!r}".format(optimization_pass))
            optimization_pass = FunctionPass(optimization_pass)
        self.passes.append(optimization_pass)
        return self

    def run(self, model):
        """Returns the optimized model; ``model`` itself is left untouched."""
        model = copy.deepcopy(model)
        if self.verbose:
            print("[INFO] ONNX optimization: {} ({} nodes)".format(
                ", ".join(p.name for p in self.passes) or "no pass", len(model.graph.node)))
        for optimization_pass in self.passes:
            before = model
            try:
                result = optimization_pass.run(copy.deepcopy(model))
                if self.check:
                    onnx.checker.check_model(result)
            except Exception as e:
                if not self.continue_on_error:
                    raise ConversionError("[ERROR] optimization pass {} failed: {}".format(optimization_pass.name, e))
                print("[WARNING] optimization pass {} failed, skipped: {}".format(optimization_pass.name, e))
                model = before
                continue
            if self.verbose:
                self._report(optimization_pass, before, result)
            model = result
        return model

    @staticmethod
    def _report(optimization_pass, before, after):
        count_before = Counter(n.op_type for n in before.graph.node)
        count_after = Counter(n.op_type for n in after.graph.node)
        changes = []
        for op in sorted(set(count_before) | set(count_after)):
            delta = count_after[op] - count_before[op]
            if delta:
                changes.append("{}{:+d}".format(op, delta))
        print("[INFO]   {}: {} -> {} nodes {}".format(
            optimization_pass.name, len(before.graph.node), len(after.graph.node), " ".join(changes)))
