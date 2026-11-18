# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .optimization_pass import OptimizationPass


class FunctionPass(OptimizationPass):
    """Wraps a plain function ``model -> model`` (or ``model -> None`` if it works in place)."""

    def __init__(self, function, name=None, description=""):
        self.function = function
        self.name = name or getattr(function, "__name__", "function_pass")
        self.description = description or (function.__doc__ or "").strip().split("\n")[0]

    def run(self, model):
        result = self.function(model)
        return model if result is None else result
