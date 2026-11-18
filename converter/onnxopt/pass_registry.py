# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.errors.conversion_error import ConversionError

from .constant_to_initializer_pass import ConstantToInitializerPass
from .function_pass import FunctionPass
from .onnxsim_pass import OnnxSimPass
from .optimization_pass import OptimizationPass
from .remove_identity_pass import RemoveIdentityPass
from .shape_inference_pass import ShapeInferencePass
from .split_slices_pass import SplitSlicesPass


class PassRegistry:
    """Name -> pass factory. Built-in passes are registered; users add their own with ``register``."""

    BUILTIN = (OnnxSimPass, SplitSlicesPass, RemoveIdentityPass, ConstantToInitializerPass, ShapeInferencePass)
    # split_slices undoes onnxsim's Slice fusion, which SADL cannot convert
    DEFAULT_PASSES = ("onnxsim", "split_slices")

    def __init__(self):
        self._factories = {}
        for cls in self.BUILTIN:
            self.register(cls)

    def register(self, factory, name=None):
        """``factory``: an OptimizationPass subclass, an instance, or a function ``model -> model``."""
        if isinstance(factory, type) and issubclass(factory, OptimizationPass):
            key = name or factory.name
        elif isinstance(factory, OptimizationPass):
            instance = factory
            key = name or instance.name
            factory = lambda **_: instance  # noqa: E731
        elif callable(factory):
            function = factory
            key = name or function.__name__
            factory = lambda **_: FunctionPass(function, key)  # noqa: E731
        else:
            raise ConversionError("[ERROR] cannot register {!r} as an optimization pass".format(factory))
        if not key:
            raise ConversionError("[ERROR] optimization pass without a name: {!r}".format(factory))
        self._factories[key] = factory
        return key

    def create(self, name, **options):
        if name not in self._factories:
            raise ConversionError(
                "[ERROR] unknown optimization pass {!r}; available: {}".format(name, ", ".join(self.names()))
            )
        return self._factories[name](**options)

    def names(self):
        return sorted(self._factories)

    def describe(self):
        lines = []
        for name in self.names():
            factory = self._factories[name]
            doc = getattr(factory, "description", "") if isinstance(factory, type) else ""
            default = " (default)" if name in self.DEFAULT_PASSES else ""
            lines.append("  {:24s}{}{}".format(name, doc, default))
        return "\n".join(lines)
