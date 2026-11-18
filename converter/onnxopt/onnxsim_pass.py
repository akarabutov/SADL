# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import inspect

from onnx2sadl.errors.conversion_error import ConversionError

from .optimization_pass import OptimizationPass


class OnnxSimPass(OptimizationPass):
    """onnx-simplifier (``pip install onnxsim``): constant folding, shape
    inference and graph rewrites that remove the clutter of exporters.

    Initializers are kept as initializers (the SADL converter reads weights
    from there). Extra keyword arguments are passed to ``onnxsim.simplify``
    when the installed version supports them.
    """

    name = "onnxsim"
    description = "onnx-simplifier: constant folding and graph simplification"

    def __init__(self, input_shapes=None, check_n=0, skip_fuse_bn=False, **simplify_kwargs):
        self.input_shapes = input_shapes
        self.options = dict(check_n=check_n, skip_fuse_bn=skip_fuse_bn, initializers_as_constants=False)
        self.options.update(simplify_kwargs)

    def run(self, model):
        try:
            import onnxsim
        except ImportError:
            raise ConversionError("[ERROR] the onnxsim pass needs onnx-simplifier: pip install onnxsim")

        options = dict(self.options)
        if self.input_shapes:
            options["overwrite_input_shapes"] = self.input_shapes
        supported = inspect.signature(onnxsim.simplify).parameters
        options = {k: v for k, v in options.items() if k in supported}

        simplified, ok = onnxsim.simplify(model, **options)
        if not ok:
            raise ConversionError("[ERROR] onnxsim: the simplified model does not match the original")
        return simplified
