# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import onnx

from .optimization_pass import OptimizationPass


class ShapeInferencePass(OptimizationPass):
    """Adds the inferred intermediate shapes (value_info) to the model."""

    name = "shape_inference"
    description = "ONNX shape inference (adds value_info)"

    def run(self, model):
        return onnx.shape_inference.infer_shapes(model)
