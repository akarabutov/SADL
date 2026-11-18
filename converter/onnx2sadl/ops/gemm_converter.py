# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .static_weight_converter import StaticWeightConverter


class GemmConverter(StaticWeightConverter):
    """Gemm -> MatMul (+ BiasAdd)."""

    onnx_ops = ("Gemm",)
    sadl_op = OpType.MatMul

    def annotate(self, annotator, node, data_layout):
        weight = annotator.model.initializer(node.input[1])
        if annotator.global_layout == "nchw":
            annotator.annotations[weight.name].to_transpose = True
