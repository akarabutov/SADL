# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .static_weight_converter import StaticWeightConverter


class ConvTransposeConverter(StaticWeightConverter):
    """ConvTranspose -> Conv2DTranspose (+ BiasAdd)."""

    onnx_ops = ("ConvTranspose",)
    sadl_op = OpType.Conv2DTranspose

    def layer_attrs(self, node):
        pads = self.attribute(node, "pads")
        output_padding = self.attribute(node, "output_padding")
        return {
            "strides": self.attribute(node, "strides").ints,
            "pads": pads.ints if pads else [0, 0, 0, 0],
            "output_padding": output_padding.ints if output_padding else [0, 0],
        }

    def annotate(self, annotator, node, data_layout):
        weight = annotator.model.initializer(node.input[1])
        annotator.annotations[weight.name].to_transpose = True
        annotator.annotations[weight.name].layout_onnx = "nhwc"
