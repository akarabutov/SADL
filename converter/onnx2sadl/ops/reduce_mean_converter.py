# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from ..types.op_type import OpType
from .unary_converter import UnaryConverter


class ReduceMeanConverter(UnaryConverter):
    """ReduceMean computed in NCHW (input transposed back), keepdims required."""

    onnx_ops = ("ReduceMean",)
    sadl_op = OpType.ReduceMean

    def layer_attrs(self, node):
        keepdims = self.attribute(node, "keepdims")
        if keepdims.i == 0:
            raise ConversionError("[ERROR] Currently, the value of keepdims in ReduceMean can not be set to 0.")
        return {"keepdims": keepdims.i, "axes": self.attribute(node, "axes").ints}

    def annotate(self, annotator, node, data_layout):
        if annotator.global_layout == "nchw":
            annotation = annotator.annotations[node.name]
            annotation.transpose_before_in0 = "nchw"
            annotation.to_nhwc_after_out = True
