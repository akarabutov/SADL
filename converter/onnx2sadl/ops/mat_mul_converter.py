# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .binary_converter import BinaryConverter


class MatMulConverter(BinaryConverter):
    onnx_ops = ("MatMul",)
    sadl_op = OpType.MatMul
    const_to_transpose = "producer"
    reject_two_constants = True
    const_first_swaps = True

    def annotate(self, annotator, node, data_layout):
        if annotator.global_layout != "nchw":
            return
        annotation = annotator.annotations[node.name]
        dim0 = annotator.model.declared_dims(node.input[0])
        dim1 = annotator.model.declared_dims(node.input[1])
        # len == 0 is assumed to be the output of a computing node (conv), i.e. 4-D
        if len(dim0) == 4 or len(dim0) == 0:
            annotation.transpose_before_in0 = "nchw"
            annotation.to_nhwc_after_out = True
        if len(dim1) == 4 or len(dim1) == 0:
            annotation.transpose_before_in1 = "nchw"
