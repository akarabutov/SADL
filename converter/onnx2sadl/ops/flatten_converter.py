# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .unary_converter import UnaryConverter


class FlattenConverter(UnaryConverter):
    onnx_ops = ("Flatten",)
    sadl_op = OpType.Flatten

    def layer_attrs(self, node):
        return {"axis": self.attribute(node, "axis").i}

    def annotate(self, annotator, node, data_layout):
        annotation = annotator.annotations[node.name]
        if annotation.layout_onnx != "nchw":
            return
        a = self.attribute(node, "axis")
        dims = annotator.model.try_shape(node.input[0])
        need_transpose = True
        if a is not None and a.i == 1 and dims is not None and len(dims) == 4:
            c, h, w = dims[1], dims[2], dims[3]
            if c == 1 or (h == 1 and w == 1):  # flattening order does not matter
                need_transpose = False
        if need_transpose:
            annotation.transpose_before_in0 = "nchw"
