# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .unary_converter import UnaryConverter


class ShapeConverter(UnaryConverter):
    onnx_ops = ("Shape",)
    sadl_op = OpType.Shape

    def annotate(self, annotator, node, data_layout):
        annotator.annotations[node.name].layout_onnx = None
