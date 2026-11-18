# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .unary_converter import UnaryConverter


class DepthToSpaceConverter(UnaryConverter):
    onnx_ops = ("DepthToSpace",)
    sadl_op = OpType.Depth2Space

    def layer_attrs(self, node):
        return {"blocksize": self.attribute(node, "blocksize").i}
