# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .op_converter import OpConverter


class ExpandConverter(OpConverter):
    onnx_ops = ("Expand",)

    def convert(self, ctx, node, inputs):
        resolve = ctx.graph.resolve
        self.emit(ctx, node, OpType.Expand, [resolve(inputs[0]), resolve(inputs[1])])
