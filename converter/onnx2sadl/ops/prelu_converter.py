# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .op_converter import OpConverter


class PReluConverter(OpConverter):
    """PReLU(X, Const(slope))."""

    onnx_ops = ("PRelu",)

    def convert(self, ctx, node, inputs):
        slope = inputs[1]
        ctx.add_const(slope, ctx.extract_const(slope, False), node)
        resolve = ctx.graph.resolve
        self.emit(ctx, node, OpType.PReLU, [resolve(inputs[0]), resolve(slope)])
