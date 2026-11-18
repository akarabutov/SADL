# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .op_converter import OpConverter


class BitShiftConverter(OpConverter):
    """BitShift(X, Y); direction: 0 = LEFT, 1 = RIGHT."""

    onnx_ops = ("BitShift",)

    def convert(self, ctx, node, inputs):
        x, y = inputs[0], node.input[1]
        for name in (x, y):
            if ctx.is_initializer(name):
                ctx.add_const(name, ctx.extract_const(name, ctx.annotation(name).to_transpose), node)
        direction = self.attribute(node, "direction").s.decode("utf-8")
        resolve = ctx.graph.resolve
        self.emit(
            ctx, node, OpType.BitShift, [resolve(x), resolve(y)], direction=1 if direction == "RIGHT" else 0
        )
