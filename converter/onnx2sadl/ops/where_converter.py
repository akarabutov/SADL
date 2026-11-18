# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .op_converter import OpConverter


class WhereConverter(OpConverter):
    """Where(condition, A, B); A and B may be constants."""

    onnx_ops = ("Where",)

    def convert(self, ctx, node, inputs):
        cond, a, b = inputs[0], inputs[1], node.input[2]
        for name in (a, b):
            if ctx.is_initializer(name):
                ctx.add_const(name, ctx.extract_const(name, False, allow_force_4d=True), node)
        resolve = ctx.graph.resolve
        self.emit(ctx, node, OpType.Where, [resolve(cond), resolve(a), resolve(b)])
