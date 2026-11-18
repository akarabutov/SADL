# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .op_converter import OpConverter


class CompareConverter(OpConverter):
    """Comparison mapped to the SADL Compare layer; ``mode`` selects the operator."""

    mode = None

    def convert(self, ctx, node, inputs):
        a, b = inputs[0], inputs[1]
        if ctx.is_initializer(b):
            ctx.add_const(b, ctx.extract_const(b, False, allow_force_4d=True), node)
        resolve = ctx.graph.resolve
        self.emit(ctx, node, OpType.Compare, [resolve(a), resolve(b)], mode=self.mode)
