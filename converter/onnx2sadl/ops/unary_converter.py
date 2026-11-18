# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .op_converter import OpConverter


class UnaryConverter(OpConverter):
    """One input, one SADL layer, no parameters."""

    sadl_op = None

    def convert(self, ctx, node, inputs):
        self.emit(ctx, node, self.sadl_op, [ctx.graph.resolve(inputs[0])], **self.layer_attrs(node))

    def layer_attrs(self, node):
        return {}
