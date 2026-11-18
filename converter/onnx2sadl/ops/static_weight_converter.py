# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from .op_converter import OpConverter


class StaticWeightConverter(OpConverter):
    """Layer(X, W[, B]) with a constant weight W and an optional bias B (Gemm, ConvTranspose)."""

    sadl_op = None

    def convert(self, ctx, node, inputs):
        if len(node.input) not in (2, 3):
            raise ConversionError("parse_graph_node: Error on node type")

        weight = inputs[1]
        tensor = ctx.extract_const(weight, ctx.producer_to_transpose(weight))
        ctx.add_const(weight, tensor, node)

        attrs = {"data": node}
        attrs.update(self.layer_attrs(node))
        resolve = ctx.graph.resolve
        layer_inputs = [resolve(inputs[0]), resolve(weight)]
        self.bind_output_before_bias(ctx, node)
        ctx.graph.add(node.output[0], self.sadl_op, layer_inputs, attrs)
        if len(node.input) == 3:
            self.add_bias(ctx, node)

    def layer_attrs(self, node):
        return {}
