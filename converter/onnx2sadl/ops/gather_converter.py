# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .op_converter import OpConverter


class GatherConverter(OpConverter):
    """Gather(X, indices, axis); the axis of a 4-D input is remapped to NHWC."""

    onnx_ops = ("Gather",)

    def convert(self, ctx, node, inputs):
        data, indices = inputs[0], inputs[1]
        indices_is_scalar = False
        if ctx.is_initializer(indices):
            ctx.graph.add_const(indices, ctx.extract_const(indices, False), node)
            indices_is_scalar = len(ctx.model.shape(indices)) == 0
        ctx.graph.bind(indices, indices)

        resolve = ctx.graph.resolve
        layer_inputs = [resolve(data), resolve(indices)]
        axis = self.attribute(node, "axis")
        if len(ctx.model.shape(data)) == 4:  # nchw -> nhwc
            axis.i = self.nchw_axis_to_nhwc(axis.i)
        self.emit(ctx, node, OpType.Gather, layer_inputs, axis=axis.i, indices_is_scalar=indices_is_scalar)
