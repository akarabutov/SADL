# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class ReshapeEmitter(LayerEmitter):
    """Reshape(x, shape) and Expand(x, shape): the shape input becomes int64."""

    def __init__(self, sadl_op, onnx_op):
        self.sadl_ops = (sadl_op,)
        self.onnx_op = onnx_op

    def emit(self, ctx, layer):
        x_id, shape_id = layer.inputs
        return ctx.builder.add_node(self.onnx_op, [ctx.value(x_id), ctx.int64_input(shape_id)], layer.name)

    @classmethod
    def reshape(cls):
        return cls(OpType.Reshape, "Reshape")

    @classmethod
    def expand(cls):
        return cls(OpType.Expand, "Expand")
