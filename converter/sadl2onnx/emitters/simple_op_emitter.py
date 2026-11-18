# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_emitter import LayerEmitter


class SimpleOpEmitter(LayerEmitter):
    """SADL layer mapped 1:1 to an ONNX op with the same (float) inputs.

    Covers element-wise ops whose broadcasting matches numpy rules
    (Add, Sub, Mul, Max, Min, BiasAdd) and unary ops (Relu, Floor, Abs...).
    """

    def __init__(self, sadl_op, onnx_op):
        self.sadl_ops = (sadl_op,)
        self.onnx_op = onnx_op

    def emit(self, ctx, layer):
        return ctx.builder.add_node(self.onnx_op, self.inputs(ctx, layer), layer.name)
