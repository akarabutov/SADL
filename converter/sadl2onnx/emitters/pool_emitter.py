# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_emitter import LayerEmitter


class PoolEmitter(LayerEmitter):
    """MaxPool / AveragePool: SADL strides and kernel are NHWC 4-vectors [1, h, w, 1]."""

    def __init__(self, sadl_op, onnx_op):
        self.sadl_ops = (sadl_op,)
        self.onnx_op = onnx_op

    def emit(self, ctx, layer):
        x = self.to_nchw(ctx, ctx.value(layer.inputs[0]), layer)
        y = ctx.builder.add_node(
            self.onnx_op,
            [x],
            layer.name + "_pool",
            kernel_shape=list(layer["kernel_shape"][1:3]),
            strides=list(layer["strides"][1:3]),
            pads=list(layer["pads"]),
        )
        return self.to_nhwc(ctx, y, layer)
