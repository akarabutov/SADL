# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_emitter import LayerEmitter


class NchwOpEmitter(LayerEmitter):
    """Single-input ONNX op defined on NCHW tensors (GlobalAveragePool, DepthToSpace...)."""

    def __init__(self, sadl_op, onnx_op, **fixed_attrs):
        self.sadl_ops = (sadl_op,)
        self.onnx_op = onnx_op
        self.fixed_attrs = fixed_attrs

    def attributes(self, layer):
        return dict(self.fixed_attrs)

    def emit(self, ctx, layer):
        x = self.to_nchw(ctx, ctx.value(layer.inputs[0]), layer)
        y = ctx.builder.add_node(self.onnx_op, [x], layer.name + "_op", **self.attributes(layer))
        return self.to_nhwc(ctx, y, layer)
