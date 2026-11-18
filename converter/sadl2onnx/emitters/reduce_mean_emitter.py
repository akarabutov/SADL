# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class ReduceMeanEmitter(LayerEmitter):
    """ReduceMean with attribute axes (opset 17)."""

    sadl_ops = (OpType.ReduceMean,)

    def emit(self, ctx, layer):
        return ctx.builder.add_node(
            "ReduceMean",
            [ctx.value(layer.inputs[0])],
            layer.name,
            axes=list(layer["axes"]),
            keepdims=int(layer["keepdims"]),
        )
