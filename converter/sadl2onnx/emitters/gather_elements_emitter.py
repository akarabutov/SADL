# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx import TensorProto

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class GatherElementsEmitter(LayerEmitter):
    """GatherElements(x, indices, axis); indices are int64 in ONNX."""

    sadl_ops = (OpType.GatherElements,)

    def emit(self, ctx, layer):
        x_id, indices_id = layer.inputs
        b = ctx.builder
        if ctx.is_const(indices_id):
            indices = ctx.int64_input(indices_id)
        else:
            indices = b.add_node("Cast", [ctx.value(indices_id)], layer.name + "_idx", to=TensorProto.INT64)
        return b.add_node("GatherElements", [ctx.value(x_id), indices], layer.name, axis=layer["axis"])
