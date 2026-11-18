# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class FlattenEmitter(LayerEmitter):
    sadl_ops = (OpType.Flatten,)

    def emit(self, ctx, layer):
        return ctx.builder.add_node("Flatten", [ctx.value(layer.inputs[0])], layer.name, axis=layer["axis"])
