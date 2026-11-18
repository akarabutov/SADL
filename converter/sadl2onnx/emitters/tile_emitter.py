# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class TileEmitter(LayerEmitter):
    """Tile(x, repeats); constant repeats are already in NHWC order."""

    sadl_ops = (OpType.Tile,)

    def emit(self, ctx, layer):
        x_id, repeats_id = layer.inputs
        return ctx.builder.add_node("Tile", [ctx.value(x_id), ctx.int64_input(repeats_id)], layer.name)
