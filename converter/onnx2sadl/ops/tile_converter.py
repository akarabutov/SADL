# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .op_converter import OpConverter


class TileConverter(OpConverter):
    """Tile(X, repeats); constant repeats are re-ordered to NHWC."""

    onnx_ops = ("Tile",)

    def convert(self, ctx, node, inputs):
        repeats = inputs[1]
        producer = ctx.model.producer(repeats)
        if producer is not None and (not hasattr(producer, "op_type") or producer.op_type == "Constant"):
            ctx.graph.add_const(repeats, ctx.extract_const(repeats, True), node)
        ctx.graph.bind(repeats, repeats)
        resolve = ctx.graph.resolve
        self.emit(ctx, node, OpType.Tile, [resolve(inputs[0]), resolve(repeats)])
