# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class ConcatEmitter(LayerEmitter):
    """ConcatV2(x1, ..., xn, Const axis)."""

    sadl_ops = (OpType.ConcatV2,)

    def emit(self, ctx, layer):
        *data_ids, axis_id = layer.inputs
        axis = int(ctx.const(axis_id).reshape(-1)[0])
        return ctx.builder.add_node("Concat", [ctx.value(i) for i in data_ids], layer.name, axis=axis)
