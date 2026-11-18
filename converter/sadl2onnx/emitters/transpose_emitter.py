# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class TransposeEmitter(LayerEmitter):
    """Transpose(x, Const perm); -1 in the permutation keeps the dimension in place."""

    sadl_ops = (OpType.Transpose,)

    def emit(self, ctx, layer):
        x_id, perm_id = layer.inputs
        perm = [int(p) for p in ctx.const(perm_id).reshape(-1)]
        perm = [k if p == -1 else p for k, p in enumerate(perm)]
        return ctx.builder.transpose(ctx.value(x_id), perm, layer.name)
