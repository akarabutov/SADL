# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx import TensorProto

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class WhereEmitter(LayerEmitter):
    """Where(cond, a, b): the SADL condition is a 0/1 tensor, cast to bool."""

    sadl_ops = (OpType.Where,)

    def emit(self, ctx, layer):
        cond, a, b_ = self.inputs(ctx, layer)
        b = ctx.builder
        cond = b.add_node("Cast", [cond], layer.name + "_cond", to=TensorProto.BOOL)
        return b.add_node("Where", [cond, a, b_], layer.name)
