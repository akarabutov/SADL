# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class MatMulEmitter(LayerEmitter):
    """SADL stores B as [..., k, n] like ONNX (the C++ side transposes it at load time)."""

    sadl_ops = (OpType.MatMul,)

    def emit(self, ctx, layer):
        return ctx.builder.add_node("MatMul", self.inputs(ctx, layer), layer.name)
