# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx import TensorProto

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class CompareEmitter(LayerEmitter):
    """Compare(a, b) -> Less/Greater/Equal then Cast: SADL outputs 0/1 in the model type."""

    sadl_ops = (OpType.Compare,)

    ONNX_OPS = ["Less", "Greater", "Equal"]

    def emit(self, ctx, layer):
        b = ctx.builder
        flag = b.add_node(self.ONNX_OPS[layer["mode"]], self.inputs(ctx, layer), layer.name + "_bool")
        return b.add_node("Cast", [flag], layer.name, to=TensorProto.FLOAT)
