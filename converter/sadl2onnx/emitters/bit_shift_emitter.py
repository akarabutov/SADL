# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class BitShiftEmitter(LayerEmitter):
    """BitShift on a float model, as computed by SADL: floor(x * 2^y) (LEFT)
    or floor(x / 2^y) (RIGHT). ONNX BitShift only exists for unsigned ints."""

    sadl_ops = (OpType.BitShift,)

    def emit(self, ctx, layer):
        x_id, y_id = layer.inputs
        b = ctx.builder
        if ctx.is_const(y_id):
            factor = np.power(2.0, ctx.const(y_id, np.float64)).astype(np.float32)
            factor = b.add_initializer(factor, layer.name + "_factor")
        else:
            two = b.add_initializer(np.array(2.0, np.float32), layer.name + "_two")
            factor = b.add_node("Pow", [two, ctx.value(y_id)], layer.name + "_factor")
        op = "Div" if layer["direction"] == 1 else "Mul"
        scaled = b.add_node(op, [ctx.value(x_id), factor], layer.name + "_scaled")
        return b.add_node("Floor", [scaled], layer.name)
