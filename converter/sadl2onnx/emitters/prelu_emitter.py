# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class PReluEmitter(LayerEmitter):
    """PReLU(x, Const slope). SADL indexes the slope per channel (last NHWC dim)
    whatever its shape ([C], [C, 1, 1]...): it is flattened to broadcast on the last axis."""

    sadl_ops = (OpType.PReLU,)

    def emit(self, ctx, layer):
        x_id, slope_id = layer.inputs
        slope = ctx.const(slope_id, np.float32).reshape(-1)
        slope_name = ctx.builder.add_initializer(slope, ctx.layer(slope_id).name)
        return ctx.builder.add_node("PRelu", [ctx.value(x_id), slope_name], layer.name)
