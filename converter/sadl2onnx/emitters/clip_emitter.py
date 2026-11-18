# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class ClipEmitter(LayerEmitter):
    """Clip(x, Const min, Const max); ONNX wants scalar bounds."""

    sadl_ops = (OpType.Clip,)

    def emit(self, ctx, layer):
        x_id, min_id, max_id = layer.inputs
        b = ctx.builder
        bounds = []
        for bound_id in (min_id, max_id):
            value = ctx.const(bound_id, np.float32).reshape(-1)[0]
            bounds.append(b.add_initializer(np.array(value, np.float32), ctx.layer(bound_id).name))
        return b.add_node("Clip", [ctx.value(x_id)] + bounds, layer.name)
