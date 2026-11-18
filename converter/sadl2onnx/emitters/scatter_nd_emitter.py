# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class ScatterNDEmitter(LayerEmitter):
    """ScatterND(data [1,H,W,C], updates [1,h,w,c], Const indices [h,w,c,4] of NHWC positions).

    ONNX expects (data, indices, updates) with updates shaped indices.shape[:-1].
    """

    sadl_ops = (OpType.ScatterND,)

    def emit(self, ctx, layer):
        data_id, updates_id, indices_id = layer.inputs
        b = ctx.builder
        indices = b.add_initializer(ctx.const(indices_id, np.int64), ctx.layer(indices_id).name)
        axes = b.add_initializer(np.array([0], np.int64), layer.name + "_axis0")
        updates = b.add_node("Squeeze", [ctx.value(updates_id), axes], layer.name + "_updates")
        return b.add_node("ScatterND", [ctx.value(data_id), indices, updates], layer.name)
