# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np
from onnx import TensorProto

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class GatherEmitter(LayerEmitter):
    """Gather(x, indices, axis). Scalar indices are stored as [1] in SADL."""

    sadl_ops = (OpType.Gather,)

    def emit(self, ctx, layer):
        x_id, indices_id = layer.inputs
        b = ctx.builder
        if ctx.is_const(indices_id):
            indices = ctx.const(indices_id, np.int64)
            if layer["indices_is_scalar"]:
                indices = indices.reshape(())
            indices = b.add_initializer(indices, ctx.layer(indices_id).name)
        else:
            indices = b.add_node("Cast", [ctx.value(indices_id)], layer.name + "_idx", to=TensorProto.INT64)
        return b.add_node("Gather", [ctx.value(x_id), indices], layer.name, axis=layer["axis"])
