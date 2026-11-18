# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class SliceEmitter(LayerEmitter):
    """Slice on H, W, C of an NHWC tensor, or on the last dim of a 2-D tensor."""

    sadl_ops = (OpType.Slice,)

    def emit(self, ctx, layer):
        x_id = layer.inputs[0]
        if self._rank(ctx, x_id) == 2:
            axes, keys = [1], ["c"]
        else:
            axes, keys = [1, 2, 3], ["h", "w", "c"]
        b = ctx.builder
        starts = b.add_initializer(np.array([layer["start_" + k] for k in keys], np.int64), layer.name + "_starts")
        ends = b.add_initializer(np.array([layer["end_" + k] for k in keys], np.int64), layer.name + "_ends")
        axes = b.add_initializer(np.array(axes, np.int64), layer.name + "_axes")
        return b.add_node("Slice", [ctx.value(x_id), starts, ends, axes], layer.name)

    @staticmethod
    def _rank(ctx, layer_id):
        return ctx.shapes.rank(layer_id)
