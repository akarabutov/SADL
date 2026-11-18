# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class Conv2DEmitter(LayerEmitter):
    """Conv2D(x NHWC, kernel) -> ONNX Conv in NCHW.

    A Const kernel is stored [kh, kw, I/groups, O] (HWIO); a computed kernel
    reaches the layer as [kh, kw, O, I] (SADL only re-orders Const kernels).
    """

    sadl_ops = (OpType.Conv2D,)

    CONST_KERNEL_TO_OIHW = [3, 2, 0, 1]
    DYNAMIC_KERNEL_TO_OIHW = [2, 3, 0, 1]

    def emit(self, ctx, layer):
        x_id, w_id = layer.inputs
        if ctx.is_const(w_id):
            w = np.ascontiguousarray(np.transpose(ctx.const(w_id, np.float32), self.CONST_KERNEL_TO_OIHW))
            weight = ctx.builder.add_initializer(w, ctx.layer(w_id).name)
        else:
            weight = ctx.builder.transpose(ctx.value(w_id), self.DYNAMIC_KERNEL_TO_OIHW, layer.name + "_kernel")

        x = self.to_nchw(ctx, ctx.value(x_id), layer)
        y = ctx.builder.add_node(
            "Conv",
            [x, weight],
            layer.name + "_conv",
            strides=self.spatial_strides(layer["strides"]),
            pads=list(layer["pads"]),
            group=layer["group"],
        )
        return self.to_nhwc(ctx, y, layer)

    @staticmethod
    def spatial_strides(strides):
        """SADL stores [sh, sw] or [1, sh, sw, 1]."""
        return list(strides[1:3]) if len(strides) == 4 else list(strides)
