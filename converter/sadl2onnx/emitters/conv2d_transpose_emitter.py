# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.types.op_type import OpType

from .conv2d_emitter import Conv2DEmitter
from .layer_emitter import LayerEmitter


class Conv2DTransposeEmitter(LayerEmitter):
    """Conv2DTranspose(x NHWC, Const kernel [kh, kw, O, I]) -> ONNX ConvTranspose (W [I, O, kh, kw])."""

    sadl_ops = (OpType.Conv2DTranspose,)

    KERNEL_TO_IOHW = [3, 2, 0, 1]

    def emit(self, ctx, layer):
        x_id, w_id = layer.inputs
        w = np.ascontiguousarray(np.transpose(ctx.const(w_id, np.float32), self.KERNEL_TO_IOHW))
        weight = ctx.builder.add_initializer(w, ctx.layer(w_id).name)
        x = self.to_nchw(ctx, ctx.value(x_id), layer)
        y = ctx.builder.add_node(
            "ConvTranspose",
            [x, weight],
            layer.name + "_convt",
            strides=Conv2DEmitter.spatial_strides(layer["strides"]),
            pads=list(layer["pads"]),
            output_padding=list(layer["output_padding"]),
        )
        return self.to_nhwc(ctx, y, layer)
