# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class PlaceholderEmitter(LayerEmitter):
    """Graph input (NHWC for 4-D tensors)."""

    sadl_ops = (OpType.Placeholder,)

    def emit(self, ctx, layer):
        return ctx.builder.add_input(layer.name, layer["dims"])
