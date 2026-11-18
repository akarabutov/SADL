# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class LeakyReluEmitter(LayerEmitter):
    """LeakyReLU(x, Const alpha) -> LeakyRelu with an alpha attribute."""

    sadl_ops = (OpType.LeakyReLU,)

    def emit(self, ctx, layer):
        x_id, alpha_id = layer.inputs
        alpha = float(ctx.const(alpha_id, "float32").reshape(-1)[0])
        return ctx.builder.add_node("LeakyRelu", [ctx.value(x_id)], layer.name, alpha=alpha)
