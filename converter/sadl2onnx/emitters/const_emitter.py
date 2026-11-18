# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class ConstEmitter(LayerEmitter):
    """Constants are materialized lazily by EmitContext.value (with the dtype the consumer needs)."""

    sadl_ops = (OpType.Const,)

    def emit(self, ctx, layer):
        return None
