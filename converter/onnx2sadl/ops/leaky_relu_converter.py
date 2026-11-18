# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from ..types.op_type import OpType
from ..types.sadl_dtype import SadlDType
from .op_converter import OpConverter


class LeakyReluConverter(OpConverter):
    """LeakyReLU(X, Const(alpha))."""

    onnx_ops = ("LeakyRelu",)

    def convert(self, ctx, node, inputs):
        out = node.output[0]
        coef = out + "_NOT_IN_GRAPH"
        # NOTE(refactoring): the bound helper name (_COEF_NOT_IN_GRAPH) differs from
        # the layer name (_NOT_IN_GRAPH). Harmless (the entry is never resolved),
        # kept as in the original converter.
        ctx.graph.bind(out + "_COEF_NOT_IN_GRAPH", None)
        ctx.graph.add(
            coef,
            OpType.Const,
            [],
            {
                "data": node,
                "dims": [1],
                "raw_data": np.array(float(node.attribute[0].f), dtype=np.float32).tobytes(),
                "dtype": SadlDType.FLOAT,
            },
        )
        self.emit(ctx, node, OpType.LeakyReLU, [ctx.graph.resolve(inputs[0]), coef])
