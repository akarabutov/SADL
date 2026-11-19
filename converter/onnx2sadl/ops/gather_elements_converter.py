# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from ..types.op_type import OpType
from .op_converter import OpConverter


class GatherElementsConverter(OpConverter):
    """GatherElements(X, indices, axis); 4-D constant indices are moved to NHWC."""

    onnx_ops = ("GatherElements",)

    def convert(self, ctx, node, inputs):
        data, indices = inputs[0], inputs[1]
        if ctx.is_initializer(indices):
            tensor = ctx.extract_const(indices, False)
            if len(tensor.dims) == 4:
                tensor = tensor.transposed_4d(data_type=np.int32)
            ctx.graph.add_const(indices, tensor, node)
        ctx.graph.bind(indices, indices)

        resolve = ctx.graph.resolve
        layer_inputs = [resolve(data), resolve(indices)]
        axis = self.attribute(node, "axis")
        if len(ctx.model.shape(data) or []) == 4:  # nchw -> nhwc (unknown rank: left as is)
            axis.i = self.nchw_axis_to_nhwc(axis.i)
        self.emit(ctx, node, OpType.GatherElements, layer_inputs, axis=axis.i)
