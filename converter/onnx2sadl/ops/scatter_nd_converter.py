# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from ..errors.conversion_error import ConversionError
from ..types.op_type import OpType
from .op_converter import OpConverter


class ScatterNDConverter(OpConverter):
    """ScatterND(data, indices, updates); SADL input order is (data, updates, indices)."""

    onnx_ops = ("ScatterND",)

    def convert(self, ctx, node, inputs):
        data, indices_name = inputs[0], inputs[1]
        if not ctx.is_initializer(indices_name):
            raise ConversionError("[ERROR] The second input of the ScatterND must be indices.")

        indices = ctx.extract_const(indices_name, False, allow_force_4d=True)
        if len(indices.dims) != 5:
            raise ConversionError("[ERROR] Currently, ScatterND only supports indices of length 5.")
        # NCHW4 (or NHWC4) with N == 1 -> CHW4 (or HWC4); "4" is a position in a 4-D tensor
        indices.dims = indices.dims[1:]
        if ctx.annotation(indices_name).to_transpose:
            self._chw4_to_hwc4(indices)
        ctx.add_const(indices_name, indices, node)

        resolve = ctx.graph.resolve
        self.emit(ctx, node, OpType.ScatterND, [resolve(data), resolve(node.input[2]), resolve(indices_name)])

    @staticmethod
    def _chw4_to_hwc4(tensor):
        """Move the index grid to HWC and remap each [n, c, h, w] index to [n, h, w, c]."""
        dims = tensor.dims
        new_dims = [dims[1], dims[2], dims[0], dims[3]]
        x = np.frombuffer(tensor.raw_data, dtype=np.int32).reshape(dims).transpose(1, 2, 0, 3)
        indices = x.copy()
        for i in np.ndindex(indices.shape[:-1]):
            indices[i] = [indices[i][0], indices[i][2], indices[i][3], indices[i][1]]
        tensor.dims = new_dims
        tensor.raw_data = indices.flatten().tobytes()

    def annotate(self, annotator, node, data_layout):
        indices = annotator.model.initializer(node.input[1])
        if annotator.global_layout == "nchw":
            annotator.annotations[indices.name].to_transpose = True
