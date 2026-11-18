# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from ..types.op_type import OpType
from .op_converter import OpConverter


class SliceConverter(OpConverter):
    """Slice(X, starts, ends[, axes, steps]) on one of the H, W, C dimensions.

    TF exports give starts/ends for all 4 dims; PyTorch exports give one value
    and the sliced dim in ``axes``.
    """

    onnx_ops = ("Slice",)

    INT32_MAX = 2147483647
    NHWC_KEYS = ["b", "h", "w", "c"]
    NCHW_KEYS = ["b", "c", "h", "w"]

    def convert(self, ctx, node, inputs):
        model = ctx.model
        axes = self._axes(model, node)
        layer_inputs = [ctx.graph.resolve(inputs[0])]
        start = model.initializer_or_constant_int64(inputs[1])
        end = model.initializer_or_constant_int64(node.input[2])

        attrs = {}
        for key in self.NHWC_KEYS[1:]:
            attrs[f"start_{key}"] = 0
            attrs[f"end_{key}"] = self.INT32_MAX

        if len(start) > 1:  # TensorFlow
            if start[0] != 0 or end[0] != self.INT32_MAX:
                raise ConversionError("[ERROR] currently slicing not supported for first dimension")
            for i in range(1, len(start)):
                attrs[f"start_{self.NHWC_KEYS[i]}"] = start[i]
                attrs[f"end_{self.NHWC_KEYS[i]}"] = min(end[i], self.INT32_MAX)
        else:  # PyTorch
            for i in range(len(end) - 1):
                if start[i] != 0:
                    raise ConversionError("[ERROR] currently slicing not supported for first dimension")
                if end[i] < self.INT32_MAX:
                    raise ConversionError("[ERROR] currently slicing only supported for last channel")
            if axes is None:
                raise ConversionError("[ERROR] Slice without axes input is not supported")
            # INT64 "infinity" of PyTorch exports is clamped to INT32_MAX
            key = self.NCHW_KEYS[axes[0]]
            attrs[f"start_{key}"] = start[-1]
            attrs[f"end_{key}"] = min(end[-1], self.INT32_MAX)

        ctx.graph.add(node.output[0], OpType.Slice, layer_inputs, attrs)
        ctx.graph.bind(node.output[0], node.output[0])

    @staticmethod
    def _axes(model, node):
        """Sliced axes of a PyTorch export (5 inputs), None otherwise."""
        if len(node.input) != 5:
            return None
        axes = model.initializer_or_constant_int64(node.input[3])
        steps = model.initializer_or_constant_int64(node.input[4])
        if len(axes) != 1:
            raise ConversionError("[ERROR] currently sadl slicing support lenght of axes equal to one")
        if axes[0] == 0:
            raise ConversionError("[ERROR] currently slicing not supported for first dimension")
        if not (len(steps) == 1 and steps[0] == 1):
            raise ConversionError("[ERROR] currently step has to be default one")
        return axes
