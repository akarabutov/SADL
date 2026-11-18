# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np
from onnx import numpy_helper

from ..errors.conversion_error import ConversionError
from ..model.constant_extractor import ConstantExtractor
from ..types.op_type import OpType
from .op_converter import OpConverter


class ClipConverter(OpConverter):
    """Clip(X, min, max). Bounds come from inputs (opset >= 11) or attributes (older opsets)."""

    onnx_ops = ("Clip",)

    def convert(self, ctx, node, inputs):
        if len(node.input) not in (1, 3):
            raise ConversionError("[ERROR] Clip expects 1 or 3 inputs")
        from_attributes = len(node.input) == 1
        if from_attributes:
            bounds = [(f"{node.name}_min", "min"), (f"{node.name}_max", "max")]
        else:
            bounds = [(inputs[1], "min"), (inputs[2], "max")]

        for name, attr_name in bounds:
            if from_attributes:
                tensor = self._attribute_bound(node, attr_name)
            else:
                tensor = ctx.extract_const(name, False)
            ctx.graph.add_const(name, tensor, node)
        for name, _ in bounds:
            ctx.graph.bind(name, name)

        resolve = ctx.graph.resolve
        self.emit(ctx, node, OpType.Clip, [resolve(inputs[0])] + [resolve(name) for name, _ in bounds])

    def _attribute_bound(self, node, attr_name):
        value = self.attribute(node, attr_name).f
        value = 0 if value is None else value
        tensor = numpy_helper.from_array(np.asarray([value], dtype=np.float32), attr_name)
        return ConstantExtractor.from_tensor(tensor, False)
