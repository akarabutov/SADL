# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from ..errors.conversion_error import ConversionError
from ..types.op_type import OpType
from ..types.sadl_dtype import SadlDType
from .op_converter import OpConverter


class ResizeConverter(OpConverter):
    """Resize(X, roi, scales, sizes): SADL supports (X, scales) or (X, sizes).

    ``input_label`` is a bit mask of the constant inputs present:
    bit 3 = X, 2 = roi, 1 = scales, 0 = sizes.
    """

    onnx_ops = ("Resize",)

    SCALES_INDEX = 2
    MODES = ["linear", "nearest", "cubic"]
    NEAREST_MODES = ["floor", "round_prefer_ceil"]
    COORDINATE_MODES = {"half_pixel": 0, "pytorch_half_pixel": 0, "asymmetric": 1}

    def convert(self, ctx, node, inputs):
        input_label = 0
        layer_inputs = [ctx.graph.resolve(inputs[0])]
        for index, name in enumerate(node.input):
            if not ctx.is_initializer(name):
                continue
            tensor = ctx.extract_const(name, False, allow_force_4d=True)
            if index == self.SCALES_INDEX:
                tensor.raw_data = self._encode_scales(tensor.raw_data)
                tensor.dtype = SadlDType.FLOAT
            if tensor.is_empty():  # empty tensor: input not used
                continue
            ctx.add_const(name, tensor, node)
            layer_inputs.append(ctx.graph.resolve(name))
            input_label += 1 << (3 - index)
        if input_label not in (1, 2):
            raise ConversionError(
                "[ERROR] Currently, the inputs of Resize have to be X and sizes, or X and scales."
            )
        self.emit(ctx, node, OpType.Resize, layer_inputs, input_label=input_label, **self._modes(node))

    @staticmethod
    def _encode_scales(raw_data):
        """Integer scales; downscaling 0.5 / 0.25 is encoded as -2 / -4."""
        mapped = []
        for v in np.frombuffer(raw_data, dtype=np.float32).tolist():
            if abs(v - 0.5) < 1e-6:
                mapped.append(-2)
            elif abs(v - 0.25) < 1e-6:
                mapped.append(-4)
            else:
                mapped.append(int(round(float(v))))
        return np.asarray(mapped, dtype=np.float32).tobytes()

    def _modes(self, node):
        coordinate_mode = self.attribute(node, "coordinate_transformation_mode").s.decode("utf-8")
        cubic_coeff_a = self.attribute(node, "cubic_coeff_a")
        exclude_outside = self.attribute(node, "exclude_outside")
        mode = self.attribute(node, "mode").s.decode("utf-8")
        nearest_mode = self.attribute(node, "nearest_mode").s.decode("utf-8")

        if coordinate_mode not in self.COORDINATE_MODES:
            raise ConversionError(
                f"[ERROR] Currently, the coordinate_transformation_mode (={coordinate_mode}) of Resize "
                f"must in {list(self.COORDINATE_MODES)} {node}"
            )
        cubic_coeff_a = -0.75 if cubic_coeff_a is None else cubic_coeff_a.f
        if cubic_coeff_a != -0.75:
            raise ConversionError(f"[ERROR] Currently, the cubic_coeff_a of Resize must be default -0.75. {node}")
        exclude_outside = 0 if exclude_outside is None else exclude_outside.i
        if exclude_outside != 0:
            raise ConversionError(f"[ERROR] Currently, the exclude_outside of Resize must be default 0. {node}")
        if mode not in self.MODES:
            raise ConversionError(f"[ERROR] Currently, the mode of Resize must in {self.MODES} {node}")
        if nearest_mode not in self.NEAREST_MODES:
            raise ConversionError(
                f"[ERROR] Currently, the nearest_mode of Resize must in {self.NEAREST_MODES} {node}"
            )
        return {
            "coordinate_transformation_mode": self.COORDINATE_MODES[coordinate_mode],
            "mode": self.MODES.index(mode),
            "nearest_mode": self.NEAREST_MODES.index(nearest_mode),
        }
