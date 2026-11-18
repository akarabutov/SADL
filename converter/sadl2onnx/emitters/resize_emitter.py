# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.errors.conversion_error import ConversionError
from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class ResizeEmitter(LayerEmitter):
    """Resize(x NHWC, Const scales or sizes). Scales/sizes are in NCHW order;
    scales -2 / -4 encode downsampling by 2 / 4."""

    sadl_ops = (OpType.Resize,)

    SIZES, SCALES = 1, 2
    COORDINATE_MODES = ["half_pixel", "asymmetric"]
    MODES = ["linear", "nearest", "cubic"]
    NEAREST_MODES = ["floor", "round_prefer_ceil"]

    def emit(self, ctx, layer):
        x_id, param_id = layer.inputs
        b = ctx.builder
        x = self.to_nchw(ctx, ctx.value(x_id), layer)
        values = ctx.const(param_id, np.float64).reshape(-1)
        if layer["input_label"] == self.SCALES:
            scales = np.array([self._decode_scale(v) for v in values], np.float32)
            inputs = [x, "", b.add_initializer(scales, layer.name + "_scales")]
        elif layer["input_label"] == self.SIZES:
            inputs = [x, "", "", b.add_initializer(np.round(values).astype(np.int64), layer.name + "_sizes")]
        else:
            raise ConversionError("[ERROR] Resize: unsupported input_label {}".format(layer["input_label"]))
        y = b.add_node(
            "Resize",
            inputs,
            layer.name + "_resize",
            coordinate_transformation_mode=self.COORDINATE_MODES[layer["coordinate_transformation_mode"]],
            mode=self.MODES[layer["mode"]],
            nearest_mode=self.NEAREST_MODES[layer["nearest_mode"]],
            cubic_coeff_a=-0.75,
        )
        return self.to_nhwc(ctx, y, layer)

    @staticmethod
    def _decode_scale(v):
        return 1.0 / -v if v < 0 else float(v)
