# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from ..errors.conversion_error import ConversionError
from ..types.op_type import OpType
from .op_converter import OpConverter


class PadConverter(OpConverter):
    """Pad(X, pads[, constant_value]) on H and W only."""

    onnx_ops = ("Pad",)

    MODES = ["constant", "reflect", "edge"]

    def convert(self, ctx, node, inputs):
        mode = self.attribute(node, "mode").s.decode("utf-8")
        if mode not in self.MODES:
            raise ConversionError(f"[ERROR] Currently, the mode of Pad must in {self.MODES} {node}")

        pads_tensor = ctx.model.producer(inputs[1])
        pads = self.nchw_to_hw_pads(np.frombuffer(pads_tensor.raw_data, dtype=np.int64).tolist())

        has_value = len(node.input) > 2 and ctx.is_initializer(node.input[2])
        if has_value:
            value = node.input[2]
            ctx.add_const(value, ctx.extract_const(value, False, allow_force_4d=True), node)

        resolve = ctx.graph.resolve
        layer_inputs = [resolve(inputs[0])]
        if has_value:
            layer_inputs.append(resolve(node.input[2]))
        self.emit(ctx, node, OpType.Pad, layer_inputs, mode=self.MODES.index(mode), pads=pads)

    @staticmethod
    def nchw_to_hw_pads(pads_nchw):
        """[N_b, C_b, H_b, W_b, N_a, C_a, H_a, W_a] -> [H_b, W_b, H_a, W_a]."""
        if pads_nchw[0] != 0 or pads_nchw[4] != 0:
            raise ValueError("Padding on the batch dimension (N) is not allowed.")
        if pads_nchw[1] != 0 or pads_nchw[5] != 0:
            raise ValueError("Padding on the channel dimension (C) is not allowed.")
        return [pads_nchw[2], pads_nchw[3], pads_nchw[6], pads_nchw[7]]
