# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class PadEmitter(LayerEmitter):
    """Pad on H and W of an NHWC tensor; pads = [H_b, W_b, H_a, W_a]."""

    sadl_ops = (OpType.Pad,)

    MODES = ["constant", "reflect", "edge"]

    def emit(self, ctx, layer):
        b = ctx.builder
        hb, wb, ha, wa = layer["pads"]
        pads = b.add_initializer(np.array([0, hb, wb, 0, 0, ha, wa, 0], np.int64), layer.name + "_pads")
        inputs = [ctx.value(layer.inputs[0]), pads]
        if len(layer.inputs) > 1:  # constant value
            value = ctx.const(layer.inputs[1], np.float32).reshape(())
            inputs.append(b.add_initializer(value, layer.name + "_value"))
        return b.add_node("Pad", inputs, layer.name, mode=self.MODES[layer["mode"]])
