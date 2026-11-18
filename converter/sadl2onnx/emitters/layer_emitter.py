# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class LayerEmitter:
    """Base class of the SADL layer -> ONNX nodes translators.

    SADL tensors are NHWC; the generated ONNX graph keeps that layout, so ops
    that ONNX defines on NCHW tensors are wrapped with ``to_nchw``/``to_nhwc``.
    """

    sadl_ops = ()

    NHWC_TO_NCHW = [0, 3, 1, 2]
    NCHW_TO_NHWC = [0, 2, 3, 1]

    def emit(self, ctx, layer):
        """Add the ONNX nodes of ``layer``; returns the name of its output value."""
        raise NotImplementedError

    # --- helpers ----------------------------------------------------------
    def to_nchw(self, ctx, value, layer):
        return ctx.builder.transpose(value, self.NHWC_TO_NCHW, layer.name + "_nchw")

    def to_nhwc(self, ctx, value, layer):
        return ctx.builder.transpose(value, self.NCHW_TO_NHWC, layer.name)

    def inputs(self, ctx, layer, dtype=None):
        if dtype is None:
            return [ctx.value(i) for i in layer.inputs]
        return [ctx.value(i, dtype) for i in layer.inputs]
