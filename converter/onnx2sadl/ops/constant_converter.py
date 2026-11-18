# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .op_converter import OpConverter


class ConstantConverter(OpConverter):
    """Constant nodes are not layers: their value is read by the consuming converter."""

    onnx_ops = ("Constant",)

    def convert(self, ctx, node, inputs):
        pass
