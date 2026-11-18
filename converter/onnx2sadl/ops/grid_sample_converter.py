# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from ..types.op_type import OpType
from .binary_converter import BinaryConverter


class GridSampleConverter(BinaryConverter):
    """GridSample(X, grid). Modes are stored as their index in the lists below."""

    onnx_ops = ("GridSample",)
    sadl_op = OpType.GridSample
    const_to_transpose = "producer"

    MODES = ["nearest", "bilinear", "bicubic"]
    PADDING_MODES = ["border", "zeros"]

    def layer_attrs(self, node):
        align_corners = self.attribute(node, "align_corners").i
        mode = self.attribute(node, "mode").s.decode("utf-8")
        padding_mode = self.attribute(node, "padding_mode").s.decode("utf-8")
        if mode not in self.MODES:
            raise ConversionError(
                f"[ERROR] Currently, the mode of GridSample must in {self.MODES} {node}"
            )
        if padding_mode not in self.PADDING_MODES:
            raise ConversionError(
                f"[ERROR] Currently, the padding_mode of GridSample must in {self.PADDING_MODES} {node}"
            )
        return {
            "align_corners": align_corners,
            "mode": self.MODES.index(mode),
            "padding_mode": self.PADDING_MODES.index(padding_mode),
        }
