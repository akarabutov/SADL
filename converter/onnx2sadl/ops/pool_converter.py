# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .unary_converter import UnaryConverter


class PoolConverter(UnaryConverter):
    """2-D pooling: strides and kernel are stored as NHWC 4-vectors [1, h, w, 1]."""

    def layer_attrs(self, node):
        strides = self.attribute(node, "strides")
        pads = self.attribute(node, "pads")
        kernel = self.attribute(node, "kernel_shape")
        return {
            "strides": [1, strides.ints[0], strides.ints[1], 1],
            "pads": [0, 0, 0, 0] if pads is None else pads.ints,
            "kernel_shape": [1, kernel.ints[0], kernel.ints[1], 1],
        }
