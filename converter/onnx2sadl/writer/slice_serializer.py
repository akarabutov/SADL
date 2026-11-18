# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_serializer import LayerSerializer


class SliceSerializer(LayerSerializer):
    """(start, end) for the h, w and c dimensions."""

    DIMS = ("h", "w", "c")

    def write(self, stream, node):
        for dim in self.DIMS:
            stream.i32(node[f"start_{dim}"], f"start index for {dim} slicing")
            stream.i32(node[f"end_{dim}"], f"end index for {dim} slicing")
