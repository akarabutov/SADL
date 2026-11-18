# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_serializer import LayerSerializer


class GatherSerializer(LayerSerializer):
    """axis, indices_is_scalar (bool)."""

    def write(self, stream, node):
        stream.i32(node["axis"], "axis")
        stream.bool(node["indices_is_scalar"], "indices_is_scalar")
