# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_serializer import LayerSerializer


class PlaceholderSerializer(LayerSerializer):
    """Input dims then the input quantizer (0)."""

    def write(self, stream, node):
        stream.i32_list(node["dims"], "nb input dimension")
        stream.i32(0, "quantizer_of_input")
