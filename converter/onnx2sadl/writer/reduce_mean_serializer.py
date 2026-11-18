# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_serializer import LayerSerializer


class ReduceMeanSerializer(LayerSerializer):
    """axes, keepdims (bool)."""

    def write(self, stream, node):
        stream.i32_list(node["axes"], "nb_dim_axes")
        stream.bool(node["keepdims"], "keepdims")
