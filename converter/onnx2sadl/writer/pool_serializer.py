# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_serializer import LayerSerializer


class PoolSerializer(LayerSerializer):
    """strides, kernel, pads (MaxPool, AveragePool)."""

    def write(self, stream, node):
        stream.i32_list(node["strides"], "nb_dim_strides")
        stream.i32_list(node["kernel_shape"], "nb_dim_kernel")
        stream.i32_list(node["pads"], "nb_dim_pads")
