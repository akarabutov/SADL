# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_serializer import LayerSerializer


class Conv2DTransposeSerializer(LayerSerializer):
    """strides, pads, output_padding, quantizer."""

    writes_quantizer = True

    def write(self, stream, node):
        stream.i32_list(node["strides"], "nb_dim_strides")
        stream.i32_list(node["pads"], "nb_dim_pads")
        stream.i32_list(node["output_padding"], "nb_dim_output_padding")
