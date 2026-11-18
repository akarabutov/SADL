# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.sadl_dtype import SadlDType
from .layer_serializer import LayerSerializer


class ConstSerializer(LayerSerializer):
    """dims, sparse flag, transposed flag, dtype, [quantizer if integer], data."""

    def write(self, stream, node):
        stream.i32_list(node["dims"], "nb_dim")
        stream.raw(b"\x00")  # not sparse
        stream.raw(b"\x00")  # not transposed
        stream.i32(node["dtype"], "dtype")
        if node["dtype"] != SadlDType.FLOAT:
            stream.i32(0, "quantizer")
        stream.raw(node["raw_data"])
