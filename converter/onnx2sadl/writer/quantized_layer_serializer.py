# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_serializer import LayerSerializer


class QuantizedLayerSerializer(LayerSerializer):
    """Layer without parameters but with an internal quantizer (MatMul, Mul)."""

    writes_quantizer = True
