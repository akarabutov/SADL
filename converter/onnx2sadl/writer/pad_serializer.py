# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_serializer import LayerSerializer


class PadSerializer(LayerSerializer):
    """mode, then the 4 H/W pads (no count)."""

    def write(self, stream, node):
        stream.i32(node["mode"], "mode")
        stream.log(f"#\t nb_pads {len(node['pads'])}")
        stream.i32_values(node["pads"])
