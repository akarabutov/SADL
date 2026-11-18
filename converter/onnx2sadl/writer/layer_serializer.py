# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class LayerSerializer:
    """Writes the layer-specific part of a SADL layer (after id, op, name and inputs).

    ``writes_quantizer``: the layer ends with an int32 internal quantizer (0).
    """

    writes_quantizer = False

    def write(self, stream, node):
        """Write ``node.attrs`` to ``stream``; the default layer has no parameters."""
