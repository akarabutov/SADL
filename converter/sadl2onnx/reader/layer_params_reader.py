# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class LayerParamsReader:
    """Reads the layer-specific part of a SADL layer (inverse of a LayerSerializer).

    ``reads_quantizer``: the layer ends with an int32 internal quantizer.
    """

    reads_quantizer = False

    def read(self, reader):
        """Returns the params dict; the default layer has no parameters."""
        return {}
