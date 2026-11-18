# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_params_reader import LayerParamsReader


class PlaceholderParamsReader(LayerParamsReader):
    """Input dims then input quantizer. A 1-D input [x] is seen as [1, x] by SADL."""

    def read(self, reader):
        dims = reader.i32_list()
        if len(dims) == 1:
            dims = [1, dims[0]]
        return {"dims": dims, "quantizer": reader.i32()}
