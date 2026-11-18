# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.errors.conversion_error import ConversionError
from onnx2sadl.types.sadl_dtype import SadlDType

from .layer_params_reader import LayerParamsReader


class ConstParamsReader(LayerParamsReader):
    """dims, sparse flag, transposed flag, dtype, [quantizer], data.

    ``value`` is the numpy array; integer data with a non-zero quantizer q is
    fixed point (real value = int * 2^-q), kept in ``quantizer``.
    """

    NUMPY_TYPES = {SadlDType.FLOAT: np.float32, SadlDType.INT32: np.int32, SadlDType.INT16: np.int16}

    def read(self, reader):
        dims = reader.i32_list()
        if reader.u8() != 0:
            raise ConversionError("[ERROR] sparse Const tensors are not supported")
        transposed = reader.bool()
        dtype = SadlDType(reader.i32())
        if dtype not in self.NUMPY_TYPES:
            raise ConversionError("[ERROR] unsupported Const dtype {}".format(dtype))
        quantizer = 0 if dtype == SadlDType.FLOAT else reader.i32()
        count = int(np.prod(dims)) if dims else 1
        value = reader.array(self.NUMPY_TYPES[dtype], count).reshape(dims)
        return {"dims": dims, "dtype": dtype, "quantizer": quantizer, "transposed": transposed, "value": value}
