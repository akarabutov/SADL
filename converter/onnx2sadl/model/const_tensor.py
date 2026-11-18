# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from .tensor_layout import TensorLayout


class ConstTensor:
    """Constant data ready to be stored in a SADL Const layer."""

    def __init__(self, dims, raw_data, dtype):
        self.dims = dims
        self.raw_data = raw_data
        self.dtype = dtype

    def is_empty(self):
        return self.raw_data == b""

    def transposed_4d(self, perm=None, data_type=np.float32):
        """Return a copy with 4-D data permuted (NCHW -> NHWC by default)."""
        perm = TensorLayout.NCHW_TO_NHWC if perm is None else perm
        raw_data, dims = TensorLayout.transpose_4d(self.raw_data, self.dims, perm, data_type)
        return ConstTensor(dims, raw_data, self.dtype)

    def to_attrs(self, owner):
        return {
            "data": owner,
            "dims": self.dims,
            "raw_data": self.raw_data,
            "dtype": self.dtype,
        }
