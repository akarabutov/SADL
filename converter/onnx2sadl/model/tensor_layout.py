# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np


class TensorLayout:
    """Pure functions that re-order raw tensor data between ONNX and SADL layouts.

    All functions take/return ``(raw_bytes, dims)``.
    """

    NCHW_TO_NHWC = [0, 2, 3, 1]
    NHWC_TO_NCHW = [0, 3, 1, 2]

    @staticmethod
    def transpose_4d(raw_data, dims, perm, data_type=np.float32):
        """Generic transpose of a constant tensor (used for 4-D tensors)."""
        assert len(dims) == len(perm), (
            f"[ERROR] Mismatched lengths: dims({len(dims)}) vs perm({len(perm)})"
        )
        x = np.frombuffer(raw_data, dtype=data_type)
        x = np.reshape(x, dims)
        x = np.transpose(x, perm)
        new_shape = [dims[i] for i in perm]
        return x.tobytes(), new_shape

    @staticmethod
    def conv_weight_to_sadl(raw_data, dims):
        """ONNX conv weights [O, I, H, W] -> SADL layout [H, W, I, O] (float32)."""
        new_dims = [dims[2], dims[3], dims[1], dims[0]]
        x = np.frombuffer(raw_data, dtype=np.float32)
        x = x.reshape(new_dims[3], new_dims[2], new_dims[0] * new_dims[1]).transpose().flatten()
        return x.tobytes(), new_dims

    @staticmethod
    def shape_vector_nchw_to_nhwc(raw_data, dims):
        """A 4-element int32 shape vector [N, C, H, W] -> [N, H, W, C]."""
        assert dims[0] == 4
        x = np.frombuffer(raw_data, dtype=np.int32)
        x = np.asarray([x[0], x[2], x[3], x[1]])
        return x.tobytes(), [dims[0]]

    @staticmethod
    def transpose_matrix(raw_data, dims):
        """2-D float32 matrix transpose."""
        x = np.frombuffer(raw_data, dtype=np.float32)
        x = x.reshape(dims[0], dims[1])
        x = np.transpose(x)
        return x.flatten().tobytes(), [dims[1], dims[0]]
