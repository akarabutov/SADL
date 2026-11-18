# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from enum import IntEnum

import numpy as np

from .sadl_dtype import SadlDType


class OnnxDType(IntEnum):
    """Subset of ONNX TensorProto.DataType handled by the converter."""

    # https://github.com/onnx/onnx/blob/main/onnx/onnx.in.proto
    FLOAT = 1
    INT8 = 3
    INT16 = 5
    INT32 = 6
    INT64 = 7

    @property
    def numpy_type(self):
        """numpy type used to decode raw_data of this ONNX type."""
        return {
            # NOTE(refactoring): the original used np.float (= float64, removed in
            # numpy 1.24), so FLOAT tensors (float32 on disk) are decoded as float64
            # here. Kept for identical behaviour; only used for Gather indices in
            # SillyReshapeDetector, which are integers in practice.
            OnnxDType.FLOAT: np.float64,
            OnnxDType.INT8: np.int8,
            OnnxDType.INT16: np.int16,
            OnnxDType.INT32: np.int32,
            OnnxDType.INT64: np.int64,
        }[self]

    @property
    def sadl_type(self):
        """SADL storage type (INT64 is narrowed to INT32)."""
        return {
            OnnxDType.FLOAT: SadlDType.FLOAT,
            OnnxDType.INT8: SadlDType.INT8,
            OnnxDType.INT16: SadlDType.INT16,
            OnnxDType.INT32: SadlDType.INT32,
            OnnxDType.INT64: SadlDType.INT32,
        }[self]

    @classmethod
    def has(cls, value):
        return value in cls._value2member_map_

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
