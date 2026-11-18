# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .binary_converter import BinaryConverter


class MinConverter(BinaryConverter):
    onnx_ops = ("Min",)
    sadl_op = OpType.Minimum
    const_to_transpose = "producer"
