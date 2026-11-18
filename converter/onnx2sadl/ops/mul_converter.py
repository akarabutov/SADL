# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .binary_converter import BinaryConverter


class MulConverter(BinaryConverter):
    onnx_ops = ("Mul",)
    sadl_op = OpType.Mul
    const_to_transpose = "producer"
    reject_two_constants = True
    const_first_swaps = True
