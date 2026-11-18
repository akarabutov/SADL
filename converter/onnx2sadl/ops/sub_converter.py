# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .binary_converter import BinaryConverter


class SubConverter(BinaryConverter):
    """Not commutative: inputs are never swapped (the constant should be second)."""

    onnx_ops = ("Sub",)
    sadl_op = OpType.Sub
