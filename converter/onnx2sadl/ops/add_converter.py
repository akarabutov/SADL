# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .binary_converter import BinaryConverter


class AddConverter(BinaryConverter):
    """SADL expects the constant / lower-rank operand second."""

    onnx_ops = ("Add",)
    sadl_op = OpType.Add
    const_first_swaps = True
    lower_rank_swaps = True
