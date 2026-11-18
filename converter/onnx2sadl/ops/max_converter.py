# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .binary_converter import BinaryConverter


class MaxConverter(BinaryConverter):
    onnx_ops = ("Max",)
    sadl_op = OpType.Maximum
    const_to_transpose = "producer"
