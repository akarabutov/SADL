# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .unary_converter import UnaryConverter


class ReluConverter(UnaryConverter):
    onnx_ops = ("Relu",)
    sadl_op = OpType.Relu
