# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .pool_converter import PoolConverter


class AveragePoolConverter(PoolConverter):
    onnx_ops = ("AveragePool",)
    sadl_op = OpType.AveragePool
