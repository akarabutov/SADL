# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .compare_converter import CompareConverter


class GreaterConverter(CompareConverter):
    onnx_ops = ("Greater",)
    mode = 1
