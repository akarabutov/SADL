# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .compare_converter import CompareConverter


class EqualConverter(CompareConverter):
    onnx_ops = ("Equal",)
    mode = 2
