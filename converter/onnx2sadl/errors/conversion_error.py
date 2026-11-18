# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class ConversionError(Exception):
    """Raised when the ONNX model cannot be converted to SADL."""
