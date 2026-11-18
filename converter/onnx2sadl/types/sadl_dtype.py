# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from enum import IntEnum


class SadlDType(IntEnum):
    """Tensor element types of the SADL binary format."""

    INT32 = 0
    FLOAT = 1
    INT16 = 2
    INT8 = 3

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
