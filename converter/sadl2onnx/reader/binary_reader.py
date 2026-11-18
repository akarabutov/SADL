# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import struct

import numpy as np

from onnx2sadl.errors.conversion_error import ConversionError


class BinaryReader:
    """Typed little-endian reads over an in-memory SADL file."""

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def _take(self, size):
        if self.pos + size > len(self.data):
            raise ConversionError("[ERROR] unexpected end of SADL file at byte {}".format(self.pos))
        chunk = self.data[self.pos:self.pos + size]
        self.pos += size
        return chunk

    def bytes(self, size):
        return self._take(size)

    def i32(self):
        return struct.unpack("<i", self._take(4))[0]

    def i16(self):
        return struct.unpack("<h", self._take(2))[0]

    def u8(self):
        return self._take(1)[0]

    def bool(self):
        return self.u8() != 0

    def i32_list(self):
        """Count then values."""
        count = self.i32()
        return [self.i32() for _ in range(count)]

    def array(self, dtype, count):
        dtype = np.dtype(dtype)
        return np.frombuffer(self._take(dtype.itemsize * count), dtype=dtype).copy()

    def at_end(self):
        return self.pos == len(self.data)
