# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import struct


class BinaryStream:
    """Little helper over a binary file: typed writes with optional verbose trace."""

    def __init__(self, f, verbose=0):
        self.f = f
        self.verbose = verbose

    def log(self, *message):
        if self.verbose:
            print(*message)

    def i32(self, value, label=None):
        if label is not None:
            self.log(f"#\t {label} {value}")
        self.f.write(struct.pack("i", int(value)))

    def i16(self, value):
        self.f.write(struct.pack("h", int(value)))

    def bool(self, value, label=None):
        if label is not None:
            self.log(f"#\t {label} {value}")
        self.f.write(struct.pack("?", bool(value)))

    def raw(self, data):
        self.f.write(data)

    def i32_list(self, values, label):
        """Count then values: [len(values)] values..."""
        self.i32(len(values), label)
        self.i32_values(values)

    def i32_values(self, values):
        """Values only, without their count."""
        for v in values:
            self.log(f"#\t\t {v}")
            self.f.write(struct.pack("i", int(v)))
