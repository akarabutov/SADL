# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class OpRegistry:
    """Maps an ONNX op_type to the OpConverter instance that handles it."""

    def __init__(self, converters=()):
        self._by_op = {}
        for converter in converters:
            self.register(converter)

    def register(self, converter):
        for op in converter.onnx_ops:
            if op in self._by_op:
                raise ValueError("ONNX op {} registered twice".format(op))
            self._by_op[op] = converter

    def get(self, onnx_op):
        return self._by_op.get(onnx_op)

    def supported_ops(self):
        return sorted(self._by_op)
