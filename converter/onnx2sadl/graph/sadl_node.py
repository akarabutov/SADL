# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class SadlNode:
    """One SADL layer.

    Attributes:
        op_type: OpType of the layer
        inputs:  names (keys of the SadlGraph) of the input layers
        attrs:   layer parameters read by the matching LayerSerializer
                 (e.g. "dims"/"raw_data"/"dtype" for Const, "strides" for Conv2D).
                 "data" keeps the originating ONNX node for debugging.
    """

    def __init__(self, op_type, inputs=None, attrs=None):
        self.op_type = op_type
        self.inputs = list(inputs) if inputs is not None else []
        self.attrs = attrs if attrs is not None else {}

    def __getitem__(self, key):
        return self.attrs[key]

    def __repr__(self):
        return "SadlNode({}, inputs={})".format(self.op_type, self.inputs)
