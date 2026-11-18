# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class SadlLayer:
    """A layer read from a SADL file.

    Attributes:
        id:       layer id (index in the file)
        op_type:  onnx2sadl OpType
        name:     layer name
        inputs:   ids of the input layers
        params:   layer-specific parameters (see LayerParamsReader subclasses);
                  Const layers hold their data in ``params["value"]`` (numpy array)
    """

    def __init__(self, layer_id, op_type, name, inputs, params=None):
        self.id = layer_id
        self.op_type = op_type
        self.name = name
        self.inputs = inputs
        self.params = params if params is not None else {}

    def __getitem__(self, key):
        return self.params[key]

    def __repr__(self):
        return "SadlLayer({}, {}, {!r}, inputs={})".format(self.id, self.op_type, self.name, self.inputs)
