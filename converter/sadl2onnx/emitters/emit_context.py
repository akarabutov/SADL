# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from onnx2sadl.errors.conversion_error import ConversionError
from onnx2sadl.types.op_type import OpType

from .rank_tracker import RankTracker


class EmitContext:
    """State shared by the LayerEmitters: the SADL model, the ONNX builder and
    the ONNX value produced by each SADL layer.

    Const layers are not emitted eagerly: an initializer is created the first
    time a layer consumes the constant as a tensor (see ``value``), while
    parameters (shapes, axes, permutations...) are read with ``const``.
    """

    def __init__(self, model, builder, verbose=0):
        self.model = model
        self.builder = builder
        self.verbose = verbose
        self.shapes = RankTracker(model)
        self.values = {}
        self._initializers = {}

    def layer(self, layer_id):
        return self.model.layer(layer_id)

    def is_const(self, layer_id):
        layer = self.layer(layer_id)
        if layer.op_type == OpType.Identity:
            return self.is_const(layer.inputs[0])
        return layer.op_type == OpType.Const

    def const(self, layer_id, dtype=None):
        """Real value of a Const layer (fixed point integers are de-quantized for float dtypes)."""
        layer = self.layer(layer_id)
        if layer.op_type == OpType.Identity:
            return self.const(layer.inputs[0], dtype)
        if layer.op_type != OpType.Const:
            raise ConversionError("[ERROR] layer {} ({}) must be a Const".format(layer.name, layer.op_type))
        value = layer["value"]
        if dtype is None:
            return value
        dtype = np.dtype(dtype)
        if dtype.kind == "f" and value.dtype.kind == "i" and layer["quantizer"]:
            return (value.astype(np.float64) * 2.0 ** (-layer["quantizer"])).astype(dtype)
        return value.astype(dtype)

    def value(self, layer_id, dtype=np.float32):
        """ONNX value name of a layer output (Const layers become initializers)."""
        if self.is_const(layer_id):
            key = (layer_id, np.dtype(dtype).str)
            if key not in self._initializers:
                self._initializers[key] = self.builder.add_initializer(
                    self.const(layer_id, dtype), self.layer(layer_id).name
                )
            return self._initializers[key]
        if layer_id not in self.values:
            raise ConversionError("[ERROR] layer {} used before being emitted".format(layer_id))
        return self.values[layer_id]

    def int64_input(self, layer_id):
        """A shape-like input: int64 initializer for a Const, else the computed value."""
        if self.is_const(layer_id):
            return self.value(layer_id, np.int64)
        return self.values[layer_id]
