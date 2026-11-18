# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from collections import OrderedDict

import numpy as np

from ..types.op_type import OpType
from ..types.sadl_dtype import SadlDType
from .sadl_node import SadlNode


class SadlGraph:
    """The SADL graph under construction.

    ``nodes`` keeps insertion order: the position of a name is its layer id in
    the output file. Re-adding an existing name replaces the layer in place.

    ``name_map`` maps an ONNX tensor name to the SADL layer that produces it
    (None for helper entries that must never be referenced).
    """

    def __init__(self):
        self.nodes = OrderedDict()
        self.name_map = {}

    # --- layers -----------------------------------------------------------
    def __contains__(self, name):
        return name in self.nodes

    def __len__(self):
        return len(self.nodes)

    def items(self):
        return self.nodes.items()

    def add(self, name, op_type, inputs=None, attrs=None):
        node = SadlNode(op_type, inputs, attrs)
        self.nodes[name] = node
        return node

    def add_const(self, name, tensor, owner):
        """Add a Const layer holding a ConstTensor."""
        return self.add(name, OpType.Const, [], tensor.to_attrs(owner))

    def add_int32_vector(self, name, values, owner):
        """Add an int32 1-D Const layer (a permutation, a shape, an axis...)."""
        attrs = {
            "dims": [len(values)],
            "raw_data": np.array(values, dtype=np.int32).tobytes(),
            "dtype": SadlDType.INT32,
            "data": owner,
        }
        return self.add(name, OpType.Const, [], attrs)

    def add_transpose(self, source, perm, coef_name, out_name, owner):
        """Add ``out_name = Transpose(resolve(source), Const(perm))``; returns out_name."""
        self.add_int32_vector(coef_name, perm, owner)
        self.bind(coef_name, coef_name)
        self.add(out_name, OpType.Transpose, [self.resolve(source), coef_name], {"data": owner})
        self.bind(out_name, out_name)
        return out_name

    # --- ONNX name -> SADL name -------------------------------------------
    def bind(self, onnx_name, sadl_name):
        self.name_map[onnx_name] = sadl_name

    def resolve(self, onnx_name):
        return self.name_map[onnx_name]
