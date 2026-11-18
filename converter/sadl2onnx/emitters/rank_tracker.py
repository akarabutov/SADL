# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType


class RankTracker:
    """Rank of each SADL layer output, following the C++ init() rules.

    Only the rank is tracked: it is all some emitters need (e.g. Slice on a
    2-D or a 4-D tensor) and it does not depend on symbolic sizes.
    """

    FIXED_RANK = {
        OpType.Flatten: 2,
        OpType.Shape: 1,
        OpType.Conv2D: 4,
        OpType.Conv2DTranspose: 4,
        OpType.MaxPool: 4,
        OpType.AveragePool: 4,
        OpType.GlobalAveragePool: 4,
        OpType.GridSample: 4,
        OpType.Resize: 4,
        OpType.Depth2Space: 4,
        OpType.Pad: 4,
    }

    def __init__(self, model):
        self.model = model
        self._ranks = {}

    def rank(self, layer_id):
        if layer_id not in self._ranks:
            self._ranks[layer_id] = self._compute(self.model.layer(layer_id))
        return self._ranks[layer_id]

    def _compute(self, layer):
        op = layer.op_type
        if op in self.FIXED_RANK:
            return self.FIXED_RANK[op]
        if op in (OpType.Placeholder, OpType.Const):
            return len(layer["dims"])
        if op in (OpType.Reshape, OpType.Expand):
            return self._shape_length(layer.inputs[1])
        if op == OpType.Transpose:
            return self._shape_length(layer.inputs[1])  # one entry per output dim
        if op == OpType.Gather:
            r = self.rank(layer.inputs[0])
            return r - 1 if layer["indices_is_scalar"] else r + self.rank(layer.inputs[1]) - 1
        if op == OpType.GatherElements:
            return self.rank(layer.inputs[1])
        if op == OpType.Where:
            cond = self.model.layer(layer.inputs[0])
            single = cond.op_type == OpType.Const and cond["value"].size == 1
            return self.rank(layer.inputs[1] if single else layer.inputs[0])
        return self.rank(layer.inputs[0])

    def _shape_length(self, shape_id):
        """Number of elements of a shape vector."""
        shape = self.model.layer(shape_id)
        if shape.op_type == OpType.Const:
            return int(shape["value"].size)
        if shape.op_type == OpType.Shape:
            return self.rank(shape.inputs[0])
        return 4
