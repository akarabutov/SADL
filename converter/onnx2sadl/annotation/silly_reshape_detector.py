# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..model.constant_extractor import ConstantExtractor
from ..model.onnx_model import OnnxModel


class SillyReshapeDetector:
    """Detects ``reshape(x, shape(n))`` exported as

        concat(unsqueeze(gather(shape(n), 0)), ..., unsqueeze(gather(shape(n), 3)))

    and simplifies it: the Concat/Unsqueeze/Gather/Shape chain is removed and
    the consumers of the Concat read ``shape(n)`` directly.
    """

    def __init__(self, model, annotations, verbose=0):
        self.model = model
        self.annotations = annotations
        self.verbose = verbose

    def detect_and_simplify(self, node):
        shape = self._match(node)
        if shape is None:
            return False
        self._remove_chain(node, shape)
        self.model.replace_input(node.output[0], shape.output[0], self.verbose)
        return True

    def _match(self, node):
        """The common Shape node if ``node`` is the silly pattern, else None."""
        model = self.model
        if node.op_type != "Concat":
            return None
        if OnnxModel.get_attribute(node, "axis").i != 0 or len(node.input) != 4:
            return None
        parent = None
        shape = None
        for i in range(4):
            unsqueeze = model.producer_skip_identity(node.input[i])
            if unsqueeze.op_type != "Unsqueeze":
                return None
            if OnnxModel.get_attribute(unsqueeze, "axes").ints != [0]:
                return None
            gather = model.producer_skip_identity(unsqueeze.input[0])
            if gather.op_type != "Gather" or len(gather.input) != 2:
                return None
            index = ConstantExtractor.attribute_values(model.producer_skip_identity(gather.input[1]))
            if len(index) != 1 or int(index[0]) != i:
                return None
            shape_node = model.producer_skip_identity(gather.input[0])
            if shape_node.op_type != "Shape":
                return None
            if parent is None:
                shape = shape_node
            p = model.producer_skip_identity(shape_node.input[0])
            if parent is not None and p.name != parent.name:  # all Shape on the same tensor
                return None
            parent = p
        return shape

    def _remove_chain(self, node, shape):
        model = self.model
        self.annotations[node.name].to_remove = True
        for i in range(4):
            unsqueeze = model.producer_skip_identity(node.input[i])
            self.annotations[unsqueeze.name].to_remove = True
            gather = model.producer_skip_identity(unsqueeze.input[0])
            self.annotations[gather.name].to_remove = True
            shape_node = model.producer_skip_identity(gather.input[0])
            if shape_node.name != shape.name:
                self.annotations[shape_node.name].to_remove = True
