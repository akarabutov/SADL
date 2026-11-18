# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx import helper, numpy_helper

from .optimization_pass import OptimizationPass


class ConstantToInitializerPass(OptimizationPass):
    """Moves ``Constant`` nodes holding a tensor into graph initializers.

    Recent exporters put weights or parameters in Constant nodes, while some
    converters (including parts of onnx2sadl) expect initializers.
    """

    name = "constant_to_initializer"
    description = "turn Constant nodes into initializers"

    def run(self, model):
        graph = model.graph
        kept = []
        moved = 0
        for node in graph.node:
            if node.op_type == "Constant" and len(node.attribute) == 1 and node.attribute[0].name == "value":
                tensor = helper.get_attribute_value(node.attribute[0])
                array = numpy_helper.to_array(tensor)
                graph.initializer.append(numpy_helper.from_array(array, node.output[0]))
                moved += 1
            else:
                kept.append(node)
        if moved:
            del graph.node[:]
            graph.node.extend(kept)
        return model
