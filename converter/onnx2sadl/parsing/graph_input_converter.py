# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType


class GraphInputConverter:
    """Turns an ONNX graph input into a SADL Placeholder layer."""

    def __init__(self, input_default_value):
        self.input_default_value = input_default_value

    def convert(self, graph, graph_input, to_transpose):
        graph.bind(graph_input.name, graph_input.name)
        dim = graph_input.type.tensor_type.shape.dim
        if to_transpose:  # NCHW -> NHWC
            dims = [dim[0].dim_value, dim[2].dim_value, dim[3].dim_value, dim[1].dim_value]
        else:
            dims = [d.dim_value for d in dim]
        # named (symbolic) dimensions are replaced by a default value
        dims = [self.input_default_value if d == 0 else d for d in dims]
        graph.add(graph_input.name, OpType.Placeholder, [], {"dims": dims})
