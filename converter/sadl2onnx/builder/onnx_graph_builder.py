# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .transpose_optimizer import TransposeOptimizer


class OnnxGraphBuilder:
    """Accumulates ONNX nodes, initializers, inputs and outputs with unique names."""

    def __init__(self, graph_name="sadl_model", optimize_transposes=True):
        self.graph_name = graph_name
        self.optimize_transposes = optimize_transposes
        self.nodes = []
        self.initializers = []
        self.inputs = []
        self.outputs = []
        self._names = set()

    # --- names ------------------------------------------------------------
    def unique(self, hint):
        base = re.sub(r"[^0-9A-Za-z_./:-]", "_", hint or "v") or "v"
        name, k = base, 1
        while name in self._names:
            name = "{}_{}".format(base, k)
            k += 1
        self._names.add(name)
        return name

    # --- graph content ----------------------------------------------------
    def add_input(self, hint, dims, elem_type=TensorProto.FLOAT):
        name = self.unique(hint)
        self.inputs.append(helper.make_tensor_value_info(name, elem_type, dims))
        return name

    def add_output(self, value, hint, elem_type=TensorProto.FLOAT):
        """Expose ``value`` as a graph output (through an Identity named ``hint``)."""
        name = self.add_node("Identity", [value], hint)
        self.outputs.append(helper.make_tensor_value_info(name, elem_type, None))
        return name

    def add_initializer(self, array, hint):
        name = self.unique(hint)
        self.initializers.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def add_node(self, op_type, inputs, hint, nb_outputs=1, **attrs):
        """Add a node; returns its output name (or the list of names if nb_outputs > 1)."""
        outputs = [self.unique(hint if k == 0 else "{}_out{}".format(hint, k)) for k in range(nb_outputs)]
        node_name = self.unique("{}_{}".format(op_type, hint))
        self.nodes.append(helper.make_node(op_type, list(inputs), outputs, name=node_name, **attrs))
        return outputs[0] if nb_outputs == 1 else outputs

    def transpose(self, value, perm, hint):
        return self.add_node("Transpose", [value], hint, perm=list(perm))

    # --- result -----------------------------------------------------------
    def build(self, opset=17, producer_name="sadl2onnx", doc_string=""):
        nodes = self.nodes
        if self.optimize_transposes:
            nodes = TransposeOptimizer(nodes, [o.name for o in self.outputs]).run()
        graph = helper.make_graph(
            nodes, self.graph_name, self.inputs, self.outputs, self.initializers, doc_string=doc_string
        )
        model = helper.make_model(
            graph, producer_name=producer_name, opset_imports=[helper.make_opsetid("", opset)]
        )
        model.ir_version = min(model.ir_version, 8)
        model = self._with_output_shapes(model)
        onnx.checker.check_model(model)
        return model

    @staticmethod
    def _with_output_shapes(model):
        """Fill the graph output shapes from ONNX shape inference."""
        inferred = onnx.shape_inference.infer_shapes(model)
        known = {vi.name: vi for vi in list(inferred.graph.value_info) + list(inferred.graph.output)}
        for output in model.graph.output:
            vi = known.get(output.name)
            if vi is not None and vi.type.tensor_type.HasField("shape"):
                output.type.CopyFrom(vi.type)
            else:  # unknown shape: unknown rank is expressed with an empty shape message
                output.type.tensor_type.shape.SetInParent()
                output.type.tensor_type.shape.ClearField("dim")
        return model
