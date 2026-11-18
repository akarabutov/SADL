# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from ..graph.sadl_graph import SadlGraph
from .conversion_context import ConversionContext
from .graph_input_converter import GraphInputConverter
from .transpose_inserter import TransposeInserter


class OnnxGraphParser:
    """Builds a SadlGraph from an annotated OnnxModel.

    Each ONNX node is handed to the OpConverter registered for its op_type.
    """

    def __init__(
        self,
        model,
        annotations,
        registry,
        input_default_value,
        verbose=0,
        force_transpose_constant_tensor_dim4=False,
    ):
        self.model = model
        self.annotations = annotations
        self.registry = registry
        self.verbose = verbose
        self.input_converter = GraphInputConverter(input_default_value)
        self.ctx = ConversionContext(
            model, SadlGraph(), annotations, verbose, force_transpose_constant_tensor_dim4
        )
        self.transposer = TransposeInserter(self.ctx)

    def parse(self):
        """Returns (graph, input layer names, output layer names)."""
        graph = self.ctx.graph
        for graph_input in self.model.inputs:
            self.input_converter.convert(
                graph, graph_input, self.annotations[graph_input.name].to_transpose
            )

        for node in self.model.nodes:
            if node.name in self.annotations and self.annotations[node.name].to_remove:
                self._remove_node(node)
            else:
                self._convert_node(node)

        inputs = [i.name for i in self.model.inputs]
        outputs = [graph.name_map[o.name] for o in self.model.outputs if o.name in graph.name_map]
        return graph, inputs, outputs

    def _remove_node(self, node):
        """A removed node aliases its output to the (resolved) first input."""
        name_map = self.ctx.graph.name_map
        key = node.input[0]
        while name_map[key] is not None and name_map[key] != key:
            key = name_map[key]
            if key not in name_map:
                key = node.input[0]
                break
        name_map[node.output[0]] = key

    def _resolve_inputs(self, node):
        """Names of inputs 0..2, after optional layout transposes."""
        annotation = self.annotations[node.name]
        layouts = [
            annotation.transpose_before_in0,
            annotation.transpose_before_in1,
            # NOTE(refactoring): input 2 uses transpose_before_in1, there is no
            # transpose_before_in2. Kept as in the original converter; it may be a
            # typo, but changing it could alter the output of 3-input nodes
            # (Conv/Gemm bias, Where, Clip, ScatterND...).
            annotation.transpose_before_in1,
        ]
        names = []
        for i, layout in enumerate(layouts):
            if i >= len(node.input):
                names.append(None)
            # NOTE(refactoring): the check is "is not None", so the False written by
            # --do_not_add_transpose_before does NOT disable the transpose: it is
            # inserted with the NCHW->NHWC permutation (see TransposeInserter).
            elif layout is not None:
                names.append(self.transposer.before_input(node, layout, i))
            else:
                names.append(node.input[i])
        return names

    def _convert_node(self, node):
        if node.name not in self.annotations:
            print(f"[WARN] node {node.name} never reached: removed")
            return
        if self.verbose > 1:
            print(f"parse node {node.name} remove={self.annotations[node.name].to_remove}")

        inputs = self._resolve_inputs(node)
        converter = self.registry.get(node.op_type)
        if converter is None:
            raise ConversionError("[ERROR] node not supported:\n{})".format(node))
        converter.convert(self.ctx, node, inputs)

        if self.annotations[node.name].to_nhwc_after_out:
            self.transposer.after_output(node)
