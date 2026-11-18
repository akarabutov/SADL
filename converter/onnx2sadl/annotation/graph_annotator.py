# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..graph.node_annotation import NodeAnnotation


class GraphAnnotator:
    """Tracks the data layout (NCHW/NHWC) through the ONNX graph.

    Starting from the graph inputs, each node inherits the layout of its
    inputs, then the OpConverter registered for its op_type may adjust its
    NodeAnnotation (remove layout transposes, request new ones...).
    A node is visited once all its producers are annotated.
    """

    def __init__(self, model, registry, global_layout, verbose=0):
        self.model = model
        self.registry = registry
        self.global_layout = global_layout
        self.verbose = verbose
        self.annotations = {}

    def annotate(self):
        """Returns {name: NodeAnnotation} for inputs, initializers and reached nodes."""
        model = self.model
        for graph_input in model.inputs:
            annotation = NodeAnnotation()
            if len(graph_input.type.tensor_type.shape.dim) == 4:
                annotation.layout_onnx = self.global_layout
                if self.global_layout == "nchw":
                    annotation.to_transpose = True
            self.annotations[graph_input.name] = annotation

        for init in model.initializers:
            self.annotations[init.name] = NodeAnnotation()

        for node in model.nodes:
            if node.op_type == "Constant":
                self.annotations[node.name] = NodeAnnotation()

        for graph_input in model.inputs:
            for node in model.consumers(graph_input.name):
                self._annotate_node(node)

        if self.verbose > 1:
            self._report_remaining_transposes()
        return self.annotations

    def _inherited_layout(self, node):
        """Layout of the inputs, or False if an input is not annotated yet."""
        layout = None
        for inp in node.input:
            parent = self.model.producer_skip_identity(inp)
            if parent is None:
                continue
            if self.verbose:
                print(f"[INFO]   input: {parent.name}")
            if parent.name not in self.annotations:
                if self.verbose > 2:
                    print(f"[INFO]   input not ready: {parent.name}, abord")
                return False
            parent_layout = self.annotations[parent.name].layout_onnx
            if layout is None:
                layout = parent_layout
            elif parent_layout is not None and parent_layout != layout:
                print(
                    f"[ERROR] inputs with different layout node={layout} "
                    f"input[{parent.name}]={parent_layout} for\n{node}\n"
                )
        return layout

    def _annotate_node(self, node):
        if node.name in self.annotations:
            return
        if self.verbose:
            print("[INFO] annotate_node {} op={}".format(node.name, node.op_type))

        data_layout = self._inherited_layout(node)
        if data_layout is False:
            return
        if self.verbose > 1 and data_layout is None:
            print("[WARNING] no data layout constraints for {}\n {}".format(node.name, node))

        self.annotations[node.name] = NodeAnnotation(layout_onnx=data_layout)
        converter = self.registry.get(node.op_type)
        if converter is not None:
            converter.annotate(self, node, data_layout)

        for consumer in self.model.consumers(node.output[0]):
            self._annotate_node(consumer)

    def _report_remaining_transposes(self):
        for node in self.model.nodes:
            if node.op_type == "Transpose" and (
                node.name not in self.annotations or not self.annotations[node.name].to_remove
            ):
                print(
                    "[ERROR] preprocess_onnxGraph: all transpose node should be removed "
                    "but this is not the case here: {}\n{}".format(node.name, node)
                )
