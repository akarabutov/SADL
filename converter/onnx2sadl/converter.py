# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import copy

import numpy as np
import onnx

from .annotation.graph_annotator import GraphAnnotator
from .errors.conversion_error import ConversionError
from .model.onnx_model import OnnxModel
from .ops.op_catalog import OpCatalog
from .parsing.onnx_graph_parser import OnnxGraphParser
from .writer.sadl_writer import SadlWriter


class Converter:
    """ONNX model -> SADL file.

    Pipeline: shape inference -> layout annotation -> SADL graph -> binary file.
    """

    PRODUCER_LAYOUTS = {"tf2onnx": "nhwc", "pytorch": "nchw"}

    def __init__(self, config, registry=None, writer=None):
        self.config = config
        self.registry = registry or OpCatalog.build_registry()
        self.writer = writer or SadlWriter(config.verbose)

    def convert(self, model_proto, output_path):
        """Convert ``model_proto`` (left untouched) and write ``output_path``."""
        config = self.config
        verbose = config.verbose
        model = OnnxModel(self._prepare(model_proto))

        layout = config.data_layout or self._detect_layout(model)
        if verbose:
            print("[INFO] assume data type", layout)
        if verbose > 1:
            self._print_graph_without_data(model_proto)
        if layout not in ("nhwc", "nchw"):
            raise ConversionError("[ERROR] unsupported layout {}".format(layout))

        annotations = GraphAnnotator(model, self.registry, layout, verbose).annotate()
        self._apply_user_annotations(annotations)
        if verbose > 1:
            lines = "\n".join("{!r}: {!r},".format(k, v) for k, v in annotations.items())
            print("INFO] annotations:\n{" + lines + "}")

        parser = OnnxGraphParser(
            model,
            annotations,
            self.registry,
            config.input_default_value,
            verbose,
            config.force_transpose_constant_tensor_dim4,
        )
        graph, inputs, outputs = parser.parse()
        self.writer.write(output_path, graph, inputs, outputs, config.info_string)

        if layout == "nchw":
            print("[INFO] in SADL, your inputs and outputs has been changed from NCHW to NHWC")

    def _prepare(self, model_proto):
        """Deep copy (the conversion mutates the graph) with inferred shapes."""
        model_copy = copy.deepcopy(model_proto)
        try:
            model_copy = onnx.shape_inference.infer_shapes(model_copy)
        except Exception as e:
            if self.config.verbose:
                print("[WARN] ONNX shape_inference failed, continuing without it:", e)
        return model_copy

    def _detect_layout(self, model):
        layout = self.PRODUCER_LAYOUTS.get(model.producer_name)
        if layout is None:
            raise ConversionError("[ERROR] unable to detect data layout")
        return layout

    def _apply_user_annotations(self, annotations):
        for name, override in self.config.user_annotations.items():
            if name not in annotations:
                raise ConversionError("[ERROR] unknown node user custom {}".format(name))
            annotations[name].apply_override(override)

    @staticmethod
    def _print_graph_without_data(model_proto):
        graph = copy.deepcopy(model_proto.graph)
        for init in graph.initializer:
            init.raw_data = np.array(0.0).tobytes()
        print("[INFO] original graph:\n", graph)
