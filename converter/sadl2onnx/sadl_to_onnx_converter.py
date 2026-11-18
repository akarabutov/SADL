# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import onnx

from onnx2sadl.errors.conversion_error import ConversionError
from onnx2sadl.types.op_type import OpType

from .builder.onnx_graph_builder import OnnxGraphBuilder
from .emitters.emit_context import EmitContext
from .emitters.emitter_registry import EmitterRegistry
from .reader.sadl_reader import SadlReader


class SadlToOnnxConverter:
    """SADL file -> ONNX model.

    The ONNX graph keeps SADL's NHWC layout: inputs and outputs have the
    shapes SADL uses (an NCHW PyTorch model comes back with NHWC inputs).
    """

    OPSET = 17

    def __init__(self, verbose=0, optimize_transposes=True, registry=None, reader=None):
        self.verbose = verbose
        self.optimize_transposes = optimize_transposes
        self.registry = registry or EmitterRegistry()
        self.reader = reader or SadlReader(verbose=verbose)

    def convert_file(self, sadl_path, onnx_path):
        model = self.convert(self.reader.read(sadl_path))
        onnx.save(model, onnx_path)
        return model

    def convert(self, sadl_model):
        if sadl_model.model_type.name != "FLOAT" and self.verbose:
            print("[WARN] model type {}: the ONNX model is generated in float".format(sadl_model.model_type))
        builder = OnnxGraphBuilder(optimize_transposes=self.optimize_transposes)
        ctx = EmitContext(sadl_model, builder, self.verbose)

        for layer_id in sadl_model.input_ids:
            self._emit(ctx, layer_id, set())
        for layer_id in sadl_model.output_ids:
            self._emit(ctx, layer_id, set())
        for layer_id in sadl_model.output_ids:
            layer = sadl_model.layer(layer_id)
            builder.add_output(ctx.value(layer_id), layer.name + "_output")

        return builder.build(opset=self.OPSET, doc_string=sadl_model.info)

    def _emit(self, ctx, layer_id, visiting):
        """Emit ``layer_id`` after its inputs (depth first)."""
        if layer_id in ctx.values:
            return
        if layer_id in visiting:
            raise ConversionError("[ERROR] cycle in the SADL graph at layer {}".format(layer_id))
        visiting.add(layer_id)
        layer = ctx.layer(layer_id)
        if layer.op_type != OpType.Const:
            for input_id in layer.inputs:
                self._emit(ctx, input_id, visiting)
        emitter = self.registry.get(layer.op_type)
        if emitter is None:
            raise ConversionError("[ERROR] SADL layer {} not supported: {}".format(layer.op_type, layer.name))
        if self.verbose > 1:
            print("[INFO] emit", layer)
        ctx.values[layer_id] = emitter.emit(ctx, layer)
        visiting.discard(layer_id)
