# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import inspect
import io

import onnx
import torch

from onnx2sadl.errors.conversion_error import ConversionError


class PytorchOnnxExporter:
    """Exports a torch.nn.Module to an in-memory ONNX model.

    Uses the TorchScript-based exporter (``dynamo=False``): it produces the
    classic op patterns (opset <= 17) the SADL converter is tested with.
    """

    DEFAULT_OPSET = 13

    def __init__(self, opset=DEFAULT_OPSET, do_constant_folding=True, verbose=0):
        self.opset = opset
        self.do_constant_folding = do_constant_folding
        self.verbose = verbose

    def export(self, model, example_inputs, input_names=None, output_names=None):
        """Returns the onnx.ModelProto of ``model`` traced with ``example_inputs`` (a tuple)."""
        if not isinstance(model, torch.nn.Module):
            raise ConversionError("[ERROR] expected a torch.nn.Module, got {}".format(type(model).__name__))
        for x in example_inputs:
            if x.dim() > 0 and x.shape[0] != 1:
                raise ConversionError("[ERROR] SADL supports a batch size of 1, got input shape {}".format(
                    tuple(x.shape)))

        was_training = model.training
        model.eval()
        buffer = io.BytesIO()
        kwargs = dict(
            opset_version=self.opset,
            do_constant_folding=self.do_constant_folding,
            input_names=input_names,
            output_names=output_names,
            verbose=self.verbose > 1,
        )
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            kwargs["dynamo"] = False
        try:
            with torch.no_grad():
                torch.onnx.export(model, tuple(example_inputs), buffer, **kwargs)
        finally:
            model.train(was_training)
        proto = onnx.load_from_string(buffer.getvalue())
        if self.verbose:
            print("[INFO] PyTorch model exported to ONNX (opset {}, {} nodes)".format(
                self.opset, len(proto.graph.node)))
        return proto
