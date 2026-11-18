# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import onnx

from onnx2sadl.converter import Converter
from onnx2sadl.converter_config import ConverterConfig

from .input_spec import InputSpec
from .pytorch_onnx_exporter import PytorchOnnxExporter


class PytorchToSadlConverter:
    """torch.nn.Module -> ONNX (in memory) -> [optimizer] -> SADL file.

    The SADL model is NHWC: an input of shape [1, C, H, W] in PyTorch becomes
    [1, H, W, C] in SADL (same for 4-D outputs).
    """

    def __init__(self, exporter=None, sadl_config=None, optimizer=None):
        self.exporter = exporter or PytorchOnnxExporter()
        self.sadl_config = sadl_config or ConverterConfig(data_layout="nchw")
        self.optimizer = optimizer

    def convert(self, model, example_inputs, output_path, onnx_path=None, input_names=None, output_names=None):
        """Convert ``model`` and write ``output_path``; returns the ONNX model that was converted.

        ``example_inputs``: tensor(s) or shape(s) used to trace the model.
        ``onnx_path``: optionally keep that ONNX model (after optimization).
        """
        inputs = InputSpec.to_example_inputs(example_inputs)
        proto = self.exporter.export(model, inputs, input_names, output_names)
        if self.optimizer is not None:
            proto = self.optimizer.run(proto)
        if onnx_path:
            onnx.save(proto, onnx_path)
        Converter(self.sadl_config).convert(proto, output_path)
        return proto
