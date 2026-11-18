# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.
"""PyTorch -> SADL conversion (torch.onnx export, then onnx2sadl).

Python API: convert_pytorch_to_sadl(model, example_inputs, "model.sadl").
Command line: converter/pytorch_to_sadl.py.
"""

from .api import convert_pytorch_to_sadl, export_pytorch_to_onnx  # noqa: F401
from .input_spec import InputSpec  # noqa: F401
from .pytorch_to_sadl_converter import PytorchToSadlConverter  # noqa: F401
