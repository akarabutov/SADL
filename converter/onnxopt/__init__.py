# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.
"""ONNX model optimization, run on its own or before the ONNX -> SADL conversion.

Python: optimize_onnx(model, ...) or OptimizationPipeline; command line:
converter/optimize_onnx.py. A new pass is an OptimizationPass subclass (or a
function model -> model) added to a pipeline or registered in PassRegistry.
"""

from .api import optimize_onnx  # noqa: F401
from .constant_to_initializer_pass import ConstantToInitializerPass  # noqa: F401
from .function_pass import FunctionPass  # noqa: F401
from .onnxsim_pass import OnnxSimPass  # noqa: F401
from .optimization_pass import OptimizationPass  # noqa: F401
from .optimization_pipeline import OptimizationPipeline  # noqa: F401
from .pass_registry import PassRegistry  # noqa: F401
from .remove_identity_pass import RemoveIdentityPass  # noqa: F401
from .shape_inference_pass import ShapeInferencePass  # noqa: F401
from .split_slices_pass import SplitSlicesPass  # noqa: F401
