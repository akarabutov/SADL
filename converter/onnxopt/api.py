# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

"""Python entry point of the ONNX optimizer.

Example::

    import sys
    sys.path.append("path/to/sadl/converter")
    from onnxopt import optimize_onnx

    def fold_my_pattern(model):          # any function model -> model is a pass
        ...
        return model

    optimized = optimize_onnx("model.onnx", "model_opt.onnx", extra_passes=[fold_my_pattern])
"""

import onnx

from .optimization_pipeline import OptimizationPipeline


def optimize_onnx(
    model,
    output_path=None,
    *,
    passes=None,
    extra_passes=(),
    input_shapes=None,
    check=True,
    continue_on_error=False,
    verbose=0,
):
    """Optimize an ONNX model.

    Args:
        model:         onnx.ModelProto or path to an .onnx file (not modified)
        output_path:   if given, the optimized model is saved there
        passes:        names of registered passes (default: PassRegistry.DEFAULT_PASSES)
        extra_passes:  OptimizationPass instances or functions ``model -> model``
                       run after ``passes``
        input_shapes:  {input name: shape} to fix dynamic input shapes (onnxsim)

    Returns:
        the optimized onnx.ModelProto.
    """
    if isinstance(model, str):
        model = onnx.load(model)
    options = {"onnxsim": {"input_shapes": input_shapes}} if input_shapes else None
    pipeline = OptimizationPipeline.from_names(
        passes, pass_options=options, check=check, continue_on_error=continue_on_error, verbose=verbose
    )
    for extra in extra_passes:
        pipeline.add(extra)
    optimized = pipeline.run(model)
    if output_path:
        onnx.save(optimized, output_path)
    return optimized
