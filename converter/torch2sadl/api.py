# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

"""Python entry points of the PyTorch -> SADL conversion.

Example::

    import sys
    sys.path.append("path/to/sadl/converter")
    from torch2sadl import convert_pytorch_to_sadl

    convert_pytorch_to_sadl(model, (1, 3, 64, 64), "model.sadl", info_string="my model v1")
"""

from onnx2sadl.converter_config import ConverterConfig
from onnx2sadl.graph.node_annotation import NodeAnnotation

from .pytorch_onnx_exporter import PytorchOnnxExporter
from .pytorch_to_sadl_converter import PytorchToSadlConverter


def convert_pytorch_to_sadl(
    model,
    example_inputs,
    output_path,
    *,
    info_string="",
    onnx_path=None,
    opset=PytorchOnnxExporter.DEFAULT_OPSET,
    input_names=None,
    output_names=None,
    input_default_value=128,
    force_transpose_constant_tensor_dim4=False,
    do_not_add_transpose_before=(),
    do_not_add_transpose_after=(),
    verbose=0,
    optimize=False,
):
    """Convert a torch.nn.Module to a SADL file.

    Args:
        model:          the torch.nn.Module (set to eval mode during export)
        example_inputs: tensor, tuple of tensors, or input shape(s) such as
                        ``(1, 3, 64, 64)``; the batch size must be 1
        output_path:    the .sadl file to write
        info_string:    free text stored in the SADL header
        onnx_path:      if given, also save the intermediate ONNX model there
                        (after optimization: the model actually converted)
        opset:          ONNX opset used for the export
        optimize:       ONNX optimization before conversion (package onnxopt):
                        False, True (default passes), a list of pass names /
                        OptimizationPass / functions, or an OptimizationPipeline
        other options:  same as converter/main.py (see ConverterConfig)

    Returns:
        the converted onnx.ModelProto.

    Note: SADL is NHWC; 4-D inputs/outputs [1, C, H, W] become [1, H, W, C].
    """
    user_annotations = {}
    for name in do_not_add_transpose_before:
        user_annotations.setdefault(name, NodeAnnotation.user_override()).transpose_before_in0 = False
    for name in do_not_add_transpose_after:
        user_annotations.setdefault(name, NodeAnnotation.user_override()).to_nhwc_after_out = False
    config = ConverterConfig(
        data_layout="nchw",
        verbose=verbose,
        user_annotations=user_annotations,
        input_default_value=input_default_value,
        force_transpose_constant_tensor_dim4=force_transpose_constant_tensor_dim4,
        info_string=info_string,
    )
    exporter = PytorchOnnxExporter(opset=opset, verbose=verbose)
    converter = PytorchToSadlConverter(exporter, config, _build_optimizer(optimize, verbose))
    return converter.convert(model, example_inputs, output_path, onnx_path, input_names, output_names)


def _build_optimizer(optimize, verbose):
    """OptimizationPipeline (or None) from the ``optimize`` argument."""
    if optimize is False or optimize is None:
        return None
    if hasattr(optimize, "run"):  # a ready pipeline
        return optimize
    from onnxopt.optimization_pipeline import OptimizationPipeline
    from onnxopt.pass_registry import PassRegistry

    if optimize is True:
        return OptimizationPipeline.from_names(verbose=verbose)
    registry = PassRegistry()
    pipeline = OptimizationPipeline(verbose=verbose)
    for item in optimize:
        pipeline.add(registry.create(item) if isinstance(item, str) else item)
    return pipeline


def export_pytorch_to_onnx(model, example_inputs, onnx_path=None, opset=PytorchOnnxExporter.DEFAULT_OPSET):
    """Only the first step: the ONNX model the SADL converter would receive."""
    import onnx

    from .input_spec import InputSpec

    proto = PytorchOnnxExporter(opset=opset).export(model, InputSpec.to_example_inputs(example_inputs))
    if onnx_path:
        onnx.save(proto, onnx_path)
    return proto
