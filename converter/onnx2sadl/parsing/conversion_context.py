# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..model.constant_extractor import ConstantExtractor


class ConversionContext:
    """Everything an OpConverter needs while turning one ONNX node into SADL layers."""

    def __init__(self, model, graph, annotations, verbose=0, force_transpose_constant_tensor_dim4=False):
        self.model = model
        self.graph = graph
        self.annotations = annotations
        self.verbose = verbose
        self.force_transpose_constant_tensor_dim4 = force_transpose_constant_tensor_dim4
        self.extractor = ConstantExtractor(model, verbose)

    # --- annotations ------------------------------------------------------
    def annotation(self, name):
        return self.annotations[name]

    def producer_to_transpose(self, name):
        """``to_transpose`` flag of the object producing ``name``."""
        return self.annotations[self.model.producer(name).name].to_transpose

    # --- constants --------------------------------------------------------
    def is_initializer(self, name):
        return self.model.is_initializer(name)

    def extract_const(self, name, to_transpose=False, allow_force_4d=False):
        """Read a constant; with ``allow_force_4d`` honour --force_transpose_constant_tensor_dim4."""
        tensor = self.extractor.extract(name, to_transpose)
        if allow_force_4d and self.force_transpose_constant_tensor_dim4 and len(tensor.dims) == 4:
            tensor = tensor.transposed_4d()  # nchw -> nhwc
        return tensor

    def add_const(self, name, tensor, owner):
        """Add ``name`` as a Const layer and bind the ONNX name to it."""
        self.graph.bind(name, name)
        self.graph.add_const(name, tensor, owner)
        return name
