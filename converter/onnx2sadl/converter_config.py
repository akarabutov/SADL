# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class ConverterConfig:
    """Options of one ONNX -> SADL conversion.

    Attributes:
        data_layout:          "nchw", "nhwc" or None (detected from the producer)
        verbose:              verbosity level (0 = quiet)
        user_annotations:     {node name: NodeAnnotation.user_override()} forced by the user
        input_default_value:  value replacing symbolic (named) input dimensions
        force_transpose_constant_tensor_dim4: transpose 4-D constants NCHW -> NHWC
        info_string:          free text stored in the model header
    """

    def __init__(
        self,
        data_layout=None,
        verbose=0,
        user_annotations=None,
        input_default_value=128,
        force_transpose_constant_tensor_dim4=False,
        info_string="",
    ):
        self.data_layout = data_layout
        self.verbose = verbose
        self.user_annotations = user_annotations if user_annotations is not None else {}
        self.input_default_value = input_default_value
        self.force_transpose_constant_tensor_dim4 = force_transpose_constant_tensor_dim4
        self.info_string = info_string
