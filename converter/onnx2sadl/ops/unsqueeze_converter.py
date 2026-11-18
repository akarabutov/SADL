# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .op_converter import OpConverter


class UnsqueezeConverter(OpConverter):
    """SADL handles the extra dimension itself: Unsqueeze is removed."""

    onnx_ops = ("Unsqueeze",)

    def convert(self, ctx, node, inputs):
        ctx.graph.bind(node.output[0], node.output[0])

    def annotate(self, annotator, node, data_layout):
        annotator.annotations[node.name].to_remove = True
