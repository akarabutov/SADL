# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..annotation.silly_reshape_detector import SillyReshapeDetector
from ..types.op_type import OpType
from .op_converter import OpConverter


class ConcatConverter(OpConverter):
    """Concat -> ConcatV2(inputs..., Const(axis))."""

    onnx_ops = ("Concat",)

    # NCHW axis -> NHWC axis (positive and negative forms)
    AXIS_NCHW_TO_NHWC = {1: 3, 2: 1, 3: 2, -3: -1, -2: -3, -1: -2}

    def convert(self, ctx, node, inputs):
        out = node.output[0]
        concat = out + "_NOT_IN_GRAPH"
        # the axis constant takes the ONNX output name, the concat a helper name
        ctx.graph.add_int32_vector(out, [node.attribute[0].i], node)
        ctx.graph.bind(concat, None)
        layer_inputs = [ctx.graph.resolve(i) for i in node.input] + [out]
        ctx.graph.add(concat, OpType.ConcatV2, layer_inputs, {"data": node})
        ctx.graph.bind(out, concat)

    def annotate(self, annotator, node, data_layout):
        if data_layout == "nchw":
            a = self.attribute(node, "axis")
            dims = annotator.model.try_shape(node.input[0])
            if dims is None:
                dims = annotator.model.try_shape(node.output[0])
            if dims is None or len(dims) == 4:
                a.i = self.AXIS_NCHW_TO_NHWC.get(a.i, a.i)

        detector = SillyReshapeDetector(annotator.model, annotator.annotations, annotator.verbose)
        if detector.detect_and_simplify(node):
            print("[WARNING] strange ONNX pattern detected and simplified")
