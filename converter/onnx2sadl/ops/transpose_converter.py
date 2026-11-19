# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .op_converter import OpConverter


class TransposeConverter(OpConverter):
    """Explicit Transpose kept in the graph (layout transposes are removed by annotation)."""

    onnx_ops = ("Transpose",)

    NHWC_TO_NCHW = [0, 3, 1, 2]
    NCHW_TO_NHWC = [0, 2, 3, 1]

    def convert(self, ctx, node, inputs):
        coef = node.output[0] + "_COEF_TRANSPOSE"
        ctx.graph.add_int32_vector(coef, list(self.attribute(node, "perm").ints), node)
        ctx.graph.bind(coef, coef)
        self.emit(ctx, node, OpType.Transpose, [ctx.graph.resolve(inputs[0]), coef])

    def annotate(self, annotator, node, data_layout):
        annotation = annotator.annotations[node.name]
        perm = list(self.attribute(node, "perm").ints)
        if data_layout == "nhwc":
            if perm[:4] == self.NHWC_TO_NCHW:
                annotation.to_remove = True
                annotation.layout_onnx = "nchw"
            elif annotator.verbose > 1:
                print("[WARNING] transpose not for NCHW handling in\n", node)
        elif data_layout == "nchw":
            if perm[:4] == self.NCHW_TO_NHWC:
                annotation.to_remove = True
                annotation.layout_onnx = "nhwc"
            elif len(perm) == 4 and annotator.global_layout == "nchw":
                # A user transpose (e.g. swapping H and W before a MatMul) on data
                # stored NHWC in SADL: apply it in NCHW, then go back to NHWC.
                annotation.transpose_before_in0 = "nchw"
                annotation.to_nhwc_after_out = True
            elif annotator.verbose > 1:
                print("[WARNING] transpose not for NCHW handling in\n", node)
        elif annotator.global_layout == "nchw":
            if len(perm) == 4:  # assume a user transpose
                annotation.transpose_before_in0 = "nchw"
                annotation.to_nhwc_after_out = True
                annotation.layout_onnx = "nhwc"

        if annotation.to_remove:
            # GridSample consumes the optical flow transposed (N,2,H,W) -> (N,H,W,2):
            # that transpose must be kept.
            for consumer in annotator.model.consumers(node.output[0]):
                if consumer.op_type == "GridSample":
                    annotation.to_remove = False
                    annotation.layout_onnx = "nchw"
                    break
