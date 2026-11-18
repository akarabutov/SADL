# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from ..types.op_type import OpType
from .op_converter import OpConverter


class ReshapeConverter(OpConverter):
    """Reshape(X, shape). A Reshape feeding a Conv weight keeps the OIHW semantics."""

    onnx_ops = ("Reshape",)

    def convert(self, ctx, node, inputs):
        shape = inputs[1]
        is_kernel_reshape = ctx.model.is_conv_weight(node.output[0])

        producer = ctx.model.producer(shape)
        if producer is not None and (not hasattr(producer, "op_type") or producer.op_type == "Constant"):
            if is_kernel_reshape:
                values = ctx.model.constant_ints(shape)
                if values is None:
                    raise ConversionError("Kernel Reshape shape input must be constant")
                if len(values) == 4 and values[2] == 1 and values[3] == 1:
                    values = [1, 1, values[0], values[1]]  # 1x1 kernel: [O,I,1,1] -> [1,1,O,I]
                ctx.graph.add_int32_vector(shape, values, node)
            else:
                tensor = ctx.extract_const(shape, ctx.annotation(producer.name).to_transpose)
                ctx.graph.add_const(shape, tensor, node)
        ctx.graph.bind(shape, shape)

        self.emit(ctx, node, OpType.Reshape, [ctx.graph.resolve(inputs[0]), shape])

    # --- layout analysis --------------------------------------------------
    def annotate(self, annotator, node, data_layout):
        model = annotator.model
        annotation = annotator.annotations[node.name]
        global_layout = annotator.global_layout
        verbose = annotator.verbose

        if model.is_conv_weight(node.output[0]):
            annotation.layout_onnx = None
            annotation.transpose_before_in0 = None
            annotation.transpose_before_in1 = None
            annotation.to_nhwc_after_out = False
            return

        in0_dims = model.try_shape(node.input[0])
        in0_rank = len(in0_dims) if in0_dims is not None else None
        pre_transpose_allowed = in0_rank == 4 or in0_rank is None

        dims = model.constant_ints(node.input[1])
        if dims is None:  # shape computed by the graph
            if global_layout == "nchw":
                if pre_transpose_allowed:
                    annotation.transpose_before_in0 = "nhwc"
                if verbose > 1:
                    print(" case nchw with shape from a tensor")
            return

        # special case: 2-D -> [N, C, 1, 1]
        if global_layout == "nchw" and len(dims) == 4 and dims[2] == 1 and dims[3] == 1 and in0_rank == 2:
            self._annotate_2d_to_nc11(annotator, node, annotation)
            return

        if len(dims) == 4:
            self._annotate_4d_shape(annotator, node, data_layout, dims, pre_transpose_allowed)
        else:
            annotation.layout_onnx = None
        self._inherit_nchw_parent(annotator, node, pre_transpose_allowed)

    @staticmethod
    def _annotate_2d_to_nc11(annotator, node, annotation):
        """Rewrite the shape vector [N,C,1,1] -> [N,1,1,C]; no transpose around."""
        shape_src = annotator.model.producer(node.input[1])
        if shape_src is not None and hasattr(shape_src, "name") and shape_src.name in annotator.annotations:
            annotator.annotations[shape_src.name].to_transpose = True
        annotation.transpose_before_in0 = None
        annotation.to_nhwc_after_out = False
        annotation.layout_onnx = None

    @staticmethod
    def _annotate_4d_shape(annotator, node, data_layout, dims, pre_transpose_allowed):
        annotation = annotator.annotations[node.name]
        global_layout = annotator.global_layout
        verbose = annotator.verbose
        if data_layout == "nhwc":
            if dims[1] == 1:  # nhwc -> nchw
                annotation.to_remove = True
                annotation.layout_onnx = "nchw"
            elif global_layout == "nchw":
                if pre_transpose_allowed:
                    annotation.transpose_before_in0 = "nchw"
                annotation.to_nhwc_after_out = True
                annotation.layout_onnx = None
            else:
                if verbose > 1:
                    print("[WARNING] reshape unknown for NODE:\n", node, "\n shape: dims=", dims)
                annotation.layout_onnx = None
        elif data_layout == "nchw":
            annotation.to_nhwc_after_out = True
            if pre_transpose_allowed:
                annotation.transpose_before_in0 = "nchw"
            annotation.layout_onnx = None
            if verbose > 1:
                print(" case nchw with cst shape")
        elif data_layout is None:
            annotation.layout_onnx = global_layout
            if global_layout == "nchw":
                annotation.to_nhwc_after_out = True

    @staticmethod
    def _inherit_nchw_parent(annotator, node, pre_transpose_allowed):
        """An NCHW producer forces a transpose back to NCHW before reshaping."""
        annotation = annotator.annotations[node.name]
        parent = annotator.model.producer_skip_identity(node.input[0])
        if parent is None or not pre_transpose_allowed or not hasattr(parent, "name"):
            return
        parent_annotation = annotator.annotations.get(parent.name)
        if parent_annotation is not None and parent_annotation.layout_onnx == "nchw":
            annotation.transpose_before_in0 = "nchw"
            if annotation.to_nhwc_after_out is False:
                annotation.layout_onnx = "nchw"
