# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from ..types.op_type import OpType
from .op_converter import OpConverter


class ConvConverter(OpConverter):
    """Conv -> Conv2D (+ BiasAdd). The weight may be static or computed (dynamic conv)."""

    onnx_ops = ("Conv",)

    # dynamic kernel OIKhKw -> KhKwOI
    DYNAMIC_KERNEL_PERM = [2, 3, 0, 1]
    DYNAMIC_KERNEL_SUFFIX = "_KERNEL_LAYOUT_NOT_IN_GRAPH"

    def convert(self, ctx, node, inputs):
        if len(node.input) not in (2, 3):
            raise ConversionError("parse_graph_node: Error on Conv node type")

        weight = self._weight_layer(ctx, node, inputs[1])
        pads = self.attribute(node, "pads")
        attrs = {
            "data": node,
            "strides": self.attribute(node, "strides").ints,
            "group": self.attribute(node, "group").i,
            "pads": pads.ints if pads else [0, 0, 0, 0],
        }
        self.bind_output_before_bias(ctx, node)
        ctx.graph.add(node.output[0], OpType.Conv2D, [ctx.graph.resolve(inputs[0]), weight], attrs)
        if len(node.input) == 3:
            self.add_bias(ctx, node)

    def _weight_layer(self, ctx, node, name):
        """Name of the SADL layer holding the kernel in SADL layout."""
        if ctx.model.is_static_tensor(name):
            tensor = ctx.extract_const(name, ctx.producer_to_transpose(name))
            return ctx.add_const(name, tensor, node)

        kernel_dims = ctx.model.conv_weight_dims(name)
        if kernel_dims is not None and len(kernel_dims) == 4 and kernel_dims[2] == 1 and kernel_dims[3] == 1:
            # 1x1: the producing Reshape already outputs [1, 1, O, I]
            return ctx.graph.resolve(name)

        out_name = name + self.DYNAMIC_KERNEL_SUFFIX
        return ctx.graph.add_transpose(
            name, self.DYNAMIC_KERNEL_PERM, coef_name=out_name + "_COEF", out_name=out_name, owner=node
        )

    def annotate(self, annotator, node, data_layout):
        if annotator.model.is_static_tensor(node.input[1]):
            weight = annotator.model.producer(node.input[1])
            if weight is not None and hasattr(weight, "name") and weight.name in annotator.annotations:
                annotator.annotations[weight.name].to_transpose = True
                annotator.annotations[weight.name].layout_onnx = "nhwc"
