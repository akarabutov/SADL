# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..model.onnx_model import OnnxModel
from ..types.op_type import OpType


class OpConverter:
    """Base class of all ONNX operator converters.

    A subclass declares the ONNX op types it handles in ``onnx_ops`` and
    implements ``convert``. It may also override ``annotate`` to take part in
    the NCHW/NHWC layout analysis done by GraphAnnotator.
    """

    onnx_ops = ()

    def convert(self, ctx, node, inputs):
        """Add the SADL layers of ``node`` to ``ctx.graph``.

        ``inputs`` holds the names of inputs 0..2 (None when absent), already
        redirected to inserted layout transposes when needed.
        """
        raise NotImplementedError

    def annotate(self, annotator, node, data_layout):
        """Adjust ``annotator.annotations[node.name]``; ``data_layout`` is the inherited layout."""

    # --- helpers for subclasses -------------------------------------------
    @staticmethod
    def attribute(node, name):
        return OnnxModel.get_attribute(node, name)

    @staticmethod
    def emit(ctx, node, op_type, inputs, **attrs):
        """Add the layer producing ``node.output[0]`` and bind it."""
        out = node.output[0]
        layer_attrs = {"data": node}
        layer_attrs.update(attrs)
        ctx.graph.add(out, op_type, inputs, layer_attrs)
        ctx.graph.bind(out, out)
        return out

    @staticmethod
    def add_bias(ctx, node):
        """Append ``BiasAdd(output, Const(input[2]))`` named ``<output>_NOT_IN_GRAPH``."""
        bias = node.input[2]
        ctx.add_const(bias, ctx.extract_const(bias, False), node)
        out = node.output[0]
        bias_add = out + "_NOT_IN_GRAPH"
        ctx.graph.bind(bias_add, None)
        ctx.graph.add(bias_add, OpType.BiasAdd, [out, ctx.graph.resolve(bias)], {"data": node})

    @staticmethod
    def bind_output_before_bias(ctx, node):
        """With a bias the ONNX output is produced by the BiasAdd layer."""
        out = node.output[0]
        ctx.graph.bind(out, out if len(node.input) == 2 else out + "_NOT_IN_GRAPH")

    @staticmethod
    def nchw_axis_to_nhwc(axis):
        """Axis 1/2/3 of an NCHW tensor -> axis of the same dim in NHWC."""
        return {1: 3, 2: 1, 3: 2}.get(axis, axis)
