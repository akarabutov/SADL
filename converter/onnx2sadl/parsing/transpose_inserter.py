# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from ..model.tensor_layout import TensorLayout


class TransposeInserter:
    """Inserts the layout transposes requested by the node annotations."""

    def __init__(self, ctx):
        self.ctx = ctx

    def before_input(self, node, target_layout, index):
        """Transpose input ``index`` of ``node`` to ``target_layout``; returns the name to use."""
        ctx = self.ctx
        in_name = node.input[index]
        if ctx.model.is_initializer(in_name):  # constant: transpose the data instead
            if in_name in ctx.graph:
                raise ConversionError(
                    f"[ERROR] input already processed {in_name} in processing node {node.name}"
                )
            ctx.annotation(in_name).to_transpose = True
            return in_name

        if not ctx.model.should_insert_4d_transpose(in_name):
            return in_name

        perm = TensorLayout.NHWC_TO_NCHW if target_layout == "nchw" else TensorLayout.NCHW_TO_NHWC
        return ctx.graph.add_transpose(
            in_name,
            perm,
            coef_name=in_name + "_COEF_TRANSPOSE_NOT_IN_GRAPH",
            out_name=in_name + "_TRANSPOSE_NOT_IN_GRAPH",
            owner=node,
        )

    def after_output(self, node):
        """Append an NCHW -> NHWC transpose after the first output of ``node``."""
        ctx = self.ctx
        out = node.output[0]
        if not ctx.model.should_insert_4d_transpose(out):
            return out
        name = ctx.graph.add_transpose(
            out,
            TensorLayout.NCHW_TO_NHWC,
            coef_name=out + "_COEF_TRANSPOSE_AFTER_NOT_IN_GRAPH",
            out_name=out + "_TRANSPOSE_AFTER_NOT_IN_GRAPH",
            owner=node,
        )
        ctx.graph.bind(out, name)
        return name
