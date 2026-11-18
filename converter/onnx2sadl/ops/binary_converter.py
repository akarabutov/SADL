# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from .op_converter import OpConverter


class BinaryConverter(OpConverter):
    """Two-input element-wise layer whose inputs may be initializers.

    Constant inputs become Const layers. Subclasses tune the behaviour:
        sadl_op:              SADL layer type
        const_to_transpose:   "none" or "producer" (use the producer annotation)
        reject_two_constants: raise if both inputs are constants
        const_first_swaps:    a constant first input is moved second
        lower_rank_swaps:     a lower-rank first input is moved second
    """

    sadl_op = None
    const_to_transpose = "none"
    reject_two_constants = False
    const_first_swaps = False
    lower_rank_swaps = False

    def convert(self, ctx, node, inputs):
        a, b = inputs[0], inputs[1]
        if self.reject_two_constants and ctx.is_initializer(a) and ctx.is_initializer(b):
            raise ConversionError(
                "[ERROR] unsupported double constants {}\n{}".format(node.op_type, node)
            )
        swap = False
        for index, name in enumerate((a, b)):
            if ctx.is_initializer(name):
                to_transpose = (
                    ctx.producer_to_transpose(name) if self.const_to_transpose == "producer" else False
                )
                ctx.add_const(name, ctx.extract_const(name, to_transpose, allow_force_4d=True), node)
                if index == 0 and self.const_first_swaps:
                    swap = True
        if not swap and self.lower_rank_swaps:
            d1 = ctx.model.declared_dims(a)
            d2 = ctx.model.declared_dims(b)
            if len(d1) and len(d2) and len(d1) < len(d2):
                swap = True

        resolve = ctx.graph.resolve
        layer_inputs = [resolve(b), resolve(a)] if swap else [resolve(a), resolve(b)]
        self.emit(ctx, node, self.sadl_op, layer_inputs, **self.layer_attrs(node))

    def layer_attrs(self, node):
        return {}
