# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .const_params_reader import ConstParamsReader
from .layer_params_reader import LayerParamsReader
from .list_params_reader import ListParamsReader
from .placeholder_params_reader import PlaceholderParamsReader


class LayerParamsReaderRegistry:
    """OpType -> LayerParamsReader (mirror of onnx2sadl LayerSerializerRegistry)."""

    def __init__(self):
        pool = ListParamsReader("strides[]", "kernel_shape[]", "pads[]")
        self._default = LayerParamsReader()
        self._by_op = {
            OpType.Const: ConstParamsReader(),
            OpType.Placeholder: PlaceholderParamsReader(),
            OpType.Slice: ListParamsReader("start_h", "end_h", "start_w", "end_w", "start_c", "end_c"),
            OpType.Conv2D: ListParamsReader("strides[]", "pads[]", "group", reads_quantizer=True),
            OpType.Conv2DTranspose: ListParamsReader(
                "strides[]", "pads[]", "output_padding[]", reads_quantizer=True
            ),
            OpType.MatMul: ListParamsReader(reads_quantizer=True),
            OpType.Mul: ListParamsReader(reads_quantizer=True),
            OpType.MaxPool: pool,
            OpType.AveragePool: pool,
            OpType.Flatten: ListParamsReader("axis"),
            OpType.GridSample: ListParamsReader("align_corners", "mode", "padding_mode"),
            OpType.Resize: ListParamsReader("input_label", "coordinate_transformation_mode", "mode", "nearest_mode"),
            OpType.Compare: ListParamsReader("mode"),
            OpType.ReduceMean: ListParamsReader("axes[]", "keepdims?"),
            OpType.Pad: ListParamsReader("mode", "pads[4]"),
            OpType.Depth2Space: ListParamsReader("blocksize"),
            OpType.BitShift: ListParamsReader("direction"),
            OpType.Gather: ListParamsReader("axis", "indices_is_scalar?"),
            OpType.GatherElements: ListParamsReader("axis"),
        }

    def get(self, op_type):
        return self._by_op.get(op_type, self._default)
