# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.op_type import OpType
from .const_serializer import ConstSerializer
from .conv2d_serializer import Conv2DSerializer
from .conv2d_transpose_serializer import Conv2DTransposeSerializer
from .gather_serializer import GatherSerializer
from .int_attrs_serializer import IntAttrsSerializer
from .layer_serializer import LayerSerializer
from .pad_serializer import PadSerializer
from .placeholder_serializer import PlaceholderSerializer
from .pool_serializer import PoolSerializer
from .quantized_layer_serializer import QuantizedLayerSerializer
from .reduce_mean_serializer import ReduceMeanSerializer
from .slice_serializer import SliceSerializer


class LayerSerializerRegistry:
    """OpType -> LayerSerializer. Layers not listed have no specific parameters."""

    def __init__(self):
        self._default = LayerSerializer()
        self._by_op = {
            OpType.Const: ConstSerializer(),
            OpType.Placeholder: PlaceholderSerializer(),
            OpType.Slice: SliceSerializer(),
            OpType.Conv2D: Conv2DSerializer(),
            OpType.Conv2DTranspose: Conv2DTransposeSerializer(),
            OpType.MatMul: QuantizedLayerSerializer(),
            OpType.Mul: QuantizedLayerSerializer(),
            OpType.MaxPool: PoolSerializer(),
            OpType.AveragePool: PoolSerializer(),
            OpType.Flatten: IntAttrsSerializer("axis"),
            OpType.GridSample: IntAttrsSerializer("align_corners", "mode", "padding_mode"),
            OpType.Resize: IntAttrsSerializer(
                "input_label", "coordinate_transformation_mode", "mode", "nearest_mode"
            ),
            OpType.Compare: IntAttrsSerializer("mode"),
            OpType.ReduceMean: ReduceMeanSerializer(),
            OpType.Pad: PadSerializer(),
            OpType.Depth2Space: IntAttrsSerializer("blocksize"),
            OpType.BitShift: IntAttrsSerializer("direction"),
            OpType.Gather: GatherSerializer(),
            OpType.GatherElements: IntAttrsSerializer("axis"),
        }

    def get(self, op_type):
        return self._by_op.get(op_type, self._default)
