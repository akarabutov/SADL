# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .bit_shift_emitter import BitShiftEmitter
from .clip_emitter import ClipEmitter
from .compare_emitter import CompareEmitter
from .concat_emitter import ConcatEmitter
from .const_emitter import ConstEmitter
from .conv2d_emitter import Conv2DEmitter
from .conv2d_transpose_emitter import Conv2DTransposeEmitter
from .depth2space_emitter import Depth2SpaceEmitter
from .flatten_emitter import FlattenEmitter
from .gather_elements_emitter import GatherElementsEmitter
from .gather_emitter import GatherEmitter
from .grid_sample_emitter import GridSampleEmitter
from .leaky_relu_emitter import LeakyReluEmitter
from .mat_mul_emitter import MatMulEmitter
from .nchw_op_emitter import NchwOpEmitter
from .pad_emitter import PadEmitter
from .placeholder_emitter import PlaceholderEmitter
from .pool_emitter import PoolEmitter
from .prelu_emitter import PReluEmitter
from .reduce_mean_emitter import ReduceMeanEmitter
from .reshape_emitter import ReshapeEmitter
from .resize_emitter import ResizeEmitter
from .scatter_nd_emitter import ScatterNDEmitter
from .shape_emitter import ShapeEmitter
from .simple_op_emitter import SimpleOpEmitter
from .slice_emitter import SliceEmitter
from .tile_emitter import TileEmitter
from .transpose_emitter import TransposeEmitter
from .where_emitter import WhereEmitter


class EmitterRegistry:
    """SADL OpType -> LayerEmitter. Add a new emitter here."""

    def __init__(self):
        emitters = [
            PlaceholderEmitter(),
            ConstEmitter(),
            SimpleOpEmitter(OpType.Identity, "Identity"),
            SimpleOpEmitter(OpType.Relu, "Relu"),
            SimpleOpEmitter(OpType.Floor, "Floor"),
            SimpleOpEmitter(OpType.Abs, "Abs"),
            SimpleOpEmitter(OpType.BiasAdd, "Add"),
            SimpleOpEmitter(OpType.Add, "Add"),
            SimpleOpEmitter(OpType.Sub, "Sub"),
            SimpleOpEmitter(OpType.Mul, "Mul"),
            SimpleOpEmitter(OpType.Maximum, "Max"),
            SimpleOpEmitter(OpType.Minimum, "Min"),
            Conv2DEmitter(),
            Conv2DTransposeEmitter(),
            MatMulEmitter(),
            PoolEmitter(OpType.MaxPool, "MaxPool"),
            PoolEmitter(OpType.AveragePool, "AveragePool"),
            NchwOpEmitter(OpType.GlobalAveragePool, "GlobalAveragePool"),
            Depth2SpaceEmitter(),
            ReshapeEmitter.reshape(),
            ReshapeEmitter.expand(),
            TransposeEmitter(),
            FlattenEmitter(),
            ConcatEmitter(),
            ShapeEmitter(),
            SliceEmitter(),
            LeakyReluEmitter(),
            PReluEmitter(),
            ScatterNDEmitter(),
            GridSampleEmitter(),
            ResizeEmitter(),
            CompareEmitter(),
            WhereEmitter(),
            ReduceMeanEmitter(),
            TileEmitter(),
            PadEmitter(),
            GatherEmitter(),
            GatherElementsEmitter(),
            BitShiftEmitter(),
            ClipEmitter(),
        ]
        self._by_op = {}
        for emitter in emitters:
            for op in emitter.sadl_ops:
                self._by_op[op] = emitter

    def get(self, op_type):
        return self._by_op.get(op_type)
