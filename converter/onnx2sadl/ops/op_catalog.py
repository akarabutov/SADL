# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .abs_converter import AbsConverter
from .add_converter import AddConverter
from .average_pool_converter import AveragePoolConverter
from .bit_shift_converter import BitShiftConverter
from .clip_converter import ClipConverter
from .concat_converter import ConcatConverter
from .constant_converter import ConstantConverter
from .conv_converter import ConvConverter
from .conv_transpose_converter import ConvTransposeConverter
from .depth_to_space_converter import DepthToSpaceConverter
from .equal_converter import EqualConverter
from .expand_converter import ExpandConverter
from .flatten_converter import FlattenConverter
from .floor_converter import FloorConverter
from .gather_converter import GatherConverter
from .gather_elements_converter import GatherElementsConverter
from .gemm_converter import GemmConverter
from .global_average_pool_converter import GlobalAveragePoolConverter
from .greater_converter import GreaterConverter
from .grid_sample_converter import GridSampleConverter
from .identity_converter import IdentityConverter
from .leaky_relu_converter import LeakyReluConverter
from .less_converter import LessConverter
from .mat_mul_converter import MatMulConverter
from .max_converter import MaxConverter
from .max_pool_converter import MaxPoolConverter
from .min_converter import MinConverter
from .mul_converter import MulConverter
from .op_registry import OpRegistry
from .pad_converter import PadConverter
from .prelu_converter import PReluConverter
from .reduce_mean_converter import ReduceMeanConverter
from .relu_converter import ReluConverter
from .reshape_converter import ReshapeConverter
from .resize_converter import ResizeConverter
from .scatter_nd_converter import ScatterNDConverter
from .shape_converter import ShapeConverter
from .slice_converter import SliceConverter
from .sub_converter import SubConverter
from .tile_converter import TileConverter
from .transpose_converter import TransposeConverter
from .unsqueeze_converter import UnsqueezeConverter
from .where_converter import WhereConverter


class OpCatalog:
    """The list of supported ONNX operators. Add a new converter class here."""

    CONVERTERS = (
        AbsConverter,
        AddConverter,
        AveragePoolConverter,
        BitShiftConverter,
        ClipConverter,
        ConcatConverter,
        ConstantConverter,
        ConvConverter,
        ConvTransposeConverter,
        DepthToSpaceConverter,
        EqualConverter,
        ExpandConverter,
        FlattenConverter,
        FloorConverter,
        GatherConverter,
        GatherElementsConverter,
        GemmConverter,
        GlobalAveragePoolConverter,
        GreaterConverter,
        GridSampleConverter,
        IdentityConverter,
        LeakyReluConverter,
        LessConverter,
        MatMulConverter,
        MaxConverter,
        MaxPoolConverter,
        MinConverter,
        MulConverter,
        PadConverter,
        PReluConverter,
        ReduceMeanConverter,
        ReluConverter,
        ReshapeConverter,
        ResizeConverter,
        ScatterNDConverter,
        ShapeConverter,
        SliceConverter,
        SubConverter,
        TileConverter,
        TransposeConverter,
        UnsqueezeConverter,
        WhereConverter,
    )

    @classmethod
    def build_registry(cls):
        return OpRegistry(converter() for converter in cls.CONVERTERS)
