# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from enum import IntEnum


class OpType(IntEnum):
    """Layer identifiers of the SADL binary format (must match sadl/layers.h)."""

    Const = 1
    Placeholder = 2
    Identity = 3
    BiasAdd = 4
    MaxPool = 5
    MatMul = 6
    Reshape = 7
    Relu = 8
    Conv2D = 9
    Add = 10
    ConcatV2 = 11
    Mul = 12
    Maximum = 13
    LeakyReLU = 14
    Transpose = 15
    Flatten = 16
    Shape = 17
    Expand = 18
    Conv2DTranspose = 19
    Slice = 20  # currently slicing across depth is supported with step 1
    PReLU = 21
    # The same SADL layer performs MatMul and batched MatMul: these are aliases.
    BatchMatMul = 6
    ScatterND = 22
    GridSample = 23
    Resize = 24
    Compare = 25
    Where = 26
    Minimum = 27
    AveragePool = 28
    ReduceMean = 29
    Tile = 30
    Pad = 31
    Floor = 32
    Sub = 33
    Abs = 34
    Depth2Space = 35
    Gather = 36
    GatherElements = 37
    GlobalAveragePool = 38
    BitShift = 39
    Clip = 40
    BatchMatMulV2 = 6

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
