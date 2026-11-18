# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .nchw_op_emitter import NchwOpEmitter


class Depth2SpaceEmitter(NchwOpEmitter):
    """SADL takes channel c*b*b + bh*b + bw: this is the ONNX "CRD" mode."""

    def __init__(self):
        super().__init__(OpType.Depth2Space, "DepthToSpace", mode="CRD")

    def attributes(self, layer):
        attrs = super().attributes(layer)
        attrs["blocksize"] = layer["blocksize"]
        return attrs
