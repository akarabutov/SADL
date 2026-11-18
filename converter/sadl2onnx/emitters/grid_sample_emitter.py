# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class GridSampleEmitter(LayerEmitter):
    """GridSample(x NHWC, grid). SADL reads the grid as [N, W_out, 2, H_out]
    (the ONNX (N,H,W,2) grid seen through SADL's NHWC storage)."""

    sadl_ops = (OpType.GridSample,)

    GRID_TO_NHW2 = [0, 3, 1, 2]
    MODES = ["nearest", "bilinear", "bicubic"]
    PADDING_MODES = ["border", "zeros"]

    def emit(self, ctx, layer):
        x_id, grid_id = layer.inputs
        b = ctx.builder
        x = self.to_nchw(ctx, ctx.value(x_id), layer)
        grid = b.transpose(ctx.value(grid_id), self.GRID_TO_NHW2, layer.name + "_grid")
        y = b.add_node(
            "GridSample",
            [x, grid],
            layer.name + "_gs",
            align_corners=layer["align_corners"],
            mode=self.MODES[layer["mode"]],
            padding_mode=self.PADDING_MODES[layer["padding_mode"]],
        )
        return self.to_nhwc(ctx, y, layer)
