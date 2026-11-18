# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np
from onnx import helper, numpy_helper

from .optimization_pass import OptimizationPass


class SplitSlicesPass(OptimizationPass):
    """Splits a Slice over several axes into a chain of one-axis Slices.

    onnxsim fuses consecutive Slices, but SADL slices one axis at a time
    (onnx2sadl requires ``len(axes) == 1``). Only Slices whose starts, ends,
    axes and steps are initializers are rewritten; axes that select the whole
    dimension (start 0, end >= INT32_MAX, step 1) are dropped.

    Models from ``skip_producers`` are left untouched: onnx2sadl converts a
    tf2onnx (NHWC) Slice over all 4 dims in one layer and needs it whole.
    """

    name = "split_slices"
    description = "one Slice per axis (SADL slices one axis at a time; not for tf2onnx models)"

    INT32_MAX = 2147483647

    def __init__(self, skip_producers=("tf2onnx",)):
        self.skip_producers = tuple(skip_producers)

    def run(self, model):
        if model.producer_name in self.skip_producers:
            return model
        graph = model.graph
        inits = {t.name: t for t in graph.initializer}
        new_nodes = []
        for node in graph.node:
            pieces = self._split(node, inits, graph) if node.op_type == "Slice" else None
            new_nodes.extend(pieces if pieces else [node])
        del graph.node[:]
        graph.node.extend(new_nodes)
        return model

    def _split(self, node, inits, graph):
        if len(node.input) < 4 or any(name not in inits for name in node.input[1:4]):
            return None
        starts, ends, axes = (numpy_helper.to_array(inits[n]).astype(np.int64).tolist() for n in node.input[1:4])
        if len(node.input) > 4 and node.input[4]:
            if node.input[4] not in inits:
                return None
            steps = numpy_helper.to_array(inits[node.input[4]]).astype(np.int64).tolist()
        else:
            steps = [1] * len(axes)
        if len(axes) <= 1:
            return None

        kept = [k for k in range(len(axes)) if not (starts[k] == 0 and ends[k] >= self.INT32_MAX and steps[k] == 1)]
        if not kept:
            kept = [0]
        pieces = []
        current = node.input[0]
        for i, k in enumerate(kept):
            last = i == len(kept) - 1
            out = node.output[0] if last else "{}_axis{}".format(node.output[0], axes[k])
            prefix = "{}_axis{}".format(node.name or node.output[0], axes[k])
            names = []
            for suffix, value in (("starts", starts[k]), ("ends", ends[k]), ("axes", axes[k]), ("steps", steps[k])):
                name = "{}_{}".format(prefix, suffix)
                graph.initializer.append(numpy_helper.from_array(np.array([value], np.int64), name))
                inits[name] = graph.initializer[-1]
                names.append(name)
            pieces.append(helper.make_node("Slice", [current] + names, [out], name=prefix))
            current = out
        return pieces
