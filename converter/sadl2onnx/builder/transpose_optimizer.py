# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx import helper


class TransposeOptimizer:
    """Removes ``Transpose(Transpose(x, p1), p2)`` pairs that compose to the identity.

    Layer emitters wrap NCHW-only ONNX ops (Conv, pooling...) in NHWC->NCHW and
    NCHW->NHWC transposes; between two such ops the transposes cancel out.
    Nodes left without consumers are then removed.
    """

    def __init__(self, nodes, graph_outputs):
        self.nodes = list(nodes)
        self.graph_outputs = set(graph_outputs)

    def run(self):
        changed = True
        while changed:
            changed = self._bypass_identity_pairs()
            changed = self._remove_dead_nodes() or changed
        return self.nodes

    @staticmethod
    def _perm(node):
        for a in node.attribute:
            if a.name == "perm":
                return list(helper.get_attribute_value(a))
        return None

    def _bypass_identity_pairs(self):
        producers = {out: node for node in self.nodes for out in node.output}
        renames = {}
        for node in self.nodes:
            if node.op_type != "Transpose" or node.output[0] in self.graph_outputs:
                continue
            first = producers.get(node.input[0])
            if first is None or first.op_type != "Transpose":
                continue
            p1, p2 = self._perm(first), self._perm(node)
            if p1 is None or p2 is None or len(p1) != len(p2):
                continue
            if [p1[p2[k]] for k in range(len(p2))] == list(range(len(p2))):
                renames[node.output[0]] = first.input[0]
        if not renames:
            return False
        for node in self.nodes:
            for i, name in enumerate(node.input):
                while name in renames:
                    name = renames[name]
                node.input[i] = name
        return True

    def _remove_dead_nodes(self):
        used = set(self.graph_outputs)
        for node in self.nodes:
            used.update(node.input)
        kept = [n for n in self.nodes if any(out in used for out in n.output)]
        removed = len(kept) != len(self.nodes)
        self.nodes = kept
        return removed
