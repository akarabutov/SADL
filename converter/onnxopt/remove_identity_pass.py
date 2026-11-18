# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .optimization_pass import OptimizationPass


class RemoveIdentityPass(OptimizationPass):
    """Removes Identity (and Dropout in inference) nodes by rewiring their consumers.

    Nodes producing a graph output are kept so output names do not change.
    """

    name = "remove_identity"
    description = "remove Identity / inference Dropout nodes"

    REMOVABLE = ("Identity", "Dropout")

    def run(self, model):
        graph = model.graph
        graph_outputs = {o.name for o in graph.output}
        renames = {}
        kept = []
        for node in graph.node:
            no_mask = all(not out for out in node.output[1:])  # Dropout mask not used
            if node.op_type in self.REMOVABLE and node.output[0] not in graph_outputs and no_mask:
                renames[node.output[0]] = node.input[0]
            else:
                kept.append(node)
        if not renames:
            return model
        for node in kept:
            for i, name in enumerate(node.input):
                while name in renames:
                    name = renames[name]
                node.input[i] = name
        del graph.node[:]
        graph.node.extend(kept)
        return model
