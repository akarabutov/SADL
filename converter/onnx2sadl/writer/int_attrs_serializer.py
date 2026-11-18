# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_serializer import LayerSerializer


class IntAttrsSerializer(LayerSerializer):
    """A fixed sequence of int32 attributes, written in the given order."""

    def __init__(self, *attr_names):
        self.attr_names = attr_names

    def write(self, stream, node):
        for name in self.attr_names:
            stream.i32(node[name], name)
