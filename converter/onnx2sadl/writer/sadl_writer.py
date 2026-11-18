# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..types.sadl_dtype import SadlDType
from .binary_stream import BinaryStream
from .layer_serializer_registry import LayerSerializerRegistry


class SadlWriter:
    """Writes a SadlGraph in the SADL0005 binary format.

    File format:
        MAGIC "SADL0005"                       char[8]
        type_model (0:int32 1:float 2:int16)   int32
        info length / info string (no \\0)     int16 / char[]
        nb_layers                              int32
        nb_inputs, input ids                   int32, int32[]
        nb_outputs, output ids                 int32, int32[]
        sparsity precomputed (0)               char
        for each layer:
            id, op_type, name_size, name, nb_inputs, input ids
            layer-specific parameters (see LayerSerializer subclasses)
    """

    MAGIC = b"SADL0005"
    MAX_INFO_LENGTH = 32767

    def __init__(self, verbose=0, serializers=None):
        self.verbose = verbose
        self.serializers = serializers or LayerSerializerRegistry()

    def write(self, path, graph, inputs, outputs, info_string=""):
        layer_ids = {name: idx for idx, (name, _node) in enumerate(graph.items())}
        info_bytes = str(info_string).encode("utf-8")
        if len(info_bytes) > self.MAX_INFO_LENGTH:
            raise ValueError("info_string is too long for int16 length field")

        with open(path, "wb") as f:
            stream = BinaryStream(f, self.verbose)
            stream.raw(self.MAGIC)
            stream.i32(SadlDType.FLOAT)  # network output type
            stream.i16(len(info_bytes))
            stream.raw(info_bytes)

            stream.log(f"# Nb layers: {len(graph)}")
            stream.i32(len(graph))
            self._write_ids(stream, "input", [layer_ids[n] for n in inputs])
            self._write_ids(stream, "output", [layer_ids[n] for n in outputs])
            stream.raw(b"\x00")  # no precomputed sparsity

            for name, node in graph.items():
                self._write_layer(stream, layer_ids, name, node)

    @staticmethod
    def _write_ids(stream, kind, ids):
        stream.log(f"# Nb {kind}s: {len(ids)}")
        stream.i32(len(ids))
        for i in ids:
            stream.log(f"#  {kind} {i}")
            stream.i32(i)

    def _write_layer(self, stream, layer_ids, name, node):
        stream.log(f"# Layer id {layer_ids[name]}")
        stream.i32(layer_ids[name])
        stream.i32(node.op_type.value, "op")
        stream.i32(len(name), "name_size")
        stream.log(f"#\t name {name}")
        stream.raw(str.encode(str(name)))
        stream.i32(len(node.inputs), "nb_inputs")
        for input_name in node.inputs:
            stream.log(f"#\t\t {layer_ids[input_name]} ({input_name})")
            stream.i32(layer_ids[input_name])

        serializer = self.serializers.get(node.op_type)
        serializer.write(stream, node)
        if serializer.writes_quantizer:
            stream.i32(0)  # internal quantizer, default 0
        stream.log("")
