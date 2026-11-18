# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.errors.conversion_error import ConversionError
from onnx2sadl.types.op_type import OpType
from onnx2sadl.types.sadl_dtype import SadlDType

from .binary_reader import BinaryReader
from .layer_params_reader_registry import LayerParamsReaderRegistry
from .sadl_layer import SadlLayer
from .sadl_model import SadlModel


class SadlReader:
    """Parses a SADL0005 file (format documented in onnx2sadl.writer.SadlWriter)."""

    SUPPORTED_MAGIC = b"SADL0005"

    def __init__(self, params_readers=None, verbose=0):
        self.params_readers = params_readers or LayerParamsReaderRegistry()
        self.verbose = verbose

    def read(self, path):
        with open(path, "rb") as f:
            return self.parse(f.read())

    def parse(self, data):
        reader = BinaryReader(data)
        magic = reader.bytes(8)
        if magic != self.SUPPORTED_MAGIC:
            raise ConversionError("[ERROR] unsupported SADL version {!r} (expected SADL0005)".format(magic))
        model_type = SadlDType(reader.i32())
        info = reader.bytes(reader.i16()).decode("utf-8")
        nb_layers = reader.i32()
        input_ids = reader.i32_list()
        output_ids = reader.i32_list()
        if reader.u8() != 0:
            raise ConversionError("[ERROR] models with precomputed sparsity are not supported")

        layers = [self._read_layer(reader) for _ in range(nb_layers)]
        if not reader.at_end():
            raise ConversionError("[ERROR] trailing bytes after the last layer")
        return SadlModel(magic.decode(), model_type, info, layers, input_ids, output_ids)

    def _read_layer(self, reader):
        layer_id = reader.i32()
        op_value = reader.i32()
        try:
            op_type = OpType(op_value)
        except ValueError:
            raise ConversionError("[ERROR] unknown SADL op {} for layer {}".format(op_value, layer_id))
        name = reader.bytes(reader.i32()).decode("utf-8")
        inputs = reader.i32_list()
        params_reader = self.params_readers.get(op_type)
        params = params_reader.read(reader)
        if params_reader.reads_quantizer:
            params["quantizer"] = reader.i32()
        layer = SadlLayer(layer_id, op_type, name, inputs, params)
        if self.verbose > 1:
            print("[INFO] read", layer)
        return layer
