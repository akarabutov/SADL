# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.errors.conversion_error import ConversionError


class SadlModel:
    """Content of a SADL file: header and layers in file order."""

    def __init__(self, version, model_type, info, layers, input_ids, output_ids):
        self.version = version
        self.model_type = model_type
        self.info = info
        self.layers = layers
        self.input_ids = input_ids
        self.output_ids = output_ids
        self._by_id = {layer.id: layer for layer in layers}

    def layer(self, layer_id):
        if layer_id not in self._by_id:
            raise ConversionError("[ERROR] unknown layer id {}".format(layer_id))
        return self._by_id[layer_id]

    def consumers(self, layer_id):
        return [layer for layer in self.layers if layer_id in layer.inputs]
