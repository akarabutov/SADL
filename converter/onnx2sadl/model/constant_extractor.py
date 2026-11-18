# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from ..errors.conversion_error import ConversionError
from ..types.onnx_dtype import OnnxDType
from ..types.sadl_dtype import SadlDType
from .const_tensor import ConstTensor
from .tensor_layout import TensorLayout


class ConstantExtractor:
    """Reads constant tensors (initializers / Constant nodes) of an OnnxModel."""

    def __init__(self, model, verbose=0):
        self.model = model
        self.verbose = verbose

    def extract(self, name, to_transpose=False):
        """ConstTensor for ``name``, following Identity nodes."""
        if self.verbose:
            print("[INFO] additional data {} transpose={}".format(name, to_transpose))

        init = self.model.initializer(name)
        if init is not None:
            return self.from_tensor(init, to_transpose)
        node = self.model.first_output_producer(name)
        if node is not None:
            if node.op_type == "Identity":
                return self.extract(node.input[0], to_transpose)
            return self.from_tensor(node.attribute[0].t, to_transpose)
        raise ConversionError("[ERROR] unable to extract data in {}".format(name))

    @staticmethod
    def from_tensor(tensor, to_transpose=False):
        """ConstTensor from an onnx TensorProto (INT64 is narrowed to INT32)."""
        dims = [1] if tensor.dims == [] else [dim for dim in tensor.dims]
        raw_data = tensor.raw_data

        if not OnnxDType.has(tensor.data_type):
            raise ValueError("extract_additional_data: Unknown dtype")
        onnx_type = OnnxDType(tensor.data_type)
        dtype = onnx_type.sadl_type
        if onnx_type == OnnxDType.INT64:
            x = np.frombuffer(raw_data, dtype=np.int64)
            if len(x) == 0:  # data not in raw_data: look in int64_data
                x = np.int32(tensor.int64_data)
            raw_data = x.astype(np.int32).tobytes()

        if to_transpose:
            if len(dims) == 4:
                raw_data, dims = TensorLayout.conv_weight_to_sadl(raw_data, dims)
            elif len(dims) == 2:
                raw_data, dims = TensorLayout.transpose_matrix(raw_data, dims)
            elif len(dims) == 1 and dtype == SadlDType.INT32:
                raw_data, dims = TensorLayout.shape_vector_nchw_to_nhwc(raw_data, dims)
            else:
                raise ValueError("extract_additional_data_from_node")
        return ConstTensor(dims, raw_data, dtype)

    @staticmethod
    def attribute_values(node):
        """Values of the "value" tensor attribute of a Constant node, as a list."""
        attr = next((a for a in node.attribute if a.name == "value"), None)
        if attr is None:
            raise ConversionError("[ERROR] no 'value' attribute in {}".format(node.name))
        tensor = attr.t
        if not OnnxDType.has(tensor.data_type):
            raise ValueError("extract_attribute_values: Unknown dtype")
        np_type = OnnxDType(tensor.data_type).numpy_type
        return np.frombuffer(tensor.raw_data, dtype=np_type).tolist()
