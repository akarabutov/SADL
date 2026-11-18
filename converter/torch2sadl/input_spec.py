# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import torch

from onnx2sadl.errors.conversion_error import ConversionError


class InputSpec:
    """Shape (and optional name) of one model input, used to build example tensors."""

    def __init__(self, shape, name=None):
        self.shape = [int(d) for d in shape]
        self.name = name

    @classmethod
    def parse(cls, text):
        """``"1,3,16,16"`` or ``"name:1,3,16,16"``."""
        name, _, dims = text.rpartition(":")
        try:
            shape = [int(d) for d in dims.replace("x", ",").split(",") if d.strip()]
        except ValueError:
            raise ConversionError("[ERROR] invalid input shape {!r} (expected e.g. 1,3,16,16)".format(text))
        if not shape:
            raise ConversionError("[ERROR] empty input shape {!r}".format(text))
        return cls(shape, name or None)

    def example_tensor(self, generator=None):
        return torch.rand(self.shape, generator=generator)

    @classmethod
    def to_example_inputs(cls, inputs):
        """Normalize ``inputs`` to a tuple of tensors.

        Accepts a tensor, a tuple/list of tensors, an InputSpec, a shape
        (tuple of ints) or a list mixing these.
        """
        if isinstance(inputs, (torch.Tensor, InputSpec)) or cls._is_shape(inputs):
            inputs = [inputs]
        result = []
        for item in inputs:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, InputSpec):
                result.append(item.example_tensor())
            elif cls._is_shape(item):
                result.append(cls(item).example_tensor())
            else:
                raise ConversionError("[ERROR] unsupported example input {!r}".format(type(item)))
        return tuple(result)

    @staticmethod
    def _is_shape(value):
        return isinstance(value, (tuple, list, torch.Size)) and len(value) > 0 and all(
            isinstance(d, int) for d in value
        )

    def __repr__(self):
        return "InputSpec({}, name={!r})".format(self.shape, self.name)
