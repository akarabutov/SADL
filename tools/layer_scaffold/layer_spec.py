# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import json
import re

from .param import Param


class LayerSpec:
    """Every decision about the new layer (CONTRIBUTE.md section 0).

    Saved as JSON so that a generation can be replayed with ``--config``.
    """

    CONVERTER_BASES = ("unary", "binary", "op")
    LAYOUTS = ("none", "axis", "transpose")
    EMITTERS = ("simple", "nchw", "custom")
    CUDA_MODES = ("skip", "cpu", "gpu")
    MAX_INPUTS = 3  # the ONNX -> SADL parser passes inputs 0..2 to the converters

    _CAMEL = re.compile(r"^[A-Z][A-Za-z0-9]*$")
    _SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(self, name, sadl_id, py_name=None, cpp_header=None, onnx_ops=None,
                 nb_inputs=1, const_inputs=(), params=(), internal_quantizer=False,
                 converter_base=None, layout="none", emitter=None, emitter_onnx_op=None,
                 cuda="skip", netron=True, test_model=True):
        self.name = name
        self.sadl_id = int(sadl_id)
        self.py_name = py_name or self.snake_case(name)
        self.cpp_header = cpp_header or f"layer_{name.lower()}.h"
        self.onnx_ops = list(onnx_ops or [name])
        self.nb_inputs = int(nb_inputs)
        self.const_inputs = sorted(int(i) for i in const_inputs)
        self.params = [p if isinstance(p, Param) else Param.parse(p) for p in params]
        self.internal_quantizer = bool(internal_quantizer)
        self.converter_base = converter_base or self.default_converter_base()
        self.layout = layout
        self.emitter = emitter or ("custom" if self.params else "simple")
        self.emitter_onnx_op = emitter_onnx_op or self.onnx_ops[0]
        self.cuda = cuda
        self.netron = bool(netron)
        self.test_model = bool(test_model)

    # --- derived names --------------------------------------------------------
    @staticmethod
    def snake_case(name):
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        return s.lower()

    @property
    def marker(self):
        """Grep-able marker of what is left to do by hand."""
        return f"TODO(new_layer {self.name})"

    @property
    def has_payload(self):
        """True if loadInternal reads bytes (parameters or internal quantizer)."""
        return bool(self.params) or self.internal_quantizer

    @property
    def serializer_kind(self):
        """none | int_attrs | quantized | custom (see CONTRIBUTE.md 4.5)."""
        if not self.has_payload:
            return "none"
        if not self.params:
            return "quantized"
        if not self.internal_quantizer and all(p.kind == "int" for p in self.params):
            return "int_attrs"
        return "custom"

    @property
    def needs_emitter_file(self):
        return self.emitter == "custom" or (self.emitter == "nchw" and bool(self.params))

    def default_converter_base(self):
        if self.nb_inputs == 1 and not self.const_inputs:
            return "unary"
        if self.nb_inputs == 2:
            return "binary"
        return "op"

    # --- files ----------------------------------------------------------------
    @property
    def cpp_layer_path(self):
        return "sadl/" + self.cpp_header

    @property
    def cuda_layer_path(self):
        return "sadl_cuda/" + self.cpp_header

    @property
    def converter_path(self):
        return f"converter/onnx2sadl/ops/{self.py_name}_converter.py"

    @property
    def serializer_path(self):
        return f"converter/onnx2sadl/writer/{self.py_name}_serializer.py"

    @property
    def emitter_path(self):
        return f"converter/sadl2onnx/emitters/{self.py_name}_emitter.py"

    @property
    def test_model_script_path(self):
        return f"utests/models/{self.py_name}_pytorch.py"

    @property
    def notes_path(self):
        return f"NEW_LAYER_{self.name}.md"

    @property
    def config_path(self):
        return f"NEW_LAYER_{self.name}.json"

    # --- checks ---------------------------------------------------------------
    def validate(self):
        """Raise ValueError on an inconsistent specification."""
        if not self._CAMEL.match(self.name):
            raise ValueError(f"layer name '{self.name}' must be CamelCase (e.g. MyOp)")
        if not self._SNAKE.match(self.py_name):
            raise ValueError(f"python name '{self.py_name}' must be snake_case")
        if not re.match(r"^layer_[a-z0-9_]+\.h$", self.cpp_header):
            raise ValueError(f"C++ header '{self.cpp_header}' must look like layer_xxx.h")
        if not 1 <= self.nb_inputs <= self.MAX_INPUTS:
            raise ValueError(f"number of inputs must be 1..{self.MAX_INPUTS}")
        for i in self.const_inputs:
            if not 0 < i < self.nb_inputs:
                raise ValueError(f"constant input {i} must be in 1..{self.nb_inputs - 1} (input 0 is the data)")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError("duplicate parameter names")
        if not self.onnx_ops or not all(re.match(r"^[A-Za-z][A-Za-z0-9_]*$", o) for o in self.onnx_ops):
            raise ValueError("invalid ONNX op type list")
        self._validate_choices()

    def _validate_choices(self):
        self._check_choice("converter_base", self.CONVERTER_BASES)
        self._check_choice("layout", self.LAYOUTS)
        self._check_choice("emitter", self.EMITTERS)
        self._check_choice("cuda", self.CUDA_MODES)
        if self.converter_base == "unary" and (self.nb_inputs != 1 or self.const_inputs):
            raise ValueError("a 'unary' converter needs exactly 1 input and no constant input")
        if self.converter_base == "binary" and self.nb_inputs != 2:
            raise ValueError("a 'binary' converter needs exactly 2 inputs")
        if self.emitter == "simple" and self.params:
            raise ValueError("a 'simple' emitter drops the parameters: use 'nchw' or 'custom'")
        if self.emitter == "nchw" and self.nb_inputs != 1:
            raise ValueError("an 'nchw' emitter handles a single input: use 'custom'")

    def _check_choice(self, field, choices):
        if getattr(self, field) not in choices:
            raise ValueError(f"{field} must be one of {', '.join(choices)}")

    # --- JSON -----------------------------------------------------------------
    def to_dict(self):
        return {
            "name": self.name,
            "sadl_id": self.sadl_id,
            "py_name": self.py_name,
            "cpp_header": self.cpp_header,
            "onnx_ops": self.onnx_ops,
            "nb_inputs": self.nb_inputs,
            "const_inputs": self.const_inputs,
            "params": [p.to_json() for p in self.params],
            "internal_quantizer": self.internal_quantizer,
            "converter_base": self.converter_base,
            "layout": self.layout,
            "emitter": self.emitter,
            "emitter_onnx_op": self.emitter_onnx_op,
            "cuda": self.cuda,
            "netron": self.netron,
            "test_model": self.test_model,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_dict(cls, data, default_id=None):
        data = dict(data)
        if "name" not in data:
            raise ValueError("the configuration has no 'name'")
        if data.get("sadl_id") is None:
            if default_id is None:
                raise ValueError("the configuration has no 'sadl_id'")
            data["sadl_id"] = default_id
        unknown = set(data) - set(cls(name="X", sadl_id=0).to_dict())
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
        return cls(**data)
