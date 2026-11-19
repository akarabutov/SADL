# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_spec import LayerSpec
from .param import Param


class Questionnaire:
    """Asks the questions of CONTRIBUTE.md section 0 and builds a LayerSpec."""

    def __init__(self, repository, prompter):
        self.repo = repository
        self.p = prompter

    def run(self):
        p = self.p
        sadl_id = self.repo.operation_type_count()
        p.say("New SADL layer. Empty answer = default in [brackets]; '?' = help.")
        p.say("")

        name = p.ask("Layer name (CamelCase)", None, self._layer_name,
                     "C++ class sadl::layers::<Name>, OpType.<Name> in Python, e.g. MyOp")
        p.say(f"  SADL id: {sadl_id} (next free id, OperationTypeCount in sadl/layer.h)")
        py_name = p.ask("Python module prefix", LayerSpec.snake_case(name), self._not_empty,
                        "files ops/<prefix>_converter.py, writer/<prefix>_serializer.py...")
        cpp_header = p.ask("C++ header", f"layer_{name.lower()}.h", self._not_empty)
        onnx_ops = p.ask("ONNX op type(s) converted to this layer, comma separated", name, self._onnx_ops,
                         "op_type of the ONNX nodes handled by the new converter")

        p.say("")
        nb_inputs = p.ask(f"Number of inputs (1..{LayerSpec.MAX_INPUTS})", 1, self._nb_inputs,
                          "input tensors of the SADL layer, including constant ones")
        const_inputs = []
        if nb_inputs > 1:
            const_inputs = p.ask("Constant inputs (indices, comma separated, empty = none)", "",
                                 self._indices, "inputs that are always constants (weights, bounds...): "
                                 "they become Const layers")
        params = self._ask_params()
        quantizer = p.yes_no("Internal quantizer (like Conv2D/MatMul: an int32 'q' read last)", False)

        p.say("")
        base_default = LayerSpec(name, sadl_id, nb_inputs=nb_inputs,
                                 const_inputs=const_inputs).default_converter_base()
        converter_base = p.choice("ONNX -> SADL converter base class", {
            "unary": "UnaryConverter: 1 input, attributes via layer_attrs()",
            "binary": "BinaryConverter: 2 inputs, initializers become Const layers",
            "op": "OpConverter: free convert() (constant inputs, several inputs)",
        }, base_default)
        layout = p.choice("Layout handling for NCHW (PyTorch) models", {
            "none": "element-wise: layout does not matter",
            "axis": "remap NCHW axes/attributes in the converter (TODO left)",
            "transpose": "transpose around the layer: it receives NCHW data",
        }, "none")
        emitter_default = "custom" if params else "simple"
        emitter = p.choice("SADL -> ONNX emitter", {
            "simple": "SimpleOpEmitter: 1:1 ONNX op, no parameters, no new file",
            "nchw": "NchwOpEmitter: 1 input, ONNX op computed in NCHW",
            "custom": "own LayerEmitter class (new file)",
        }, emitter_default)
        emitter_onnx_op = p.ask("ONNX op produced by the emitter", onnx_ops[0], self._not_empty)

        p.say("")
        cuda = p.choice("CUDA backend (sadl_cuda/)", {
            "skip": "do not add the layer to sadl_cuda",
            "cpu": "CUDA model runs it on the host (simplest)",
            "gpu": "CUDA kernel skeleton",
        }, "skip")
        netron = p.yes_no("Register in the Netron viewer (tools/netron/sadl.js)", True)
        test_model = p.yes_no(f"Create utests/models/{py_name}_pytorch.py (test model export)", True)

        return LayerSpec(
            name=name, sadl_id=sadl_id, py_name=py_name, cpp_header=cpp_header, onnx_ops=onnx_ops,
            nb_inputs=nb_inputs, const_inputs=const_inputs, params=params, internal_quantizer=quantizer,
            converter_base=converter_base, layout=layout, emitter=emitter, emitter_onnx_op=emitter_onnx_op,
            cuda=cuda, netron=netron, test_model=test_model,
        )

    def _ask_params(self):
        self.p.say("Parameters read by loadInternal, in binary order, as name:type.")
        self.p.say("  types: int (int32), bool, int[] (count + values), int[N] (N values). Empty = done.")
        params = []
        while True:
            text = self.p.ask(f"  parameter {len(params) + 1}", "", None)
            if not text:
                return params
            try:
                param = Param.parse(text)
            except ValueError as e:
                self.p.say(f"  ! {e}")
                continue
            if any(q.name == param.name for q in params):
                self.p.say(f"  ! {param.name} already given")
                continue
            params.append(param)

    # --- converters (raise ValueError to re-ask) ------------------------------
    def _layer_name(self, text):
        spec = LayerSpec(text, 0)
        if not LayerSpec._CAMEL.match(text):
            raise ValueError("use CamelCase, e.g. MyOp")
        if spec.name in self.repo.enum_entries("sadl/layer.h"):
            raise ValueError(f"{text} already exists in sadl/layer.h")
        return text

    @staticmethod
    def _not_empty(text):
        if not text:
            raise ValueError("an answer is needed")
        return text

    def _onnx_ops(self, text):
        ops = [o.strip() for o in text.split(",") if o.strip()]
        if not ops:
            raise ValueError("at least one ONNX op type")
        handled = self.repo.converted_onnx_ops()
        for op in ops:
            if op in handled:
                raise ValueError(f"{op} is already converted by ops/{handled[op]}")
        return ops

    @staticmethod
    def _nb_inputs(text):
        n = int(text)
        if not 1 <= n <= LayerSpec.MAX_INPUTS:
            raise ValueError(f"1..{LayerSpec.MAX_INPUTS}")
        return n

    @staticmethod
    def _indices(text):
        return [int(i) for i in text.replace(" ", "").split(",") if i]
