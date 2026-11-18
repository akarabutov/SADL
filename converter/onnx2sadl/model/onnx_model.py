# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import numpy as np

from ..errors.conversion_error import ConversionError
from ..types.onnx_dtype import OnnxDType


class OnnxModel:
    """Read-mostly view of an onnx ModelProto with name -> node indexes.

    Lookups follow the original converter semantics: when a name could match
    several kinds of objects, nodes are searched first, then initializers,
    then graph inputs.
    """

    def __init__(self, proto):
        self.proto = proto
        self.graph = proto.graph
        self._build_indexes()

    # --- indexes ----------------------------------------------------------
    def _build_indexes(self):
        g = self.graph
        self._initializers = {}
        for t in g.initializer:
            self._initializers.setdefault(t.name, t)
        self._inputs = {}
        for i in g.input:
            self._inputs.setdefault(i.name, i)
        self._outputs = {}
        for o in g.output:
            self._outputs.setdefault(o.name, o)
        self._value_infos = {}
        for v in g.value_info:
            self._value_infos.setdefault(v.name, v)
        self._producers = {}
        self._first_output_producers = {}
        self._consumers = {}
        for node in g.node:
            for out in node.output:
                self._producers.setdefault(out, node)
            if len(node.output) > 0:
                self._first_output_producers.setdefault(node.output[0], node)
            for inp in node.input:
                self._consumers.setdefault(inp, []).append(node)

    @property
    def producer_name(self):
        return self.proto.producer_name

    @property
    def nodes(self):
        return self.graph.node

    @property
    def inputs(self):
        return self.graph.input

    @property
    def outputs(self):
        return self.graph.output

    @property
    def initializers(self):
        return self.graph.initializer

    # --- basic lookups ----------------------------------------------------
    @staticmethod
    def get_attribute(node, name):
        for a in node.attribute:
            if a.name == name:
                return a
        return None

    def initializer(self, name):
        return self._initializers.get(name)

    def is_initializer(self, name):
        return name in self._initializers

    def consumers(self, name):
        """Nodes using ``name`` as input (a node appears once per use)."""
        return list(self._consumers.get(name, []))

    def first_output_producer(self, name):
        """Node whose *first* output is ``name``."""
        return self._first_output_producers.get(name)

    def producer(self, name):
        """Node, initializer or graph input producing ``name`` (or None)."""
        if name in self._producers:
            return self._producers[name]
        if name in self._initializers:
            return self._initializers[name]
        return self._inputs.get(name)

    def producer_skip_identity(self, name):
        """Producing node (Identity skipped) or graph input; initializers are ignored."""
        node = self._producers.get(name)
        if node is not None:
            if node.op_type == "Identity":
                return self.producer_skip_identity(node.input[0])
            return node
        return self._inputs.get(name)

    # --- shapes -----------------------------------------------------------
    def declared_dims(self, name):
        """Dims of an initializer / Constant node / graph input ([] if unknown)."""
        if name in self._initializers:
            return self._initializers[name].dims
        node = self._first_output_producers.get(name)
        if node is not None:
            a = self.get_attribute(node, "value")
            return a.t.dims if a is not None else []
        if name in self._inputs:
            return self._inputs[name].type.tensor_type.shape.dim
        raise ConversionError("[ERROR] unable to extract dims in {}".format(name))

    @staticmethod
    def _dims_from_value_info(vi):
        dims = []
        if not vi.type.HasField("tensor_type"):
            return dims
        ttype = vi.type.tensor_type
        if not ttype.HasField("shape"):
            return dims
        for d in ttype.shape.dim:
            which = d.WhichOneof("value")
            if which == "dim_value":
                dims.append(d.dim_value)
            elif which == "dim_param":
                dims.append(d.dim_param)
            else:
                dims.append(None)
        return dims

    def shape(self, name):
        """Best known shape of ``name`` (uses shape-inference value_info)."""
        if name in self._initializers:
            return [int(x) for x in self._initializers[name].dims]
        node = self._first_output_producers.get(name)
        if node is not None:
            a = self.get_attribute(node, "value")
            if a is not None and a.t is not None:
                return [int(x) for x in a.t.dims]
        for table in (self._inputs, self._outputs, self._value_infos):
            if name in table:
                return self._dims_from_value_info(table[name])
        raise ConversionError("[ERROR] unable to extract dim values in {}".format(name))

    def try_shape(self, name):
        try:
            return self.shape(name)
        except ConversionError:
            return None

    def should_insert_4d_transpose(self, name):
        dims = self.try_shape(name)
        if dims is not None:
            return len(dims) == 4
        n = self.producer_skip_identity(name)
        if n is None:
            return True
        if hasattr(n, "type") and hasattr(n.type, "tensor_type"):
            return len(n.type.tensor_type.shape.dim) == 4
        return True

    # --- constants --------------------------------------------------------
    def constant_tensor(self, name):
        """TensorProto of an initializer or a direct Constant node (no Identity)."""
        t = self.initializer(name)
        if t is not None:
            return t
        n = self.producer(name)
        if n is None or not hasattr(n, "op_type") or n.op_type != "Constant":
            return None
        a = self.get_attribute(n, "value")
        if a is None or a.t is None:
            return None
        return a.t

    def is_static_tensor(self, name):
        """True for an initializer or the output of a Constant node."""
        if self.initializer(name) is not None:
            return True
        n = self.producer(name)
        return n is not None and hasattr(n, "op_type") and n.op_type == "Constant"

    def constant_ints(self, name):
        """Integer values of a constant INT64/INT32 tensor, None if not constant."""
        t = self.constant_tensor(name)
        if t is None:
            return None
        if t.data_type == OnnxDType.INT64:
            x = np.frombuffer(t.raw_data, dtype=np.int64)
            if len(x) == 0:
                x = np.asarray(t.int64_data, dtype=np.int64)
            return [int(v) for v in x.tolist()]
        if t.data_type == OnnxDType.INT32:
            x = np.frombuffer(t.raw_data, dtype=np.int32)
            if len(x) == 0:
                x = np.asarray(t.int32_data, dtype=np.int32)
            return [int(v) for v in x.tolist()]
        raise ValueError(f"Expected INT64/INT32 constant for {name}")

    def initializer_or_constant_int64(self, name):
        """int64 raw values of an initializer, else of the producing Constant node."""
        tensor = self.initializer(name)
        if tensor is None:
            tensor = self.get_attribute(self.producer(name), "value").t
        if tensor.data_type != OnnxDType.INT64:
            raise ConversionError("[ERROR] bad node type fpr getDims {}".format(tensor))
        return np.frombuffer(tensor.raw_data, dtype=np.int64).tolist()

    # --- convolution weights ----------------------------------------------
    def is_conv_weight(self, name):
        for consumer in self.consumers(name):
            if consumer.op_type == "Conv" and len(consumer.input) > 1:
                if consumer.input[1] == name:
                    return True
        return False

    def conv_weight_dims(self, name):
        """[O, I, kh, kw] of a (possibly dynamic) Conv weight, None if unknown."""
        init = self.initializer(name)
        if init is not None:
            return [int(x) for x in init.dims]
        prod = self.producer(name)
        if prod is None:
            return None
        if hasattr(prod, "op_type") and prod.op_type == "Constant":
            a = self.get_attribute(prod, "value")
            if a is not None and a.t is not None:
                return [int(x) for x in a.t.dims]
            return None
        if hasattr(prod, "op_type") and prod.op_type == "Reshape":  # dynamic conv
            if len(prod.input) < 2:
                return None
            return self.constant_ints(prod.input[1])
        return None

    # --- mutation ---------------------------------------------------------
    def replace_input(self, old_name, new_name, verbose=0):
        """Make every node reading ``old_name`` read ``new_name`` instead."""
        for node in self.graph.node:
            for i in range(len(node.input)):
                if node.input[i] == old_name:
                    if verbose > 2:
                        print("Replace ", node, node.input[i], "->", new_name)
                    node.input[i] = new_name
        self._build_indexes()
