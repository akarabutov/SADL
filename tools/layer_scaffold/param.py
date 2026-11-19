# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import re


class Param:
    """One SADL layer parameter: a value read by ``loadInternal``.

    Kinds and binary layout (identical in C++, the Python serializer and reader,
    the dumper and Netron):
        int      int32
        bool     1 byte
        int[]    int32 count, then count int32
        int[N]   N int32 (no count)
    """

    KINDS = ("int", "bool", "int[]", "int[N]")
    _NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _FIXED = re.compile(r"^int\[(\d+)\]$")
    RESERVED = ("q", "data")  # "q" is the internal quantizer, "data" the ONNX node

    def __init__(self, name, kind):
        if not self._NAME.match(name):
            raise ValueError(f"invalid parameter name '{name}'")
        if name in self.RESERVED:
            raise ValueError(f"parameter name '{name}' is reserved")
        fixed = self._FIXED.match(kind)
        if kind not in ("int", "bool", "int[]") and not (fixed and int(fixed.group(1)) > 0):
            raise ValueError(f"invalid parameter type '{kind}' (int, bool, int[], int[N])")
        self.name = name
        self.kind = kind
        self.fixed_size = int(fixed.group(1)) if fixed else None

    @classmethod
    def parse(cls, text):
        """``"name:type"`` (type defaults to int)."""
        name, _, kind = text.partition(":")
        return cls(name.strip(), kind.strip() or "int")

    def to_json(self):
        return f"{self.name}:{self.kind}"

    @property
    def is_list(self):
        return self.kind == "int[]" or self.fixed_size is not None

    @property
    def member(self):
        return "m_" + self.name

    def describe(self):
        if self.kind == "int":
            return "int32"
        if self.kind == "bool":
            return "bool (1 byte)"
        if self.kind == "int[]":
            return "int32 count + int32 values"
        return f"{self.fixed_size} x int32"

    # --- C++ ------------------------------------------------------------------
    def cpp_decl(self):
        if self.kind == "int":
            return f"int32_t {self.member} = 0;"
        if self.kind == "bool":
            return f"bool {self.member} = false;"
        if self.kind == "int[]":
            return f"std::vector<int32_t> {self.member};"
        return f"std::array<int32_t, {self.fixed_size}> {self.member}{{}};"

    def cpp_read(self, layer):
        """Lines of loadInternal reading this parameter (2-space indent)."""
        m = self.member
        if self.kind in ("int", "bool"):
            return [f"  file.read((char *) &{m}, sizeof({m}));"]
        if self.fixed_size is not None:
            return [f"  file.read((char *) {m}.data(), sizeof(int32_t) * {self.fixed_size});"]
        return [
            "  {",
            "    int32_t n = 0;",
            "    file.read((char *) &n, sizeof(n));",
            "    if (!file || n < 0 || n > 4096)",
            "    {",
            f'      std::cerr << "[ERROR] {layer}: invalid size of {self.name}: " << n << std::endl;',
            "      return false;",
            "    }",
            f"    {m}.resize(n);",
            f"    file.read((char *) {m}.data(), sizeof(int32_t) * n);",
            "  }",
        ]

    def cpp_dump(self):
        """Lines of dump() writing this parameter (2-space indent)."""
        m = self.member
        if self.kind in ("int", "bool"):
            return [f"  file.write((const char *) &this->{m}, sizeof(this->{m}));"]
        if self.fixed_size is not None:
            return [f"  file.write((const char *) this->{m}.data(), sizeof(int32_t) * {self.fixed_size});"]
        return [
            "  {",
            f"    int32_t n = (int32_t) this->{m}.size();",
            "    file.write((const char *) &n, sizeof(n));",
            f"    file.write((const char *) this->{m}.data(), sizeof(int32_t) * n);",
            "  }",
        ]

    # --- Python ---------------------------------------------------------------
    def serializer_line(self):
        """BinaryStream call of the onnx2sadl serializer."""
        n = self.name
        if self.kind == "int":
            return f'stream.i32(node["{n}"], "{n}")'
        if self.kind == "bool":
            return f'stream.bool(node["{n}"], "{n}")'
        if self.kind == "int[]":
            return f'stream.i32_list(node["{n}"], "{n}")'
        return f'stream.i32_values(node["{n}"])'

    def reader_field(self):
        """ListParamsReader field of the sadl2onnx reader."""
        if self.kind == "int":
            return self.name
        if self.kind == "bool":
            return self.name + "?"
        if self.kind == "int[]":
            return self.name + "[]"
        return f"{self.name}[{self.fixed_size}]"

    def onnx_value(self, attr):
        """Python expression turning the ONNX AttributeProto ``attr`` into the value."""
        if self.kind == "int":
            return f"{attr}.i"
        if self.kind == "bool":
            return f"bool({attr}.i)"
        return f"list({attr}.ints)"

    def sample_value(self):
        """A test value (used by the round-trip check)."""
        if self.kind == "int":
            return 7
        if self.kind == "bool":
            return True
        if self.kind == "int[]":
            return [3, -1, 5]
        return list(range(1, self.fixed_size + 1))

    # --- Netron ---------------------------------------------------------------
    def netron_value(self):
        if self.kind == "int":
            return "cursor.readI32()"
        if self.kind == "bool":
            return "cursor.readU8() !== 0"
        if self.kind == "int[]":
            return "sadl.decode.readCountedI32(cursor)"
        return f"Array.from({{ length: {self.fixed_size} }}, () => cursor.readI32())"
