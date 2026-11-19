# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import json
import os
import shutil
import subprocess
import tempfile


class Verifier:
    """Checks run after the generation. Each check returns (status, message), status in OK/FAIL/SKIP."""

    # Runs in a separate interpreter (the converter's Python, with onnx and numpy).
    PYTHON_CHECK = r'''
import io, json, sys
sys.path.insert(0, "converter")
try:
    import numpy, onnx  # noqa: F401
except ImportError as e:
    print("onnx/numpy not available:", e)
    sys.exit(3)
from onnx2sadl.types.op_type import OpType
from onnx2sadl.ops.op_catalog import OpCatalog
from onnx2sadl.ops.op_registry import OpRegistry
from onnx2sadl.writer.binary_stream import BinaryStream
from onnx2sadl.writer.layer_serializer_registry import LayerSerializerRegistry
from sadl2onnx.reader.binary_reader import BinaryReader
from sadl2onnx.reader.layer_params_reader_registry import LayerParamsReaderRegistry
from sadl2onnx.emitters.emitter_registry import EmitterRegistry

spec = json.loads(sys.argv[1])
op = OpType[spec["name"]]
assert op.value == spec["sadl_id"], "OpType value"
registry = OpRegistry(cls() for cls in OpCatalog.CONVERTERS)  # also rejects duplicate ONNX ops
for onnx_op in spec["onnx_ops"]:
    conv = registry.get(onnx_op)
    assert conv is not None and type(conv).__name__ == spec["name"] + "Converter", "converter of " + onnx_op
assert EmitterRegistry().get(op) is not None, "emitter"

# parameters: serializer -> reader round trip, and size = what loadInternal reads
values = spec["sample_values"]
buf = io.BytesIO()
serializer = LayerSerializerRegistry().get(op)
serializer.write(BinaryStream(buf), values)
if serializer.writes_quantizer:
    BinaryStream(buf).i32(5)
data = buf.getvalue()
assert len(data) == spec["payload_size"], "serializer wrote %d bytes, loadInternal reads %d" % (
    len(data), spec["payload_size"])
reader = LayerParamsReaderRegistry().get(op)
r = BinaryReader(data)
got = reader.read(r)
if reader.reads_quantizer:
    assert r.i32() == 5, "quantizer"
assert bool(serializer.writes_quantizer) == bool(reader.reads_quantizer), "quantizer flags"
assert r.pos == len(data), "reader consumed %d of %d bytes" % (r.pos, len(data))
assert got == values, "read %r, wrote %r" % (got, values)
print("registries OK; parameters round trip OK (%d bytes)" % len(data))
'''

    def __init__(self, repository, spec, python=None, cxx=None):
        self.repo = repository
        self.spec = spec
        self.python = python or os.environ.get("PYTHON") or shutil.which("python3") or shutil.which("python")
        self.cxx = cxx

    def _run(self, args, **kwargs):
        return subprocess.run(args, cwd=self.repo.root, capture_output=True, text=True, **kwargs)

    def _payload_size(self):
        """Bytes read by the generated loadInternal for the sample values."""
        size = 0
        for p in self.spec.params:
            if p.kind == "bool":
                size += 1
            elif p.kind == "int[]":
                size += 4 + 4 * len(p.sample_value())
            elif p.fixed_size is not None:
                size += 4 * p.fixed_size
            else:
                size += 4
        return size + (4 if self.spec.internal_quantizer else 0)

    # --- checks ---------------------------------------------------------------
    def python_registries(self):
        if not self.python:
            return "SKIP", "no python interpreter found (use --python)"
        data = self.spec.to_dict()
        data["sample_values"] = {p.name: p.sample_value() for p in self.spec.params}
        data["payload_size"] = self._payload_size()
        res = self._run([self.python, "-c", self.PYTHON_CHECK, json.dumps(data)])
        text = (res.stdout + res.stderr).strip()
        if res.returncode == 3:
            return "SKIP", text + " (use --python with the converter's environment)"
        return ("OK" if res.returncode == 0 else "FAIL"), text.splitlines()[-1] if res.returncode == 0 else text

    def flake8(self, files):
        files = [f for f in files if f.endswith(".py") and f.startswith("converter/")]
        if not files:
            return "SKIP", "no converter file"
        if not self.python:
            return "SKIP", "no python interpreter found"
        res = self._run([self.python, "-m", "flake8", *files])
        text = (res.stdout + res.stderr).strip()
        if "No module named flake8" in text:
            return "SKIP", "flake8 not installed"
        return ("OK" if res.returncode == 0 else "FAIL"), text or f"{len(files)} file(s) clean"

    def find_cxx(self):
        if self.cxx:
            return self.cxx
        if os.environ.get("CXX"):
            return os.environ["CXX"]
        for name in ("cl", "g++", "clang++", "c++"):
            if shutil.which(name):
                return name
        return None

    def cpp_build(self):
        """Compile the sample tools using every C++ registration (factory, copy, dump, quantization)."""
        cxx = self.find_cxx()
        if cxx is None:
            return "SKIP", "no C++ compiler found (set CXX, or run from a compiler prompt)"
        sources = ["sample/naive_quantization.cpp", "sample/update_model_version.cpp"]
        msvc = os.path.basename(cxx).lower() in ("cl", "cl.exe")
        with tempfile.TemporaryDirectory() as tmp:
            for src in sources:
                obj = os.path.join(tmp, os.path.basename(src) + ".o")
                if msvc:
                    args = [cxx, "/nologo", "/std:c++17", "/EHsc", "/W0", "/c", "/I", str(self.repo.root),
                            src, "/Fo" + obj]
                else:
                    args = [cxx, "-std=c++17", "-c", "-I", str(self.repo.root), src, "-o", obj]
                res = self._run(args)
                if res.returncode != 0:
                    return "FAIL", f"{' '.join(args)}\n{(res.stdout + res.stderr).strip()}"
        note = " (sadl_cuda not compiled: needs nvcc)" if self.spec.cuda != "skip" else ""
        return "OK", f"{cxx}: {', '.join(sources)} compile{note}"
