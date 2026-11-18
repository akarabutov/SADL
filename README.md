# SADL: Small Adhoc Deep-Learning Library

A header-only C++ library for neural network inference.
Models trained in any major framework (TensorFlow, PyTorch, …) can be exported to ONNX,
converted to the compact `.sadl` binary format, and then run in pure C++ with no external dependencies.

---

## Repository layout

```
sadl/        Header-only inference library (C++17)
converter/   ONNX → SADL converter (Python)
sample/      Example programs and integration scripts
utests/      Unit-test suite
tools/       Extra tooling (Netron visualiser patch)
```

---

## sadl — inference library

`sadl/` is a header-only C++ library.  Include `sadl/model.h` and you are ready to run inference.

Full API documentation (Model, Tensor, Dimensions, Layer, supported operations, build macros, SIMD levels):

**→ [sadl/README.md](sadl/README.md)**

---

## converter — ONNX to SADL

`converter/main.py` converts an ONNX file to the `.sadl` binary format.

```shell
python3 converter/main.py --input_onnx model.onnx --output model.sadl
```

Key options:

| Option | Description |
|--------|-------------|
| `--input_onnx FILE` | Input ONNX file |
| `--output FILE` | Output `.sadl` file |
| `--nchw` / `--nhwc` | Override the detected data layout |
| `--do_not_add_transpose_before NODE` | Disable auto-transpose insertion before a named node |
| `--do_not_add_transpose_after NODE` | Disable auto-transpose insertion after a named node |
| `--force_transpose_constant_tensor_dim4` | Force NCHW→NHWC transpose on 4-D constant tensors |
| `--info_string STR` | Embed an info string in the model (SADL v5+) |
| `--verbose` | Increase verbosity (repeat for more) |

NCHW graphs (e.g., PyTorch default) are automatically rewritten to NHWC; adapt your pre/post-processing accordingly.

### Quick examples

```shell
# TensorFlow 2
python3 sample/tf2.py            # produces tf2.onnx
python3 converter/main.py --input_onnx tf2.onnx --output tf2.sadl

# PyTorch
python3 sample/pytorch.py        # produces pytorch.onnx
python3 converter/main.py --input_onnx pytorch.onnx --output pytorch.sadl
```

### SADL to ONNX

`converter/sadl_to_onnx.py` does the reverse conversion (SADL0005 files only):

```shell
python3 converter/sadl_to_onnx.py --input_sadl model.sadl --output model.onnx
```

The ONNX model keeps the SADL NHWC layout: a PyTorch (NCHW) model comes back
with NHWC inputs and outputs. ONNX ops defined on NCHW (Conv, pooling, Resize,
GridSample...) are wrapped in transposes; consecutive transposes that cancel
out are removed (`--keep_transposes` disables this). Integer fixed-point
constants are de-quantized to float. Implementation: `converter/sadl2onnx/`
(one `LayerEmitter` per SADL layer, registered in `emitters/emitter_registry.py`).

### Updating an old model

```shell
sample_test/update_model_version old_model.sadl new_model.sadl
```

---

## sample — example programs

`sample/` contains C++ programs that demonstrate library usage, model diagnostics,
quantization, and sparsification, plus Python scripts to generate test models.

**→ [sample/README.md](sample/README.md)**

---

## utests — unit tests

`utests/` contains a C++ test binary built against the SADL headers, plus shell scripts that drive round-trip tests (ONNX inference via onnxruntime vs. SADL inference).

Build:

```shell
cd utests
cmake -B build .
cmake --build build
```

Run all tests:

```shell
./utests/check.sh
```

Run a single ONNX round-trip test:

```shell
./utests/utest.sh utests/models/my_op.onnx
./utests/utest.sh utests/models/my_op.onnx --no_transpose      # skip layout rewrite
./utests/utest.sh utests/models/my_op.onnx --no_onnxruntime    # skip onnxruntime comparison
```

Each test compares SADL inference output against stored reference values (`.results` files) and, when onnxruntime is available, against its output directly.

---

## tools — Netron visualiser

`tools/netron/` provides a patch that adds SADL model support to the
[Netron](https://github.com/lutzroeder/netron) neural-network visualiser.

```shell
git clone https://github.com/lutzroeder/netron.git
cd netron
git checkout v8.9.0
git apply /path/to/tools/netron/netron-sadl.patch
cd source
python3 -m http.server 8080
# then open http://127.0.0.1:8080 and drag-and-drop a .sadl file
```

See [tools/netron/README.md](tools/netron/README.md) for details.

---

## Developing new code

### C++ (sadl/)

**Formatting** — All C++ code must be formatted with `clang-format` using the project's
`.clang-format` file (Allman braces, 2-space indent, 160-column limit):

```shell
clang-format -i sadl/layer_myop.h
```

**Style summary**
- C++17, no external dependencies.
- Use `sadl::` namespace; new layers live in `sadl::layers::`.
- Header-only: no `.cpp` files in `sadl/`.
- Avoid dynamic allocation in hot paths; reuse the pre-allocated `output()` tensor.
- SIMD paths are optional but strongly encouraged for compute-heavy layers
  (see existing `layer_conv2d_3x3.h` for the pattern).

**Adding a new layer**
1. Create `sadl/layer_myop.h` following the pattern of an existing layer.
2. Register the new `OperationType` value in `sadl/layers.h`.
3. Handle the new op in the converter: add a class in `converter/onnx2sadl/ops/`
   (one `OpConverter` subclass per ONNX op), list it in `ops/op_catalog.py`, and,
   if the SADL layer has parameters, a serializer in `converter/onnx2sadl/writer/`
   registered in `writer/layer_serializer_registry.py`.
4. Add a unit test (see below).

**Unit test** — Every new layer must have a matching test in `utests/`:
1. Export a minimal ONNX model that exercises the op (a small Python script is fine).
2. Run `./utests/utest.sh utests/my_op.onnx` — this converts, runs, and compares outputs.
3. Save the resulting `.results` file and the `.onnx` file in `utests/`.
4. Add the test name to the appropriate loop in `utests/check.sh` so it runs automatically.

### Python (converter/)

**Formatting / linting** — The converter uses `flake8` (config in `.flake8`):

```shell
flake8 converter/
```

Key rules: max line length 88, complexity ≤ 12, `E501` ignored.
Format with `black` before committing (line length matches flake8 config).

```shell
black converter/
```

---

## License

SADL is licensed under the BSD-3-Clause.

---

## Citation

If you use this software in a publication:

```bibtex
@techreport{SADL2021,
      title       = "{SADL} Small Adhoc Deep-Learning Library",
      author      = "Franck Galpin and Pavel Nikitin and Thierry Dumas and Philippe Bordes",
      institution = "InterDigital",
      number      = "JVET-W0181",
      year        = 2021,
      month       = jul,
      url         = {https://jvet-experts.org/doc_end_user/current_document.php?id=11012}
}
```
