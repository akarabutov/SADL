# sample — Example Programs

This directory contains example C++ programs that demonstrate how to use the SADL inference library,
along with Python scripts to generate small test models.

---

## Build

```shell
cmake -B ../sample_test .
cmake --build ../sample_test
```

Binaries land in `../sample_test/`.  On ARM64 only the scalar target is built; AVX2/AVX-512 variants require an x86-64 host.

Optional: enable sparse MatMul support:

```shell
cmake -B ../sample_test -DSPARSE_MATMULT_SUPPORT=ON .
cmake --build ../sample_test
```

---

## Generating example models

Two Python scripts produce small ONNX models that can then be converted with the converter.
Both implement the same toy CNN: a residual-block classification network on 16×16 RGB inputs.

### TensorFlow 2 → ONNX

```shell
# requires: tensorflow, tf2onnx, onnx
python3 sample/tf2.py
```

Produces `tf2.onnx` in the current directory and prints the model output for a synthetic input.
The model uses NHWC layout throughout (TensorFlow native), so no layout rewrite is needed.

Convert to SADL:

```shell
python3 converter/main.py --input_onnx tf2.onnx --output tf2.sadl
```

### PyTorch → ONNX

```shell
# requires: torch, numpy
python3 sample/pytorch.py
```

Produces `pytorch.onnx`.
PyTorch uses NCHW layout; the converter automatically rewrites the graph to NHWC.
Adapt any pre/post-processing code accordingly (transpose inputs before inference,
transpose outputs after if needed).

Convert to SADL:

```shell
python3 converter/main.py --input_onnx pytorch.onnx --output pytorch.sadl
```

---

## Programs

### `sample` — Timed inference

```
sample_generic   model.sadl [override_size]
sample_simd256   model.sadl [override_size]
sample_simd512   model.sadl [override_size]
```

Loads a model, runs one inference with linearly-spaced synthetic inputs, and prints the elapsed time in milliseconds.

- `sample_generic` — scalar (no SIMD), baseline
- `sample_simd256` — AVX2 path (requires `-mavx2 -mfma` at build time)
- `sample_simd512` — AVX-512 path (requires `-mavx512bw -mavx512f -mavx512dq`)

The optional `override_size` argument replaces the spatial dimensions (height and width) of all inputs.
Useful to measure performance at a different resolution than the model was trained on.

Example:

```shell
sample_test/sample_generic  tf2.sadl
# [INFO] 6.80042 ms

sample_test/sample_simd512  tf2.sadl
# [INFO] 0.449472 ms
```

The three variants are compiled from the same source (`sample.cpp`); only the compiler flags differ.
The model type (`float`, `int32`, `int16`) is detected automatically at run time.

---

### `count_mac` — Complexity assessment

```
count_mac model.sadl [override_size]
```

Loads and initialises the model (scalar mode, no SIMD) and counts the number of operations
(MACs and other ops) per layer, then prints per-layer and total figures.

Must be built in scalar mode (enforced by the CMakeLists) so that every layer's MAC counter
is accurate — SIMD code paths may merge or skip iterations.

```shell
sample_test/count_mac tf2.sadl
# [INFO] layer 2 ... [Conv2D]: 2048 op
# ...
# [INFO] 216160 MACs
```

---

### `debug_model` — Layer-by-layer diagnostic

```
debug_model model.sadl
```

Built with all diagnostic macros enabled (`DEBUG_VALUES`, `DEBUG_MODEL`, `DEBUG_PRINT`,
`DEBUG_SIMD`, `DEBUG_MODEL_ANALYZE`) and with AVX2 so that SIMD paths are reachable.

Runs through three phases and prints detailed information for each:

**Loading** — magic number, version, layer types, tensor shapes, quantizer values.

**Initialisation** — shape propagation, inserted copy layers, memory layout.

**Inference** — per-layer output values, SIMD/generic path taken, overflow warnings.

Common diagnostic patterns:

```shell
# show only layers that fall back to the generic (non-SIMD) path
debug_model tf2.sadl | grep WARN

# show per-layer complexity
debug_model tf2.sadl | grep "Complexity\|layer.*op"

# show the final output value
debug_model tf2.sadl | grep -A5 "output 0"
```

A `[WARN] generic version …` line means that layer has no SIMD implementation for
the given kernel size / channel count combination — a candidate for optimisation.

---

### `naive_quantization` — Float → int16 conversion

```
naive_quantization model_float.sadl model_int16.sadl
```

Converts a float SADL model to int16 by computing per-layer quantizer values
from the weight statistics.  The resulting model can be run with `Model<int16_t>`.

This is a *naive* scheme (uniform quantization based on weight range only);
use it as a starting point or for quick experiments.

---

### `quantization_test` — int16 accuracy check

```
quantization_test model_float.sadl model_int16.sadl inputs_range shift max_error
```

Runs both models on random inputs and verifies that the int16 output is within
`max_error` of the dequantized float output.

| Argument | Description |
|----------|-------------|
| `model_float.sadl` | Reference float model |
| `model_int16.sadl` | Quantized model to validate |
| `inputs_range` | Input value range (power of 2, equals 2^N where N is the Placeholder quantizer) |
| `shift` | Right-shift applied to the int16 output before comparison |
| `max_error` | Maximum tolerated absolute error |

Exits with a `PASSED` or `FAILED` message.

---

### `sparsify_with_packing` — Sparse MatMul preprocessing

```
sparsify_with_packing model.sadl sparse_model.sadl
```

Analyses weight matrices and converts those that are sufficiently sparse
(fraction of zeros above threshold) to a packed sparse representation.
The output model is written using the same SADL format; at inference time
pass `-DSPARSE_SUPPORT=1` so the library activates the sparse code path.

Must be built with `SPARSE_MATMULT_SUPPORT=ON` to take effect
(the binary still runs without it but prints a warning and produces a dense model).

---

### `update_model_version` — Format migration

```
update_model_version old_model.sadl new_model.sadl [info_string]
```

Re-serialises an older `.sadl` file to the current format version (v5).
The optional `info_string` is embedded in the output file and retrievable via `Model::info()`.

---

## Integration smoke-test

`sample_test.sh` runs the full pipeline end-to-end: generates ONNX models,
converts them to SADL, runs inference with every binary, and checks the outputs.

```shell
./sample/sample_test.sh
```

Run this after any change to the converter or the library to catch regressions.

---

## Helper header — `helper.h`

`helper.h` is a shared utility header (not a standalone program) used by all sample programs.
It provides:

- `getModelType(filename)` — read the tensor type (`float`, `int32`, `int16`) from a `.sadl` file
  without loading the full model; used for compile-time type dispatch.
- `showMagic(filename)` — return the magic string (version identifier) from a file header.
