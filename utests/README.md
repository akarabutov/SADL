# SADL Unit Tests

This directory contains two test flows:

- ONNX round-trip tests: compare SADL float inference with ONNXRuntime reference output.
- int16 anchor tests: compare current SADL int16 inference with saved bit-exact anchors.

## Build

The shell scripts build the test binaries automatically. To build manually:

```sh
cd utests
cmake -B build .
cmake --build build
```

## ONNX Float Tests

Run the full ONNX test suite from the repository root:

```sh
utests/check.sh
```

`check.sh` builds the C++ runners, creates/uses `utests/results`, and runs the model list defined in the script. Each model is converted from ONNX to SADL, then checked against ONNXRuntime output with the scalar, AVX2, AVX512, memory-pool, and sparse runners where configured.

Run one ONNX model manually:

```sh
mkdir -p utests/results
cd utests/results
../utest.sh ../models/relu.onnx
```

Useful options:

```sh
../utest.sh ../models/slice.onnx --no_transpose
../utest.sh ../models/bitshift_LEFT__INT16.onnx --no_onnxruntime
```

## int16 Anchor Tests

The int16 tests do not use ONNXRuntime. They use existing `.sadl` int16 models and one anchor file per model.

Generate anchors for the NNVC int16 models:

```sh
utests/generate_anchors.sh utests/nnvc_models
```

This creates files next to each model, for example:

```txt
utests/nnvc_models/nnlf_lop7_model_int16.sadl
utests/nnvc_models/nnlf_lop7_model_int16.int16_anchor
```

Check the models against their anchors:

```sh
utests/check_models.sh utests/nnvc_models
```

The checker runs all available int16 runners:

- `test_sadl_int16_scalar`
- `test_sadl_int16_avx2`
- `test_sadl_int16_avx512`
- `test_sadl_int16_avx2_mempool`
- `test_sadl_int16_avx2_sparse`

Expected output is one line per model/runner:

```txt
   [PASS] nnlf_lop7_model_int16.sadl test_sadl_int16_scalar
```

`[PASS]` is printed in green and `[FAIL]` in red.

## int16 Options

Default random input range is `[-512, 512]` with seed `5`.

```sh
utests/generate_anchors.sh --min -256 --max 256 --seed 12 utests/nnvc_models
```

Use a different anchor suffix:

```sh
utests/generate_anchors.sh --suffix .my_anchor utests/nnvc_models
utests/check_models.sh --suffix .my_anchor utests/nnvc_models
```

Regenerate anchors only when a model changes or when an intentional implementation change updates the bit-exact int16 output.
