# Contributing: adding a new layer to SADL

This guide lists everything needed to add a new layer (operation) to SADL and
make it reachable from the converters: the C++ inference code, the auxiliary
C++ tools, the ONNX → SADL converter, the SADL → ONNX converter, and the tests.

Throughout the guide the new layer is called **`MyOp`**, with SADL id **41**
and one integer parameter **`mode`**. Replace these with your own values.

---

## Quick start: `tools/new_layer.py`

Most of this guide is mechanical: the script below does it for you, then this
guide is the reference for what it leaves to write by hand.

```shell
python tools/new_layer.py              # asks the questions of section 0
python tools/new_layer.py --dry-run    # same, shows the diff and writes nothing
```

The script asks for the layer name, the ONNX op type(s), the number of inputs
and which ones are constant, the parameters (`name:type`, with `int`, `bool`,
`int[]` or `int[N]`, in binary order), the internal quantizer, the converter
base class, the layout handling, the emitter, the CUDA variant, Netron and the
test model. Then it:

* **creates** `sadl/layer_myop.h`, the converter, the serializer and the emitter
  if needed, `sadl_cuda/layer_myop.h` and `utests/models/my_op_pytorch.py`;
* **registers** the layer in every file of the checklist (section 7): ids,
  `layers.h`, `model.h`, `copy.h`, `dumper.h`, `naive_quantization.cpp`,
  CUDA, Netron, `op_type.py`, `op_catalog.py`, serializer, reader and emitter
  registries, plus a reminder in `utests/check.sh`;
* **generates completely** everything that only depends on the parameters and
  must agree byte for byte: `loadInternal`, `dump()`, the `copy.h` case, the
  serializer, the `ListParamsReader` entry and the Netron decoder;
* **marks** what is left with `TODO(new_layer MyOp)` (the computation in `apply`,
  shape checks in `init`, the ONNX attribute mapping...) and writes
  `NEW_LAYER_MyOp.md`: answers, changed files, manual steps, checklist;
* **checks** the result: the registries load, the parameters survive a
  serializer → reader round trip with the size `loadInternal` reads, `flake8`
  passes, and with `--build-check` the sample tools compile.

Nothing is written if a name, id or ONNX op is already taken. A change whose
anchor is not found in an existing file is not guessed: it is listed with its
code under "Manual steps" in `NEW_LAYER_MyOp.md`. Line endings are preserved.

```shell
python tools/new_layer.py --check MyOp                 # remaining TODO markers
python tools/new_layer.py --config NEW_LAYER_MyOp.json --yes   # replay the answers
python tools/new_layer.py --python /path/to/venv/python --build-check   # checks with the converter's env and a C++ compiler
```

Delete `NEW_LAYER_MyOp.md` and `NEW_LAYER_MyOp.json` before committing.

---

## 0. Before you start

Decide the following, since several files must agree on them:

| Decision | Where it matters | Notes |
|---|---|---|
| **SADL id** | `sadl/layer.h`, `converter/onnx2sadl/types/op_type.py`, `tools/netron/sadl.js`, `sadl_cuda/layer.h` | Next free id (41 at the time of writing). Ids are stored in `.sadl` files: never reuse or renumber an existing one. |
| **Layout** | C++ layer, ONNX converter | SADL tensors are **NHWC** (`[N, H, W, C]`). ONNX models from PyTorch are NCHW: axes, pads, shapes given in NCHW must be remapped (or the input transposed) by the converter. |
| **Parameters** | C++ `loadInternal`, Python serializer and reader, `sample/dumper.h`, `sample/copy.h` | Everything not given as an input tensor. Their binary order must be **identical** in every place that reads or writes them. |
| **Constant inputs** | C++ layer, ONNX converter | Inputs that are always constants (weights, shapes, axes...) become `Const` layers. |
| **Quantization** | C++ `apply`, `sample/naive_quantization.cpp` | Integer models use fixed point: a value `x` is stored as `round(x·2^q)`, where `q` is the tensor's `quantizer`. The layer must handle inputs with different quantizers and set its output quantizer. |

The file format is documented in `converter/onnx2sadl/writer/sadl_writer.py`.
Each layer is stored as: `id, op, name, input ids`, followed by its
**parameters** (written by a serializer, read by `loadInternal`).

---

## 1. C++ inference library (`sadl/`)

### 1.1 Operation id — `sadl/layer.h`

Add the id to `OperationType::Type` and increase `OperationTypeCount`:

```cpp
    Clip                       = 40,
    MyOp                       = 41,
    OperationTypeCount         = 42,
```

### 1.2 Layer implementation — `sadl/layer_myop.h`

Create one header per layer. Start from a layer with a similar shape:

| Pattern | Example to copy |
|---|---|
| unary, no parameters | `layer_abs.h`, `layer_floor.h` |
| element-wise with broadcasting | `layer_add.h`, `layer_sub.h`, `layer_maximum.h` |
| one integer parameter | `layer_depth2space.h`, `layer_flatten.h` |
| several parameters | `layer_resize.h`, `layer_pad.h` |
| constant inputs (bounds, indices) | `layer_clip.h`, `layer_gather.h` |
| weights + internal quantizer | `layer_conv2d.h`, `layer_matmul.h` |

A layer derives from `Layer<T>` (`T` is `float`, `int16_t` or `int32_t`) and
implements three methods:

```cpp
#pragma once
#include "layer.h"

namespace sadl
{
namespace layers
{
template<typename T> class MyOp : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
  int m_mode = 0;   // parameters are members, read in loadInternal
  DUMP_MODEL_EXT;   // needed if the layer has parameters (see 2.2)
};

// called once per inference: compute m_out from the inputs
template<typename T> bool MyOp<T>::apply(std::vector<Tensor<T> *> &in)
{
  auto scope = makeLayerExecutionScope(m_out, in);   // memory management, mandatory
  const Tensor<T> &A = *in[0];
  m_out.quantizer    = A.quantizer;                  // output fixed-point precision
  for (int k = 0; k < m_out.size(); ++k)
  {
    typename ComputationType<T>::type z = A[k];      // wider type for integer models
    // ... compute z ...
    COUNTERS(z);    // statistics in debug builds
    SATURATE(z);    // clamp for integer types
    m_out[k] = static_cast<T>(z);
  }
  return true;
}

// called once when the model is initialized: check the inputs, size m_out
template<typename T> bool MyOp<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1)
    return false;
  m_out.resize(in[0]->dims());
  m_initDone = true;
  return true;
}

// reads the parameters, in the exact order the converter writes them
template<typename T> bool MyOp<T>::loadInternal(std::istream &file, Version)
{
  int32_t x = 0;
  file.read((char *) &x, sizeof(x));
  m_mode = x;
  return true;
}

}   // namespace layers
}   // namespace sadl
```

Things to keep in mind:

* **`init` validates**: return `false` (with a `std::cerr` message) for any
  shape or parameter you do not support, rather than computing garbage later.
* **Batch size is 1** in practice; most layers assume `N == 1`.
* **Integer models**: inputs may have different `quantizer` values. Align them
  with `ComputationType<T>::quantize` (right shift) or `shift_left`, and set
  `m_out.quantizer`. Use `ComputationType<T>::type` (a wider type) for
  intermediate results. See `layer_add.h`, `layer_clip.h` and `layer_mul.h`.
* **In-place layers**: if the output can reuse the input buffer, override
  `mutateInput()` to return `true` and `swap(*in[0], m_out)` (see `layer_clip.h`).
* **SIMD**: optional, guarded by `__AVX2__` / `__AVX512F__`; keep a scalar path.
  The tests build scalar, AVX2 and AVX-512 variants and all must pass.

### 1.3 Register the layer

| File | Change |
|---|---|
| `sadl/layers.h` | `#include "layer_myop.h"` and `DIRTYCASEPRINT(MyOp);` in `opName()` |
| `sadl/model.h` | in the layer factory (`switch (op)`): `case layers::OperationType::MyOp: return std::unique_ptr<layers::Layer<T>>(new layers::MyOp<T>{ id, op });` |

---

## 2. Auxiliary C++ tools (`sample/`)

These tools copy, dump or quantize models layer by layer. They have no default
case on purpose, so the compiler warns when a layer is missing.

### 2.1 `sample/copy.h` — model copy (used by quantization)

Add a `case` that copies the parameters read in `loadInternal`:

```cpp
  case sadl::layers::OperationType::MyOp:
    dynamic_cast<sadl::layers::MyOp<T> &>(layerQ).m_mode = dynamic_cast<const sadl::layers::MyOp<float> &>(layer).m_mode;
    break;
```

(For a layer without parameters, the `case` is just `break;`.)

### 2.2 `sample/dumper.h` — model writer (used by quantization, sparsification, version update)

If the layer has parameters, add a `dump()` that writes them **in the same
order as `loadInternal` reads them**:

```cpp
template<typename T> bool sadl::layers::MyOp<T>::dump(std::ostream &file)
{
  int32_t x = m_mode;
  file.write((const char *) &x, sizeof(x));
  return true;
}
```

Without it, tools that rewrite models (`naive_quantization`,
`update_model_version`, `sparsify_with_packing`) silently drop the parameters
and produce a file that cannot be read back. Also add `DUMP_MODEL_EXT;` to the
class declaration in `sadl/layer_myop.h` (see `sadl/layer_flatten.h`).

To check, rewrite a float model with `update_model_version`: for a layer whose
parameters are stored as read, the output must be byte-identical to the input.

### 2.3 `sample/naive_quantization.cpp`

`toQuantize()` lists, in negative logic, the layers that do **not** need a
quantizer. Add `MyOp` to this list unless the layer has its own internal
quantizer (like `Conv2D` or `MatMul`). In that case, also extend `quantize()`
to set it.

---

## 3. Optional components

These are not needed for the CPU library or the converters, but a `.sadl` file
containing the new layer cannot be loaded by them until they know it.

### 3.1 CUDA backend (`sadl_cuda/`)

`sadl_cuda` is a separate, header-only GPU backend, API-compatible with
`sadl::Model<T>` (see `sadl_cuda/README.md`). It reads the same `.sadl` files,
so it must know every layer id, but it only runs **`int16_t` models**. It has its
own copy of the layer headers: the CPU `sadl/` headers are not reused.

**1. Operation id — `sadl_cuda/layer.h`.** Add the same value as in
`sadl/layer.h` and increase `OperationTypeCount`:

```cpp
    Clip               = 40,
    MyOp               = 41,
    OperationTypeCount = 42,
```

Ids from `OperationExperimentalStart` (65535) up are used by layers that the
CUDA model creates internally by fusing others (`Conv2DBias`,
`Conv2DBiasPreLU`). Never put a layer read from files there.

**2. Layer — `sadl_cuda/layer_myop.h`.** The class has the same interface as on
the CPU (`init`, `apply`, `loadInternal`, in namespace `sadl_cuda::layers`), plus
one choice: **GPU or CPU execution**.

* **CPU layer** (the default, `isGpuLayer()` returns `false`): `apply` works on
  host memory exactly like the CPU version. The model copies device-resident
  inputs back to the host before calling it. Start from `sadl_cuda/layer_floor.h`.
  This is the simplest way to support a layer; the cost is a device → host
  transfer at this point of the graph.
* **GPU layer** (`isGpuLayer()` returns `true`): `apply` launches a CUDA kernel
  on device buffers. Start from `sadl_cuda/layer_abs.h` (element-wise) or
  `layer_clip.h` and `layer_global_averagepool.h` (more involved):

```cpp
#pragma once
#include "layer.h"
#include "cuda_utils.h"

namespace sadl_cuda
{
namespace layers
{

#ifdef __CUDACC__
__global__ static void myop_kernel(const int16_t *src, int16_t *dst, int n, int mode)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    int32_t z = src[i];
    // ... same integer computation as the CPU layer ...
    dst[i] = SADL_CUDA_SAT(z);   // saturate to int16, as SATURATE() on the CPU
  }
}
#endif

template<typename T> class MyOp : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    m_out.quantizer = in[0]->quantizer;
    const int n = static_cast<int>(m_out.size());
#ifdef __CUDACC__
    myop_kernel<<<gridDim1D(n, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
      in[0]->dev_data(), m_out.dev_data(), n, m_mode);
#else
    // host fallback when compiled without nvcc: same computation on m_out[i]
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override;           // as on the CPU
protected:
  virtual bool loadInternal(std::istream &file, Version v) override;  // reads the same bytes as the CPU
  int m_mode = 0;
};

}   // namespace layers
}   // namespace sadl_cuda
```

Rules for GPU layers:

* Use `dev_data()` on inputs and output. The model allocates the device
  buffers (`classifyTensorMemory()` in `sadl_cuda/model.h`), uploads `Const`
  tensors once, and keeps data on the device between consecutive GPU layers.
  Do not call `h2d()`/`d2h()` yourself.
* Wrap all device code in `#ifdef __CUDACC__` and keep a host fallback in
  `#else`, so the header still compiles with a host compiler.
* Results must be **bit-exact with the CPU `int16_t` layer**: same rounding,
  shifts and saturation (`SADL_CUDA_SAT`; `SATURATE_RESULT` must stay `1` in
  `sadl_cuda/options.h`). The CUDA tests compare with a tolerance of 0 by default.
* Use `CUDA_CHECK(...)` around CUDA API calls (`cuda_utils.h`) and
  `SADL_CUDA_BLOCK_SIZE` / `gridDim1D` for 1-D launches.
* A layer that only changes metadata (like `Reshape`, `Flatten`) can be a GPU
  layer without a kernel: `swapData(*in[0], m_out)` also swaps the device
  pointer, so no transfer is needed.

**3. Register the layer.** In `sadl_cuda/layers.h`, add
`#include "layer_myop.h"` and `DIRTYCASEPRINT(MyOp);`. In `createLayer()` in
`sadl_cuda/model.h`, add
`case OperationType::MyOp: return std::make_unique<MyOp<T>>(id, op);`.

**4. Document it** in the "Layer support" table of `sadl_cuda/README.md`:
`✓` (CUDA kernel), `pass-through`, or `CPU`.

**5. Test it.** `utests/check_cuda.sh` runs every int16 model of
`utests/model_int16/` with the CPU (`test_cpu_int16`) and with CUDA
(`test_cuda_int16`) and compares the outputs. This needs `nvcc`: the CUDA
targets are skipped with a warning when no CUDA compiler is found. The
`model_int16/` directory is not in the repository: create int16 models of your
test with `naive_quantization` (section 2.3), then run:

```shell
./utests/check_cuda.sh --model my_op            # tolerance 0 (bit-exact)
./utests/check_cuda.sh --model my_op --tol 1    # if a difference is justified
```

### 3.2 Netron viewer (`tools/netron/`)

`tools/netron/sadl.js` is a Netron module that displays `.sadl` files.
`tools/netron/netron-sadl.patch` adds it to Netron (tag `v8.9.0`; see
`tools/netron/README.md`). Netron stops with "No decoder for op_id=41" on a
model that contains an unknown layer, so register every new id.

**1. Declare the operation** in `sadl.OpSpec` (`sadl.js`), with the same id as in
`sadl/layer.h`:

```js
  Clip:             { id: 40, payload: false },
  MyOp:             { id: 41, payload: true  },
```

`payload` says whether the layer has bytes after its input ids, i.e. whether
`loadInternal` reads something. This includes a final quantizer (`MatMul` and
`Mul` have `payload: true` for that reason).

**2. Without payload**: nothing else to do. Layers with `payload: false` and no
decoder are registered automatically as pass-through nodes.

**3. With payload**: add a decoder next to the others (before the
`// passthrough` block). It must read **exactly** the bytes written by the
converter's serializer, in the same order, and return them as attributes:

```js
// MyOp
dec[sadl.OpCode.MyOp] = sadl.decode.decodeOp('MyOp', ({ cursor }) => {
  return [
    { name: 'mode', value: cursor.readI32() }
  ];
});
```

The `cursor` offers `readI32()`, `readI16()`, `readU8()` (booleans) and
`readBytes(n)`. For counted lists (count, then values) use
`sadl.decode.readCountedI32(cursor)`. See the `Pad`, `Gather` and `Conv2D`
decoders. If the decoder reads fewer or more bytes than the layer contains,
every following layer is decoded wrongly. Check with a model that has layers
after `MyOp`.

**4. Regenerate the patch.** `sadl.js` and the copy inside the patch must stay
identical:

```shell
git clone https://github.com/lutzroeder/netron.git && cd netron
git checkout v8.9.0
git apply /path/to/sadl/tools/netron/netron-sadl.patch
cp /path/to/sadl/tools/netron/sadl.js source/sadl.js
git add -A && git commit -m "Add SADL module for Netron"
git format-patch -1 --stdout > /path/to/sadl/tools/netron/netron-sadl.patch
```

**5. Check visually**: serve `source/` (`python3 -m http.server 8080`), open
http://127.0.0.1:8080 and load a `.sadl` model containing the new layer. The
node must show its parameters, and the layers after it must still be correct.

---

## 4. ONNX → SADL converter (`converter/onnx2sadl/`)

The converter reads an ONNX model, tracks the data layout (NCHW/NHWC), builds a
SADL graph with one `OpConverter` per ONNX operator, and writes it with one
`LayerSerializer` per SADL layer. One class lives in one file.

### 4.1 SADL id — `types/op_type.py`

```python
    Clip = 40
    MyOp = 41
```

The value must equal the C++ `OperationType::MyOp`.

### 4.2 Converter class — `ops/my_op_converter.py`

Pick the closest base class:

| Base class | For |
|---|---|
| `UnaryConverter` | one input, optional attributes via `layer_attrs()` (see `flatten_converter.py`, `depth_to_space_converter.py`) |
| `BinaryConverter` | two inputs that may be initializers, with swap rules (see `add_converter.py`, `mul_converter.py`) |
| `CompareConverter`, `PoolConverter`, `StaticWeightConverter` | the matching families |
| `OpConverter` | anything else (see `clip_converter.py`, `gather_converter.py`) |

```python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from ..errors.conversion_error import ConversionError
from ..types.op_type import OpType
from .unary_converter import UnaryConverter


class MyOpConverter(UnaryConverter):
    """ONNX MyOp -> SADL MyOp."""

    onnx_ops = ("MyOp",)          # ONNX op_type(s) handled by this class
    sadl_op = OpType.MyOp

    MODES = ["first", "second"]

    def layer_attrs(self, node):
        mode = self.attribute(node, "mode").s.decode("utf-8")
        if mode not in self.MODES:
            raise ConversionError(f"[ERROR] MyOp: unsupported mode {mode} in {node.name}")
        return {"mode": self.MODES.index(mode)}
```

For a class deriving directly from `OpConverter`, implement
`convert(self, ctx, node, inputs)`:

* `inputs` holds the names of inputs 0..2, already redirected to inserted
  layout transposes when needed; use them rather than `node.input[...]`.
* `ctx.graph.resolve(name)` gives the SADL layer producing an ONNX tensor.
* Constant inputs: `ctx.is_initializer(name)` to test,
  `ctx.extract_const(name, to_transpose, allow_force_4d=True)` to read them as a
  `ConstTensor`, and `ctx.add_const(name, tensor, node)` to add the `Const` layer.
  `ctx.model.is_static_tensor(name)` also accepts `Constant` nodes, which recent
  PyTorch exports use.
* `self.emit(ctx, node, OpType.MyOp, [inputs...], **attrs)` adds the layer
  named after `node.output[0]` and binds it. `attrs` are the parameters given to
  the serializer.
* Raise `ConversionError` for unsupported cases.

### 4.3 Layout (NCHW models)

For a PyTorch model, SADL works in NHWC while the ONNX attributes are NCHW.
Choose one of the following:

* **Remap the attributes** in `convert` (see `gather_converter.py` and
  `OpConverter.nchw_axis_to_nhwc`, or `concat_converter.py`).
* **Ask for transposes** around the layer in `annotate(self, annotator, node,
  data_layout)`, by setting the node annotation
  (`annotator.annotations[node.name]`): `transpose_before_in0 = "nchw"` and
  `to_nhwc_after_out = True` (see `reduce_mean_converter.py`). The layer then
  receives NCHW data.
* Element-wise layers usually need nothing: the layout does not change the result.

`annotate()` is optional. The default does nothing, and the layout of the
inputs is propagated.

### 4.4 Register the converter — `ops/op_catalog.py`

Import the class and add it to `OpCatalog.CONVERTERS`.

### 4.5 Parameters serializer — `writer/`

Only needed if the SADL layer has parameters (what `loadInternal` reads). The
layout must match `loadInternal` byte for byte.

* Integers only, in a fixed order: reuse `IntAttrsSerializer("mode")`, no new file.
* Lists, booleans and other layouts: write a `LayerSerializer` subclass, e.g.
  `writer/my_op_serializer.py`, using `BinaryStream`: `i32`, `i32_list` (count
  then values), `i32_values`, `bool`, `raw`. See `reduce_mean_serializer.py`
  and `pad_serializer.py`.
* If the C++ layer ends with an internal quantizer (as `Conv2D` does), set
  `writes_quantizer = True`.

Register it in `writer/layer_serializer_registry.py`:

```python
            OpType.MyOp: IntAttrsSerializer("mode"),
```

### 4.6 Optional: graph optimization (`converter/onnxopt/`)

If exporters produce a pattern SADL does not support but that can be rewritten
(fused or split ops, extra Casts...), add an `OptimizationPass` rather than
complicating the converter (see `split_slices_pass.py`). Register it in
`PassRegistry.BUILTIN`; add it to `DEFAULT_PASSES` only if it is safe for all
models. It runs with `--optimize`.

---

## 5. SADL → ONNX converter (`converter/sadl2onnx/`)

So that `converter/sadl_to_onnx.py` can read models containing the new layer.

### 5.1 Parameters reader — `reader/layer_params_reader_registry.py`

This mirrors the serializer of step 4.5:

```python
            OpType.MyOp: ListParamsReader("mode"),
```

`ListParamsReader` fields: `"name"` (int32), `"name[]"` (count + int32 list),
`"name[N]"` (N int32, e.g. `"pads[4]"`), `"name?"` (bool); `reads_quantizer=True` if the layer
ends with a quantizer. For anything else, write a `LayerParamsReader` subclass.

### 5.2 Emitter — `emitters/my_op_emitter.py`

The emitter translates the SADL layer into ONNX nodes **with SADL semantics**.
The generated ONNX graph is NHWC, like SADL.

```python
from onnx2sadl.types.op_type import OpType

from .layer_emitter import LayerEmitter


class MyOpEmitter(LayerEmitter):
    sadl_ops = (OpType.MyOp,)

    MODES = ["first", "second"]

    def emit(self, ctx, layer):
        x = ctx.value(layer.inputs[0])
        return ctx.builder.add_node("MyOp", [x], layer.name, mode=self.MODES[layer["mode"]])
```

* `ctx.value(id)` gives the ONNX value of an input (Const layers become
  initializers, dequantized to float); `ctx.const(id)` gives the numpy value of
  a Const input used as a parameter; `ctx.int64_input(id)` gives shape-like inputs.
* For ONNX ops defined on NCHW tensors, wrap them:
  `self.to_nchw(ctx, x, layer)` → op → `self.to_nhwc(ctx, y, layer)` (see
  `pool_emitter.py`). Redundant transpose pairs are removed automatically.
* Simple 1:1 mappings need no new class:
  `SimpleOpEmitter(OpType.MyOp, "MyOp")` or `NchwOpEmitter(...)`.

Register the emitter in `emitters/emitter_registry.py`. If the layer's output
rank differs from its first input's, add a rule in `emitters/rank_tracker.py`.

### 5.3 PyTorch → SADL (`converter/torch2sadl/`)

Nothing to do: PyTorch models are exported to ONNX first. Check which ONNX ops
`torch.onnx.export` produces for your PyTorch operation (opset 13 by default)
and make sure step 4 handles them, if needed through an `onnxopt` pass.

---

## 6. Tests

### 6.1 Test model — `utests/models/`

Export a minimal ONNX model that uses only the new op (plus whatever it needs).
A small Python script is fine; keep it next to the model if it is useful
(e.g. `utests/models/slice_chw_pytorch.py`):

```python
import torch

class M(torch.nn.Module):
    def forward(self, x):
        return my_op(x)

torch.onnx.export(M(), torch.rand(1, 4, 8, 8), "utests/models/my_op.onnx",
                  opset_version=13, dynamo=False)
```

Cover each supported mode or parameter, and both layouts if relevant (one
PyTorch (NCHW) and one tf2onnx (NHWC) model).

### 6.2 Round-trip test — `utests/`

```shell
cd utests && cmake -B build . && cmake --build build   # once
./utests/utest.sh utests/models/my_op.onnx                 # convert, run, compare
./utests/utest.sh utests/models/my_op.onnx --no_onnxruntime
```

`utest.sh` converts the model, runs onnxruntime for the reference outputs, then
the SADL test binaries (scalar, AVX2, AVX-512, memory pool, sparse) with a
tolerance of 0.001. If onnxruntime cannot run the model (e.g. a non-standard
use of an ONNX op), use `--no_onnxruntime` and commit the reference
`utests/results/my_op.results`.

Add the model name to the matching list in `utests/check.sh`, which runs all
the tests.

### 6.3 Converter checks

* **Regression**: converting the existing `utests/models/*.onnx` must give
  byte-identical `.sadl` files before and after your change, unless you
  deliberately changed the output. Convert all models with the old and new
  converter and compare with `cmp`.
* **Round trip through SADL → ONNX**: this checks the serializer, the reader and
  the emitter together, without building C++:

  ```python
  import sys; sys.path.insert(0, "converter")
  import numpy as np, onnxruntime as ort
  from sadl2onnx.sadl_to_onnx_converter import SadlToOnnxConverter

  SadlToOnnxConverter().convert_file("my_op.sadl", "my_op_back.onnx")
  x = np.random.rand(1, 4, 8, 8).astype(np.float32)                        # NCHW
  ref = ort.InferenceSession("utests/models/my_op.onnx").run(None, {"x": x})[0]
  sess = ort.InferenceSession("my_op_back.onnx")
  got = sess.run(None, {sess.get_inputs()[0].name: x.transpose(0, 2, 3, 1).copy()})[0]
  print(np.abs(ref - got.transpose(0, 3, 1, 2)).max())                     # NHWC -> NCHW
  ```

* **Integer model** (if relevant): quantize the float model with
  `naive_quantization` and compare with `quantization_test` (see
  `sample/README.md` and `sample/sample_test.sh`).

### 6.4 Style

```shell
flake8 converter/          # config in .flake8
black converter/
clang-format -i sadl/layer_myop.h   # config in .clang-format
```

---

## 7. Checklist

**C++**
- [ ] `sadl/layer.h`: `OperationType::MyOp`, `OperationTypeCount` + 1
- [ ] `sadl/layer_myop.h`: `init`, `apply`, `loadInternal` (float and integer types)
- [ ] `sadl/layers.h`: include + `DIRTYCASEPRINT`
- [ ] `sadl/model.h`: factory `case`
- [ ] `sample/copy.h`: `case` copying the parameters
- [ ] `sample/dumper.h`: `dump()` if the layer has parameters
- [ ] `sample/naive_quantization.cpp`: `toQuantize()` / `quantize()`

**Optional**
- [ ] CUDA: id in `sadl_cuda/layer.h`, `sadl_cuda/layer_myop.h` (GPU kernel or CPU), `sadl_cuda/layers.h`, `createLayer()` in `sadl_cuda/model.h`, table in `sadl_cuda/README.md`
- [ ] CUDA: `./utests/check_cuda.sh --model my_op` passes (bit-exact with the CPU int16 layer)
- [ ] Netron: `sadl.OpSpec` entry and, if `payload: true`, a decoder in `tools/netron/sadl.js`
- [ ] Netron: `tools/netron/netron-sadl.patch` regenerated, model checked in the viewer

**ONNX → SADL** (`converter/onnx2sadl/`)
- [ ] `types/op_type.py`: same id as in C++
- [ ] `ops/my_op_converter.py` (+ layout handling), listed in `ops/op_catalog.py`
- [ ] serializer registered in `writer/layer_serializer_registry.py` if parameters
- [ ] optional: `onnxopt` pass for exporter patterns

**SADL → ONNX** (`converter/sadl2onnx/`)
- [ ] reader in `reader/layer_params_reader_registry.py` if parameters
- [ ] emitter registered in `emitters/emitter_registry.py` (+ `rank_tracker.py` if needed)

**Tests and docs**
- [ ] `utests/models/my_op.onnx` (+ `utests/results/my_op.results` if `--no_onnxruntime`)
- [ ] model listed in `utests/check.sh`, `./utests/check.sh` passes
- [ ] existing models convert byte-identically
- [ ] SADL → ONNX round trip matches
- [ ] `flake8`, `black`, `clang-format`
- [ ] README updated if the layer adds converter options
