# SADL — Library API Reference

This document describes the public C++ API of the SADL header-only inference library.
For build instructions, conversion tools, and usage examples see the top-level `README.md`.

## Contents

- [Overview](#overview)
- [Including the library](#including-the-library)
- [Build options — `options.h`](#build-options--optionsh)
- [`Model<T>` — main inference class](#modelt--main-inference-class)
- [`Tensor<T>` — data container](#tensort--data-container)
- [`Dimensions` — shape descriptor](#dimensions--shape-descriptor)
- [`Layer<T>` — layer base class (advanced)](#layert--layer-base-class-advanced)
- [Supported operations](#supported-operations)
- [Data types](#data-types)
- [Typical usage pattern](#typical-usage-pattern)
- [Querying model type before instantiation](#querying-model-type-before-instantiation)

---

## Overview

SADL is a header-only C++ library with no runtime dependencies.
The single entry point is `sadl/model.h`; including it pulls in every other header transitively.

```
sadl/
  model.h          ← include this
  tensor.h
  layer.h
  layers.h
  dimensions.h
  options.h        ← compile-time configuration
  simd_utils.h
  layer_*.h        ← individual layer implementations
```

---

## Including the library

```cpp
#include <sadl/model.h>   // everything needed for inference
```

No compilation of the library itself is required — only the consumer project needs to be compiled.

---

## Build options — `options.h`

All options are controlled by preprocessor macros or compiler flags before the library headers are included.

### Behaviour

| Macro | Default | Description |
|-------|---------|-------------|
| `SATURATE_RESULT` | `1` | Clamp integer outputs to the representable range instead of wrapping on overflow. Set to `0` to disable. |
| `SPARSE_SUPPORT` | undefined | Enable sparse MatMul acceleration. When defined, two thresholds become available (see below). |
| `NDEBUG` | undefined | Disable internal sanity assertions (improves performance). |

When `SPARSE_SUPPORT` is defined the following global constants are declared:

```cpp
static constexpr float kSparsifyThreshold     = 0.8f;   // min fraction of zeros to trigger sparsification
static constexpr float kSparsifySizeThreshold = 1000.f; // min matrix element count to consider sparsification
```

Both can be overridden at runtime through the corresponding public fields of `Model<T>` (see below).

### SIMD level

Controlled entirely by compiler flags — there is no runtime selection:

| Compiler flags | SIMD level |
|---------------|-----------|
| *(none)* or `-msse4.2` | No SIMD |
| `-mavx2` | AVX2 |
| `-mavx2 -mfma` | AVX2 + fused multiply-add |
| `-mavx512bw -mavx512f -mavx512dq` | AVX-512 |

### Debug / diagnostic macros

All are undefined by default. Enable by defining before the first `#include` or via `-D` on the compiler command line.

| Macro | Effect |
|-------|--------|
| `DEBUG_VALUES` | Print intermediate tensor values during inference |
| `DEBUG_MODEL` | Print layer information during load and init |
| `DEBUG_COUNTERS` | Count overflows and MAC operations; unlocks `Model::resetCounters()` and `Model::printOverflow()` |
| `DEBUG_PRINT` | Print model metadata and deprecation warnings |
| `DEBUG_SIMD` | Warn when a layer falls back to a non-SIMD code path |
| `DEBUG_KEEP_OUTPUT` | Retain a copy of each layer's output tensor after inference |
| `DEBUG_OVERFLOW` | Force all accumulators to `int16_t` to expose accumulator overflow |
| `DEBUG_MODEL_ANALYZE` | Print hardware-friendliness analysis for each layer |
| `DEBUG_PATH` | Print which code path is taken for each operation |

### File format versions

```cpp
namespace sadl {
  enum class Version {
    unknown  = -1,
    sadl01   =  1,  // deprecated — no longer loadable
    sadl02   =  2,
    sadl03   =  3,
    sadl04   =  4,
    sadl05   =  5,  // current (sadl_new is an alias)
    sadl_new =  5,
  };
}
```

---

## `Model<T>` — main inference class

```cpp
#include <sadl/model.h>
namespace sadl {
  template<typename T> class Model { ... };
}
```

`T` must be one of `float`, `int32_t`, or `int16_t`.  
The type must match the type encoded in the `.sadl` file (a mismatch is detected at load time).

### Nested types

```cpp
struct LayerData {
  std::unique_ptr<layers::Layer<T>> layer;   // owned layer object
  std::vector<Tensor<T>*>           inputs;  // non-owning pointers to input tensors
};
```

```cpp
// Only available when DEBUG_COUNTERS is defined
struct Stat {
  uint64_t overflow     = 0;  // output values that saturated
  uint64_t overflow_acc = 0;  // accumulator values that saturated
  uint64_t op           = 0;  // total operations
  uint64_t mac          = 0;  // multiply-accumulate operations
  uint64_t mac_nz       = 0;  // non-zero multiply-accumulate operations
};
```

### Constants

```cpp
static constexpr int kMaxInputByLayer = 2;     // max fan-in per layer
static constexpr int kMaxLayers       = 8192;  // max layers per model
```

### Core lifecycle methods

```cpp
bool load(std::istream& in);
```
Load a model from a binary stream.  
Reads the magic header, validates the data type, and deserialises all layers.  
Returns `false` on any error; error details are written to `std::cerr`.  
Calling `load` resets any previously loaded model.

```cpp
bool init(std::vector<Tensor<T>>& in);
```
Initialise the model with a set of input tensors.  
`in` provides the shapes (and optionally values) of the model's inputs — call `getInputsTemplate()` to obtain correctly shaped zero-filled tensors.  
Internally this inserts copy layers where needed, reshapes convolution filters for optimal memory layout, and propagates shapes through every layer.  
Must be called once after `load`, and again if the input spatial dimensions change.  
Returns `false` on any error.

```cpp
bool apply(std::vector<Tensor<T>>& in);
```
Run inference.  
`in` must contain the same number of tensors and the same shapes as those passed to `init`.  
Overwrites the inputs of placeholder layers with the values from `in`, then executes every layer in topological order.  
Returns `false` on any error.  
After a successful call, outputs are available through `result()`.

### Output access

```cpp
const Tensor<T>& result(int idx_out = 0) const;
```
Return the output tensor at index `idx_out` (zero-based).  
Valid after a successful call to `apply`.

```cpp
size_t nbOutputs() const;
```
Number of model outputs.

```cpp
const std::vector<layers::Layer<T>::Id>& getIdsOutput() const;
```
Layer IDs corresponding to the model outputs (same order as `result()`).

### Selective output activation

```cpp
void activateOutput(const std::vector<char>& output_activated);
```
Disable computation of outputs that are not needed.  
`output_activated` must have the same size as `nbOutputs()`.  
A value of `0` (false) skips the corresponding output and all layers that feed only into it.  
Useful to avoid computing expensive outputs when only a subset is required.

### Introspection

```cpp
std::vector<Tensor<T>> getInputsTemplate() const;
```
Return zero-filled tensors with the shapes expected by the model's input placeholders.  
Pass the result directly to `init`, then fill the tensor data before each call to `apply`.

```cpp
std::vector<layers::Layer<T>::Id> getLayersId() const;
```
Return the IDs of all layers in the model.

```cpp
const LayerData& getLayer(const layers::Layer<T>::Id& id) const;
LayerData&       getLayer(const layers::Layer<T>::Id& id);
```
Access a layer and its input wiring by ID.

```cpp
Version version() const;
```
File format version of the loaded model.

```cpp
const std::string& info() const;
```
Optional info string embedded in the model file (available for SADL v5 files only; empty string otherwise).

### Sparse support *(requires `SPARSE_SUPPORT`)*

```cpp
bool  compute_sparsity(std::vector<Tensor<T>>& in);
float sparsity_threshold;       // default: kSparsifyThreshold (0.8)
float sparsity_size_threshold;  // default: kSparsifySizeThreshold (1000)
```
Analyse weight matrices and convert those that exceed the sparsity threshold to a packed sparse representation.  
Must be called after `init` and before the first `apply`.  
The two public fields can be adjusted before the call to tune sensitivity.

### Debug counters *(requires `DEBUG_COUNTERS`)*

```cpp
void resetCounters();
Stat printOverflow(bool printinfo = false) const;
```
`resetCounters` zeroes all per-layer counters.  
`printOverflow` aggregates counters across all layers and returns a `Stat` struct; if `printinfo` is `true` it also prints a summary to `std::cout`.

---

## `Tensor<T>` — data container

```cpp
#include <sadl/tensor.h>   // included transitively via model.h
namespace sadl {
  template<typename T> class Tensor { ... };
}
```

`Tensor` is a multi-dimensional array backed by a 64-byte aligned allocation (for SIMD).

### Key members

```cpp
int quantizer = 0;   // fixed-point scale factor (log2); 0 for float models
```

### Construction and resizing

```cpp
Tensor();                    // empty tensor
explicit Tensor(Dimensions); // allocate with given shape
void resize(Dimensions);     // reallocate (data undefined afterwards)
```

### Element access

```cpp
// Linear (flat) access — fastest, no bounds derivation
T& operator[](int i);
T  operator[](int i) const;

// Multi-index access (1-D to 4-D)
T& operator()(int i);
T& operator()(int i, int j);
T& operator()(int i, int j, int k);
T& operator()(int i, int j, int k, int l);

// Pointer to element for SIMD loads
const T* addr(int i, int j, int k, int l) const;

// Range checks
bool in(int i) const;
bool in(int i, int j) const;
bool in(int i, int j, int k) const;
bool in(int i, int j, int k, int l) const;
```

### Metadata

```cpp
const Dimensions& dims() const;   // shape
int64_t           size() const;   // total element count
```

### Raw data access

```cpp
T*       data();        // pointer to first element
const T* data() const;
```

### Iteration (dense tensors only)

```cpp
iterator       begin();
const_iterator begin() const;
iterator       end();
const_iterator end() const;
```

### Utilities

```cpp
void fill(T value);   // fill all elements with a constant
```

### Sparse support *(requires `SPARSE_SUPPORT`)*

```cpp
bool isSparse() const;
bool isDenseDataAvailable() const;
bool isSparsePack8()  const;    // packed with block size 8
bool isSparsePack16() const;    // packed with block size 16

std::vector<T>&        getDataSparse();
const std::vector<T>&  getDataSparse() const;
const std::vector<index>& getIndices() const;
const std::vector<uint16_t>& getNbNonzerosCol() const;

void redensifySparseData();   // convert back to dense in-place

// Free functions
void sparsify(Tensor<T>& weights);
void sparsifyWithPacking(Tensor<T>& weights, int packingSize);
int  computePackedSparsity(Tensor<T>& weights);
```

---

## `Dimensions` — shape descriptor

```cpp
#include <sadl/dimensions.h>   // included transitively via model.h
namespace sadl {
  struct Dimensions { ... };
}
```

Stores up to 6 dimensions as a fixed-size array.

```cpp
static constexpr int MaxDim = 6;

// Construction
Dimensions();                                  // all zeros, size 0
Dimensions(std::initializer_list<int>);        // e.g. Dimensions{1, 16, 16, 3}

// Element access
int& operator[](int i);
int  operator[](int i) const;

// Iterators
int* begin();  int* end();
const int* begin() const;  const int* end() const;

// Metadata
int     size() const;          // number of dimensions (rank)
int64_t nbElements() const;    // product of all dimensions
int     back() const;          // last (innermost) dimension

bool operator==(const Dimensions&) const;
```

Tensor layout convention: `(batch, height, width, channels)` for 4-D feature maps (NHWC).

---

## `Layer<T>` — layer base class (advanced)

```cpp
#include <sadl/layer.h>   // included transitively via model.h
namespace sadl::layers {
  template<typename T> class Layer { ... };
}
```

The base class is exposed mainly to allow inspection of a loaded model.  
It is not normally subclassed by library consumers.

```cpp
using Id = int32_t;   // layer identifier

// Accessors
Tensor<T>&          output();         // output tensor (valid after init)
const std::string&  name()  const;    // layer name from the model file
Id                  id()    const;    // unique integer ID
const std::vector<Id>& inputsId() const;  // IDs of input layers
OperationType::Type op()    const;    // operation type enum value
bool                initDone() const; // true after successful init
```

---

## Supported operations

The following `OperationType::Type` values are supported:

| Category | Operations |
|----------|-----------|
| Data | `Placeholder`, `Const`, `Copy`, `Identity` |
| Linear | `MatMul`, `BiasAdd` |
| Convolution | `Conv2D`, `Conv2DTranspose` |
| Pooling | `MaxPool`, `AveragePool`, `GlobalAveragePool` |
| Activation | `Relu`, `LeakyRelu`, `PReLU`, `Clip` |
| Element-wise | `Add`, `Sub`, `Mul`, `Abs`, `Maximum`, `Minimum`, `Floor` |
| Shape | `Reshape`, `Flatten`, `Transpose`, `Expand`, `Slice`, `Tile`, `Pad`, `Depth2Space` |
| Indexing | `Gather`, `GatherElements`, `ScatterND` |
| Reduction | `ReduceMean` |
| Spatial | `Resize`, `GridSample` |
| Logic | `Compare`, `Where` |
| Integer | `BitShift`, `Shape` |

Use `layers::opName(OperationType::Type)` to convert an enum value to its string name.

---

## Data types

```cpp
namespace sadl::layers {
  struct TensorInternalType {
    enum Type {
      Int32 = 0,
      Float = 1,
      Int16 = 2,
    };
  };
}
```

The type is baked into the `.sadl` file at conversion time.  
`Model<float>` loads `Float` files, `Model<int32_t>` loads `Int32` files, `Model<int16_t>` loads `Int16` files.

To detect the type stored in a file without instantiating a full model, read bytes 7–8 (the version digits) then the next byte as a `TensorInternalType::Type`. The `getModelType()` helper in `sample/helper.h` provides a ready-made implementation:

```cpp
#include "helper.h"  // from sample/
sadl::layers::TensorInternalType::Type t = getModelType("model.sadl");
```

---

## Typical usage pattern

```cpp
#include <sadl/model.h>
#include <fstream>
#include <iostream>
using namespace std;

int main() {
  // 1. Load
  sadl::Model<float> model;
  ifstream file("model.sadl", ios::binary);
  if (!model.load(file)) {
    cerr << "[ERROR] cannot load model\n";
    return 1;
  }

  // 2. Allocate inputs with correct shapes
  vector<sadl::Tensor<float>> inputs = model.getInputsTemplate();

  // 3. (Optional) override spatial size before init
  // auto d = inputs[0].dims();
  // d[1] = 64; d[2] = 64;
  // inputs[0].resize(d);

  // 4. Initialise
  if (!model.init(inputs)) {
    cerr << "[ERROR] init failed\n";
    return 1;
  }

  // 5. Fill inputs (repeat from here for each inference call)
  float v = 0.f;
  for (auto& t : inputs)
    for (auto& x : t)
      x = v++;

  // 6. Run inference
  if (!model.apply(inputs)) {
    cerr << "[ERROR] inference failed\n";
    return 1;
  }

  // 7. Read outputs
  for (size_t i = 0; i < model.nbOutputs(); ++i)
    cout << "output " << i << ":\n" << model.result(i) << "\n";
}
```

### Multi-type dispatch

When the model type is not known at compile time, probe the file then dispatch:

```cpp
switch (getModelType("model.sadl")) {
  case sadl::layers::TensorInternalType::Float: infer<float>("model.sadl");   break;
  case sadl::layers::TensorInternalType::Int32: infer<int32_t>("model.sadl"); break;
  case sadl::layers::TensorInternalType::Int16: infer<int16_t>("model.sadl"); break;
}
```

### Selective output activation

```cpp
// model has 2 outputs; only compute the first one
model.activateOutput({1, 0});
model.apply(inputs);
const auto& out0 = model.result(0);   // valid
// model.result(1) is not computed
```
