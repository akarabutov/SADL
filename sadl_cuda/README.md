# sadl_cuda — SADL CUDA Backend

`sadl_cuda` is a GPU inference backend for the [SADL](../README.md) library.
It exposes an API that is intentionally API-compatible with the CPU `sadl::Model<T>` but runs
computationally heavy layers as CUDA kernels.

**Constraint:** only `int16_t` models are supported.

---

## Requirements

| Requirement | Minimum version |
|---|---|
| CUDA Toolkit | 11.0 |
| C++ standard | C++17 |
| CMake | 3.18 |
| GPU compute capability | 6.0 (Pascal) or higher |

---

## Design

`sadl_cuda` is a **header-only** library.  All layer logic lives in `.h` files
that are compiled by `nvcc`.  Any translation unit that includes `sadl_cuda/model.h`
must therefore be compiled as a CUDA file (`.cu` extension, or
`set_source_files_properties(… LANGUAGE CUDA)` in CMake).

The library mirrors the CPU `sadl` namespace structure:

| CPU (`sadl`) | CUDA (`sadl_cuda`) |
|---|---|
| `sadl::Model<T>` | `sadl_cuda::Model<T>` |
| `sadl::Tensor<T>` | `sadl_cuda::Tensor<T>` |
| `sadl::Dimensions` | `sadl_cuda::Dimensions` |

### Memory model

Each `Tensor` keeps a host buffer and can additionally own a discrete **device
buffer** (`activateGpu()`), tracking with an `on_device` flag which copy is
current.  At `init()`, `classifyTensorMemory()`:

- activates a device buffer for every GPU-layer output;
- walks the graph backwards and activates a device buffer for **every tensor
  that feeds a GPU layer** (so a CPU-produced input to a GPU layer gets one too);
- pre-uploads all `Const` (weight) tensors to the device once.

During `apply()`, data therefore **stays resident on the device across
consecutive GPU layers**.  A host↔device copy (`h2d`/`d2h`) is issued only at a
GPU↔CPU boundary — i.e. when a GPU layer consumes a host-only tensor, or a
CPU layer consumes a device-only tensor.  In practice only the model inputs are
uploaded at the start and the outputs are read back at the end.

`Reshape` and `Flatten` are **metadata-only pass-throughs**: their `apply()`
does a pointer swap (`swapData`, which also swaps the device pointer) and they
report `isGpuLayer() == true`, so they never force a transfer regardless of
where their data currently lives.

---

## Build

```cmake
# standalone build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

To integrate into an existing CMake project:

```cmake
add_subdirectory(sadl_cuda)
target_link_libraries(my_target PRIVATE sadl_cuda)
# the consuming target must be compiled by nvcc (.cu source or LANGUAGE CUDA)
```

---

## Usage

```cpp
#include <sadl_cuda/model.h>
#include <fstream>
#include <iostream>

int main() {
  sadl_cuda::Model<int16_t> model;

  std::ifstream file("model.sadl", std::ios::binary);
  if (!model.load(file)) {
    std::cerr << "[ERROR] Unable to read model\n";
    return 1;
  }

  std::vector<sadl_cuda::Tensor<int16_t>> inputs = model.getInputsTemplate();
  if (!model.init(inputs)) {
    std::cerr << "[ERROR] Initialization failed\n";
    return 1;
  }

  // fill inputs ...
  for (auto &t : inputs)
    for (auto &x : t)
      x = 0;

  if (!model.apply(inputs)) {
    std::cerr << "[ERROR] Inference failed\n";
    return 1;
  }

  const sadl_cuda::Tensor<int16_t> &out = model.result(0);
  std::cout << "output size: " << out.size() << "\n";
  return 0;
}
```

A ready-to-use sample program is provided in [`../sample/sample_cuda.cu`](../sample/sample_cuda.cu).
It loads a model, runs one warm-up pass and one timed pass using CUDA events:

```shell
./sample_cuda model.sadl [override_spatial_size]
```

---

## Key internals

| Symbol | File | Description |
|---|---|---|
| `SADL_CUDA_BLOCK_SIZE` | `options.h` | Default 1-D thread block size (256) |
| `SATURATE_RESULT` | `options.h` | Must be `1` — int16 accumulators saturate; **required for bit-exactness** with the CPU reference |
| `SADL_CUDA_SAT(x)` | `options.h` | Saturates `int32` result to `int16` range |
| `CUDA_CHECK(call)` | `cuda_utils.h` | Aborts on CUDA API error |
| `gridDim1D(n, bs)` | `cuda_utils.h` | Computes 1-D grid size for `n` elements |
| `Model::classifyTensorMemory()` | `model.h` | Decides which tensors get device buffers (see Memory model) |
| `Tensor::activateGpu/h2d/d2h` | `tensor.h` | Allocate device buffer / copy host→device / device→host |

---

## Layer support

The table below lists the supported layers and how each executes:

- **✓** — runs a CUDA kernel on the device.
- **pass-through** — GPU layer with no kernel; metadata-only, data stays where it is (no transfer).
- **CPU** — runs on the host; forces a device→host copy of any device-resident input.

| Layer | Execution | Notes |
|---|---|---|
| Abs | ✓ | element-wise |
| Add | ✓ | element-wise, scalar broadcast |
| AveragePool | ✓ | fixed-point reciprocal matching CPU |
| BiasAdd | ✓ | fused bias broadcast |
| Clip | ✓ | 3-input form (activation, min, max tensors) |
| Concat | ✓ | |
| Conv2D | ✓ | reshaped kernel, groups, arbitrary strides/pads |
| Conv2DTranspose | ✓ | gather-based, arbitrary strides |
| GlobalAveragePool | ✓ | block-per-(n,c) parallel reduction; two-pass fixed-point when needed |
| MatMul | ✓ | 2-D and batched |
| Maximum | ✓ | element-wise or scalar broadcast |
| MaxPool | ✓ | arbitrary kernel/stride/dilation |
| Minimum | ✓ | element-wise or scalar broadcast |
| Mul | ✓ | element-wise, fixed-point shift |
| PReLU | ✓ | per-channel or scalar slope |
| Relu | ✓ | element-wise |
| Resize | ✓ | bilinear / bicubic / nearest |
| Slice | ✓ | |
| Sub | ✓ | element-wise, scalar broadcast |
| Transpose | ✓ | |
| Flatten | pass-through | view only; `swapData`, stays on device |
| Reshape | pass-through | view only; `swapData`, stays on device |
| BitShift | CPU | |
| Compare | CPU | LessThan / GreaterThan / EqualTo |
| Const | CPU | constant tensor, pre-uploaded to device at init |
| Copy | CPU | internal bookkeeping layer |
| Depth2Space | CPU | |
| Expand | CPU | |
| Floor | CPU | |
| Gather | CPU | |
| GatherElements | CPU | |
| GridSample | CPU | bilinear / bicubic / nearest |
| Identity | CPU | pass-through |
| LeakyRelu | CPU | |
| Pad | CPU | constant / edge / reflect |
| Placeholder | CPU | input injection |
| ReduceMean | CPU | fixed-point reciprocal, keepdims required |
| ScatterND | CPU | |
| Shape | CPU | |
| Tile | CPU | |
| Where | CPU | |

---

## License

BSD-3-Clause — see [../README.md](../README.md).
