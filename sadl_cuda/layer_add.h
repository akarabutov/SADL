/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#include "layer.h"
#include "cuda_utils.h"

namespace sadl_cuda
{
namespace layers
{

#ifdef __CUDACC__
__global__ static void add_kernel(
    const int16_t *a, const int16_t *b, int16_t *out, int n, int b_n,
    int shift_a, int shift_b)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int32_t va = (int32_t)a[i] >> shift_a;
  int32_t vb = (int32_t)b[(b_n == 1) ? 0 : (i % b_n)] >> shift_b;
  int32_t r  = va + vb;
  out[i] = (int16_t)SADL_CUDA_SAT(r);
}
#endif

template<typename T> class Add : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 2);
    if (in[0] == in[1]) { std::cerr << "Add: input aliasing\n"; return false; }

    int shift_a, shift_b, qfinal;
    if (in[0]->quantizer < in[1]->quantizer)
    {
      shift_a = 0;
      shift_b = in[1]->quantizer - in[0]->quantizer;
      qfinal  = in[0]->quantizer;
    }
    else
    {
      shift_a = in[0]->quantizer - in[1]->quantizer;
      shift_b = 0;
      qfinal  = in[1]->quantizer;
    }
    m_out.quantizer = qfinal;

    const int n   = static_cast<int>(m_out.size());
    const int b_n = static_cast<int>(in[1]->size());

#ifdef __CUDACC__
    add_kernel<<<gridDim1D(n, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
        in[0]->dev_data(), in[1]->dev_data(), m_out.dev_data(), n, b_n, shift_a, shift_b);
#else
    for (int i = 0; i < n; ++i)
    {
      int32_t va = (int32_t)(*in[0])[i]                           >> shift_a;
      int32_t vb = (int32_t)(*in[1])[b_n == 1 ? 0 : (i % b_n)]  >> shift_b;
      int32_t r  = va + vb;
      m_out[i]   = static_cast<T>(SADL_CUDA_SAT(r));
    }
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    // Accept: same dims, scalar, or broadcast from last dim
    if (!(in[0]->dims() == in[1]->dims()) && in[1]->size() != 1 &&
        in[1]->dims().back() != in[0]->dims().back())
      return false;
    m_out.resize(in[0]->dims());
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
};

}  
}  // namespace sadl_cuda
