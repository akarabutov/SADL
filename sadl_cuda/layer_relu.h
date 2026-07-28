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
__global__ static void relu_kernel(const int16_t *src, int16_t *dst, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = (src[i] < 0) ? (int16_t)0 : src[i];
}
#endif

template<typename T> class Relu : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 1);
    m_out.quantizer = in[0]->quantizer;
    const int n = static_cast<int>(m_out.size());
#ifdef __CUDACC__
    int blocks = gridDim1D(n, SADL_CUDA_BLOCK_SIZE);
    relu_kernel<<<blocks, SADL_CUDA_BLOCK_SIZE>>>(in[0]->dev_data(), m_out.dev_data(), n);
#else
    for (int i = 0; i < n; ++i)
      m_out[i] = ((*in[0])[i] < 0) ? T{} : (*in[0])[i];
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 1) return false;
    m_out.resize(in[0]->dims());
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
};

} 
}  
