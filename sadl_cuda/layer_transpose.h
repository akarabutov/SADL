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
#include <array>

namespace sadl_cuda
{
namespace layers
{


#ifdef __CUDACC__
struct TransposePerm
{
  int dims_out[6];
  int mapped_strides[6];
  int nd;
};

__global__ static void transpose_kernel(
    const int16_t * __restrict__ A,
    int16_t       * __restrict__ out,
    int total, TransposePerm p)
{
  int flat_out = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_out >= total) return;

  int flat_in = 0, tmp = flat_out;
  for (int k = p.nd - 1; k >= 0; --k)
  {
    flat_in += (tmp % p.dims_out[k]) * p.mapped_strides[k];
    tmp /= p.dims_out[k];
  }

  out[flat_out] = A[flat_in];
}
#endif

template<typename T> class Transpose : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    const Tensor<T> &A  = *in[0];
    m_out.quantizer = A.quantizer;

    const int nd    = static_cast<int>(m_out.dims().size());
    const int total = static_cast<int>(m_out.size());

#ifdef __CUDACC__
    int strides[6] = {};
    strides[nd - 1] = 1;
    for (int k = nd - 2; k >= 0; --k)
      strides[k] = strides[k + 1] * A.dims()[k + 1];

    TransposePerm p = {};
    p.nd = nd;
    for (int k = 0; k < nd; ++k)
    {
      p.dims_out[k]      = m_out.dims()[k];
      p.mapped_strides[k] = strides[m_perm[k]];
    }
    transpose_kernel<<<gridDim1D(total, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
        A.dev_data(), m_out.dev_data(), total, p);
#else
    const Dimensions &Ad = A.dims();
    const Dimensions &d  = m_out.dims();
    (void)total;
    if (nd == 4)
    {
      std::array<int, 4> idx;
      std::array<int *, 4> mapped;
      for (int k = 0; k < 4; ++k) mapped[k] = &idx[m_perm[k]];
      for (idx[0] = 0; idx[0] < Ad[0]; ++idx[0])
        for (idx[1] = 0; idx[1] < Ad[1]; ++idx[1])
          for (idx[2] = 0; idx[2] < Ad[2]; ++idx[2])
            for (idx[3] = 0; idx[3] < Ad[3]; ++idx[3])
            {
              auto oa = Ad[3] * (Ad[2] * (Ad[1] * idx[0] + idx[1]) + idx[2]) + idx[3];
              auto ob = d[3]  * (d[2]  * (d[1]  * *mapped[0] + *mapped[1]) + *mapped[2]) + *mapped[3];
              m_out[ob] = A[oa];
            }
    }
    else
    {
      const int n = static_cast<int>(A.size());
      for (int flat = 0; flat < n; ++flat)
      {
        int tmp = flat;
        std::array<int, Dimensions::MaxDim> idx_src{};
        for (int k = nd - 1; k >= 0; --k)
        {
          idx_src[k] = tmp % Ad[k];
          tmp /= Ad[k];
        }
        int ob = 0;
        for (int k = 0; k < nd; ++k)
          ob = ob * d[k] + idx_src[m_perm[k]];
        m_out[ob] = A[flat];
      }
    }
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2 || in[1]->dims().size() != 1) return false;
    Dimensions dim;
    dim.resize(static_cast<int>(in[1]->size()));
    for (int k = 0; k < static_cast<int>(in[1]->size()); ++k)
    {
      int p = static_cast<int>((*in[1])(k));
      if (p == -1) p = k;
      dim[k]    = in[0]->dims()[p];
      m_perm[k] = p;
    }
    m_out.resize(dim);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
  std::array<int, Dimensions::MaxDim> m_perm{};
};

}  // namespace layers
}  // namespace sadl_cuda
