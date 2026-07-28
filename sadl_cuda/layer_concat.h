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
#include <cstring>

namespace sadl_cuda
{
namespace layers
{

#ifdef __CUDACC__
__global__ static void concat_kernel(
    const int16_t * __restrict__ src,
    int16_t       * __restrict__ dst,
    int src_axis, int out_axis, int inner,
    int dst_axis_offset, int shift, int total)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int e = idx % inner;
  int a = (idx / inner) % src_axis;
  int o = idx / (src_axis * inner);
  dst[o * out_axis * inner + (dst_axis_offset + a) * inner + e] =
      (int16_t)((int32_t)src[idx] >> shift);
}
#endif

template<typename T> class Concat : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() >= 3);
    const int nb_in = static_cast<int>(in.size()) - 1;

    int qmin = in[0]->quantizer;
    for (int i = 1; i < nb_in; ++i)
      if (in[i]->quantizer < qmin) qmin = in[i]->quantizer;
    m_out.quantizer = qmin;

    int inner = 1;
    for (int k = m_axis + 1; k < static_cast<int>(in[0]->dims().size()); ++k)
      inner *= m_out.dims()[k];
    int outer = 1;
    for (int k = 0; k < m_axis; ++k) outer *= m_out.dims()[k];
    const int out_axis_len = m_out.dims()[m_axis];

#ifdef __CUDACC__
    int cursor = 0;
    for (int n = 0; n < nb_in; ++n)
    {
      const int len_a = in[n]->dims()[m_axis];
      const int shift = in[n]->quantizer - qmin;
      const int total = outer * len_a * inner;
      concat_kernel<<<gridDim1D(total, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
          in[n]->dev_data(), m_out.dev_data(),
          len_a, out_axis_len, inner, cursor, shift, total);
      cursor += len_a;
    }
#else
    for (int o = 0; o < outer; ++o)
    {
      int cursor = 0;
      for (int n = 0; n < nb_in; ++n)
      {
        const int   len_a      = in[n]->dims()[m_axis];
        const int   copy_elems = len_a * inner;
        const int   in_off     = o * copy_elems;
        const int   out_off    = o * (out_axis_len * inner) + cursor * inner;
        const int   shift      = in[n]->quantizer - qmin;
        const T    *src        = in[n]->data() + in_off;
        T          *dst        = m_out.data() + out_off;
        if (shift == 0)
          std::memcpy(dst, src, copy_elems * sizeof(T));
        else
          for (int e = 0; e < copy_elems; ++e)
            dst[e] = static_cast<T>((int32_t)src[e] >> shift);
        cursor += len_a;
      }
    }
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() < 3) return false;
    const int axis_idx = static_cast<int>(in.size()) - 1;
    if ((*in[axis_idx]).size() != 1) return false;

    int axis = static_cast<int>((*in[axis_idx])[0]);
    if (axis < 0) axis += static_cast<int>(in[0]->dims().size());
    if (axis < 0 || axis >= static_cast<int>(in[0]->dims().size())) return false;
    m_axis = axis;

    int sum_dim = 0;
    for (int i = 1; i < axis_idx; ++i)
    {
      if (in[0]->dims().size() != in[i]->dims().size()) return false;
      sum_dim += in[i]->dims()[axis];
    }
    Dimensions dim = in[0]->dims();
    dim[axis] += sum_dim;
    m_out.resize(dim);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
  int m_axis = 0;
};

}  // namespace layers
} 
