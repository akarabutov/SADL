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
#include <cmath>

namespace sadl_cuda
{
namespace layers
{


#ifdef __CUDACC__
__global__ static void avgpool_kernel(
    const int16_t * __restrict__ A, int16_t * __restrict__ out,
    int N, int in_H, int in_W, int D,
    int out_H, int out_W,
    int Kh, int Kw, int s_h, int s_w,
    int divq, int q)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * out_H * out_W * D;
  if (idx >= total) return;

  int d     = idx % D;
  int tmp   = idx / D;
  int out_w = tmp % out_W;
  tmp       = tmp / out_W;
  int out_h = tmp % out_H;
  int n     = tmp / out_H;

  const int16_t *A_batch = A + n * in_H * in_W * D;

  int32_t sum = 0;
  for (int fh = 0; fh < Kh; ++fh)
  {
    int ih = out_h * s_h + fh;
    if (ih >= in_H) continue;
    for (int fw = 0; fw < Kw; ++fw)
    {
      int iw = out_w * s_w + fw;
      if (iw >= in_W) continue;
      sum += (int32_t)A_batch[(ih * in_W + iw) * D + d];
    }
  }
  int32_t r = (int32_t)(((int64_t)sum * divq) >> q);
  out[idx] = (int16_t)SADL_CUDA_SAT(r);
}
#endif

template<typename T> class AveragePool : public Layer<T>
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

    const int N     = in[0]->dims()[0];
    const int in_H  = in[0]->dims()[1];
    const int in_W  = in[0]->dims()[2];
    const int D     = in[0]->dims()[3];
    const int out_H = m_out.dims()[1];
    const int out_W = m_out.dims()[2];
    const int Kh    = m_kernel[1];
    const int Kw    = m_kernel[2];
    const int s_h   = m_strides[1];
    const int s_w   = m_strides[2];
    const int total = N * out_H * out_W * D;
    (void)total;

#ifdef __CUDACC__
    avgpool_kernel<<<gridDim1D(total, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
        in[0]->dev_data(), m_out.dev_data(),
        N, in_H, in_W, D, out_H, out_W, Kh, Kw, s_h, s_w, m_divq, m_q);
#else
    for (int n = 0; n < N; ++n)
    for (int out_h = 0; out_h < out_H; ++out_h)
      for (int out_w = 0; out_w < out_W; ++out_w)
        for (int d = 0; d < D; ++d)
        {
          int32_t sum = 0;
          for (int fh = 0; fh < Kh; ++fh)
          {
            int ih = out_h * s_h + fh;
            if (ih >= in_H) continue;
            for (int fw = 0; fw < Kw; ++fw)
            {
              int iw = out_w * s_w + fw;
              if (iw >= in_W) continue;
              sum += (int32_t)(*in[0])(n, ih, iw, d);
            }
          }
          int32_t r = (int32_t)(((int64_t)sum * m_divq) >> m_q);
          m_out(n, out_h, out_w, d) = static_cast<T>(SADL_CUDA_SAT(r));
        }
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 1 || in[0]->dims().size() != 4) return false;
    if (m_kernel.size() != 4 || m_strides.size() != 4) return false;

    constexpr int dilatation = 1;
    Dimensions    dim;
    dim.resize(4);
    dim[0] = in[0]->dims()[0];
    dim[1] = (int)std::floor((in[0]->dims()[1] + m_pads[0] + m_pads[2] -
                              ((m_kernel[1] - 1) * dilatation + 1)) / (float)m_strides[1] + 1);
    dim[2] = (int)std::floor((in[0]->dims()[2] + m_pads[1] + m_pads[3] -
                              ((m_kernel[2] - 1) * dilatation + 1)) / (float)m_strides[2] + 1);
    dim[3] = in[0]->dims()[3];
    m_out.resize(dim);

    const int cnt      = m_kernel[1] * m_kernel[2];
    const int acc_bits = (int)std::ceil(std::log2((double)cnt));
    m_q    = (int)sizeof(T) * 8 - 1 - acc_bits;
    m_divq = (1 << m_q) / cnt;

    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t x = 0;
    auto readDims = [&](Dimensions &d) -> bool
    {
      file.read(reinterpret_cast<char *>(&x), sizeof(x));
      if (x <= 0 || x > Dimensions::MaxDim) return false;
      d.resize(x);
      for (int k = 0; k < d.size(); ++k)
      {
        file.read(reinterpret_cast<char *>(&x), sizeof(x));
        d[k] = x;
      }
      return true;
    };
    return readDims(m_strides) && readDims(m_kernel) && readDims(m_pads);
  }

  Dimensions m_kernel;
  Dimensions m_strides;
  Dimensions m_pads;
  int        m_divq = 1;
  int        m_q    = 0;
};

}  // namespace layers
}  
