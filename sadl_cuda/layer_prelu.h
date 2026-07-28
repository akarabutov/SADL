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
__global__ static void prelu_kernel_single(
    const int16_t *src, int16_t *dst, int n, int32_t alpha, int alpha_q)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int16_t v = src[i];
  if (v < 0)
  {
    int32_t neg = ((int32_t)v * alpha) >> alpha_q;
    neg         = max((int32_t)-32768, min((int32_t)32767, neg));
    dst[i]      = (int16_t)neg;
  }
  else
  {
    dst[i] = v;
  }
}

__global__ static void prelu_kernel_multi4d(
    const int16_t *src, int16_t *dst, int N, int H, int W, int C,
    const int16_t *alpha, int alpha_size, int alpha_q)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * H * W * C;
  if (idx >= total) return;
  int c     = idx % C;
  int16_t v = src[idx];
  if (v < 0)
  {
    // alpha_size==1 => shared alpha, else per-channel
    int32_t a   = (int32_t)alpha[(alpha_size == 1) ? 0 : c];
    int32_t neg = ((int32_t)v * a) >> alpha_q;
    neg         = max((int32_t)-32768, min((int32_t)32767, neg));
    dst[idx]    = (int16_t)neg;
  }
  else
  {
    dst[idx] = v;
  }
}
#endif

template<typename T> class PReLU : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 2);
    const Tensor<T> &alpha    = *in[1];
    const int        alpha_q  = alpha.quantizer;
    m_out.quantizer = in[0]->quantizer;
    const int n = static_cast<int>(m_out.size());

#ifdef __CUDACC__
    if (m_out.dims().size() == 4 && alpha.size() >= 1)
    {
      const int N = m_out.dims()[0], H = m_out.dims()[1];
      const int W = m_out.dims()[2], C = m_out.dims()[3];
      int blocks  = gridDim1D(n, SADL_CUDA_BLOCK_SIZE);
      prelu_kernel_multi4d<<<blocks, SADL_CUDA_BLOCK_SIZE>>>(
          in[0]->dev_data(), m_out.dev_data(), N, H, W, C,
          alpha.dev_data(), static_cast<int>(alpha.size()), alpha_q);
    }
    else
    {
      int32_t a = static_cast<int32_t>(alpha[0]);
      int blocks = gridDim1D(n, SADL_CUDA_BLOCK_SIZE);
      prelu_kernel_single<<<blocks, SADL_CUDA_BLOCK_SIZE>>>(
          in[0]->dev_data(), m_out.dev_data(), n, a, alpha_q);
    }
#else
    std::copy(in[0]->begin(), in[0]->end(), m_out.begin());
    if (alpha.size() == 1)
    {
      const int32_t a = static_cast<int32_t>(alpha[0]);
      for (int i = 0; i < n; ++i)
      {
        T v = m_out[i];
        if (v < 0)
        {
          int32_t neg = ((int32_t)v * a) >> alpha_q;
          neg         = (neg < -32768) ? -32768 : (neg > 32767 ? 32767 : neg);
          m_out[i]    = static_cast<T>(neg);
        }
      }
    }
    else if (m_out.dims().size() == 4)
    {
      const int N = m_out.dims()[0], H = m_out.dims()[1];
      const int W = m_out.dims()[2], C = m_out.dims()[3];
      const bool shared = (alpha.size() == 1 || (alpha.dims().size() == 3 && alpha.dims()[0] == 1));
      for (int nb = 0; nb < N; ++nb)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
            for (int c = 0; c < C; ++c)
            {
              T v = m_out(nb, h, w, c);
              if (v < 0)
              {
                int32_t a   = shared ? (int32_t)alpha[0] : (int32_t)alpha[c];
                int32_t neg = ((int32_t)v * a) >> alpha_q;
                neg         = (neg < -32768) ? -32768 : (neg > 32767 ? 32767 : neg);
                m_out(nb, h, w, c) = static_cast<T>(neg);
              }
            }
    }
    else
    {
      const int last = m_out.dims().back();
      for (int i = 0; i < n; ++i)
      {
        T v = m_out[i];
        if (v < 0)
        {
          int32_t a   = static_cast<int32_t>(alpha[i % last]);
          int32_t neg = ((int32_t)v * a) >> alpha_q;
          neg         = (neg < -32768) ? -32768 : (neg > 32767 ? 32767 : neg);
          m_out[i]    = static_cast<T>(neg);
        }
      }
    }
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    m_out.resize(in[0]->dims());
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
};

template<typename T> using LeakyRelu = PReLU<T>;

}  
}  
