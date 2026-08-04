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
__global__ static void conv2d_kernel(
    const int16_t * __restrict__ A,
    const int16_t * __restrict__ K,
    int16_t       * __restrict__ out,
    int N, int in_H, int in_W, int Cin,
    int Kh,   int Kw,   int Cout, int cin_by_g,
    int out_H, int out_W,
    int s_h, int s_w, int pad_h, int pad_w,
    int groups, int shift)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * out_H * out_W * Cout;
  if (idx >= total) return;

  int out_c  = idx % Cout;
  int tmp    = idx / Cout;
  int out_w  = tmp % out_W;
  tmp        = tmp / out_W;
  int out_h  = tmp % out_H;
  int n      = tmp / out_H;

  int cout_by_g  = Cout / groups;
  int g          = out_c / cout_by_g;
  int ic_offset  = g * cin_by_g;

  const int16_t *A_batch = A + n * in_H * in_W * Cin;

  int32_t acc = 0;
  for (int kh = 0; kh < Kh; ++kh)
  {
    int ih = out_h * s_h + kh - pad_h;
    if (ih < 0 || ih >= in_H) continue;
    for (int kw = 0; kw < Kw; ++kw)
    {
      int iw = out_w * s_w + kw - pad_w;
      if (iw < 0 || iw >= in_W) continue;
      const int16_t *a_row = A_batch + (ih * in_W + iw) * Cin + ic_offset;
      const int16_t *k_row = K + ((kh * Kw + kw) * Cout + out_c) * cin_by_g;
      for (int ic = 0; ic < cin_by_g; ++ic)
        acc += (int32_t)a_row[ic] * (int32_t)k_row[ic];
    }
  }
  acc >>= shift;
  out[idx] = (int16_t)SADL_CUDA_SAT(acc);
}
#endif

template<typename T> class Conv2D : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  const Dimensions &strides() const { return m_strides; }
  const Dimensions &pads()    const { return m_pads; }
  int               groups()  const { return m_groups; }
  int               mq()      const { return m_q; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 2);
    const Tensor<T> &A      = *in[0];
    const Tensor<T> &kernel = *in[1];

    const int N        = A.dims()[0];
    const int in_H     = A.dims()[1];
    const int in_W     = A.dims()[2];
    const int Cin      = A.dims()[3];
    const int Kh       = kernel.dims()[0];
    const int Kw       = kernel.dims()[1];
    const int Cout     = kernel.dims()[2];
    const int cin_by_g = kernel.dims()[3];
    const int out_H    = m_out.dims()[1];
    const int out_W    = m_out.dims()[2];
    const int s_h      = m_strides[1];
    const int s_w      = m_strides[2];
    const int pad_h    = m_pads[0];
    const int pad_w    = m_pads[1];
    const int shift    = kernel.quantizer + m_q;
    const int total    = N * out_H * out_W * Cout;
    (void)Cin; (void)total;  

#ifdef __CUDACC__
    conv2d_kernel<<<gridDim1D(total, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
        A.dev_data(), kernel.dev_data(), m_out.dev_data(),
        N, in_H, in_W, Cin, Kh, Kw, Cout, cin_by_g,
        out_H, out_W, s_h, s_w, pad_h, pad_w, m_groups, shift);
#else
    const int cout_by_g = Cout / m_groups;
    for (int n = 0; n < N; ++n)
    for (int out_h = 0; out_h < out_H; ++out_h)
      for (int out_w = 0; out_w < out_W; ++out_w)
        for (int out_c = 0; out_c < Cout; ++out_c)
        {
          const int g         = out_c / cout_by_g;
          const int ic_offset = g * cin_by_g;
          int32_t   acc       = 0;
          for (int kh = 0; kh < Kh; ++kh)
          {
            int ih = out_h * s_h + kh - pad_h;
            if (ih < 0 || ih >= in_H) continue;
            for (int kw = 0; kw < Kw; ++kw)
            {
              int iw = out_w * s_w + kw - pad_w;
              if (iw < 0 || iw >= in_W) continue;
              for (int ic = 0; ic < cin_by_g; ++ic)
                acc += (int32_t)A(n, ih, iw, ic_offset + ic) *
                       (int32_t)kernel(kh, kw, out_c, ic);
            }
          }
          acc >>= shift;
          m_out(n, out_h, out_w, out_c) = static_cast<T>(SADL_CUDA_SAT(acc));
        }
#endif
    m_out.quantizer = A.quantizer - m_q;
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    if (in[0]->dims().size() != 4 || in[1]->dims().size() != 4) return false;

    const int N     = in[0]->dims()[0];
    const int in_H  = in[0]->dims()[1];
    const int in_W  = in[0]->dims()[2];
    const int Kh    = in[1]->dims()[0];
    const int Kw    = in[1]->dims()[1];
    const int Cout  = in[1]->dims()[2];
    const int s_h   = m_strides[1];
    const int s_w   = m_strides[2];

    int out_H, out_W;
    if (s_h == 1)
      out_H = (in_H + m_pads[0] + m_pads[2] - Kh) + 1;
    else
      out_H = (int)std::floor((float)(in_H + m_pads[0] + m_pads[2] - Kh) / s_h + 1);
    if (s_w == 1)
      out_W = (in_W + m_pads[1] + m_pads[3] - Kw) / s_w + 1;
    else
      out_W = (int)std::floor((float)(in_W + m_pads[1] + m_pads[3] - Kw) / s_w + 1);

    Dimensions dim;
    dim.resize(4);
    dim[0] = N;
    dim[1] = out_H;
    dim[2] = out_W;
    dim[3] = Cout;
    m_out.resize(dim);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version v) override
  {
    int32_t x = 0;
    file.read(reinterpret_cast<char *>(&x), sizeof(x));
    if (x <= 0 || x > Dimensions::MaxDim) return false;
    m_strides.resize(x);
    for (int k = 0; k < m_strides.size(); ++k)
    {
      file.read(reinterpret_cast<char *>(&x), sizeof(x));
      m_strides[k] = x;
    }
    if (m_strides.size() == 2)
      m_strides = Dimensions({1, m_strides[0], m_strides[1], 1});
    if (m_strides.size() != 4) return false;

    file.read(reinterpret_cast<char *>(&x), sizeof(x));
    if (x <= 0 || x > Dimensions::MaxDim) return false;
    m_pads.resize(x);
    for (int k = 0; k < m_pads.size(); ++k)
    {
      file.read(reinterpret_cast<char *>(&x), sizeof(x));
      m_pads[k] = x;
    }

    if ((int)v > 2)
      file.read(reinterpret_cast<char *>(&m_groups), sizeof(m_groups));

    file.read(reinterpret_cast<char *>(&m_q), sizeof(m_q));
    return true;
  }

  Dimensions m_strides;
  Dimensions m_pads;
  int        m_q      = 0;
  int        m_groups = 1;
};

}  // namespace layers
}  // namespace sadl_cuda
