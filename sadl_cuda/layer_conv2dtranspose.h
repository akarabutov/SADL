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
__global__ static void conv2dtranspose_kernel(
    const int16_t * __restrict__ A,
    const int16_t * __restrict__ K,
    int16_t       * __restrict__ out,
    int N, int in_H, int in_W, int Cin,
    int Kh,   int Kw,   int Cout,
    int out_H, int out_W,
    int s_h,  int s_w,
    int half_h, int half_w,
    int shift)
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

  const int16_t *A_batch = A + n * in_H * in_W * Cin;

  int32_t acc = 0;
  for (int ki = 0; ki < Kh; ++ki)
  {
    int nom_h = out_h + half_h - ki;
    if (nom_h < 0 || nom_h % s_h != 0) continue;
    int i1 = nom_h / s_h;
    if (i1 >= in_H) continue;
    for (int kj = 0; kj < Kw; ++kj)
    {
      int nom_w = out_w + half_w - kj;
      if (nom_w < 0 || nom_w % s_w != 0) continue;
      int j1 = nom_w / s_w;
      if (j1 >= in_W) continue;
      const int16_t *a_row = A_batch + (i1 * in_W + j1) * Cin;
      const int16_t *k_row = K + ((ki * Kw + kj) * Cout + out_c) * Cin;
      for (int ic = 0; ic < Cin; ++ic)
        acc += (int32_t)a_row[ic] * (int32_t)k_row[ic];
    }
  }
  acc >>= shift;
  out[idx] = (int16_t)SADL_CUDA_SAT(acc);
}
#endif

template<typename T> class Conv2DTranspose : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 2);
    const Tensor<T> &A      = *in[0];
    const Tensor<T> &kernel = *in[1];
    m_out.quantizer         = A.quantizer - m_q;

    const int N     = A.dims()[0];
    const int in_H  = A.dims()[1];
    const int in_W  = A.dims()[2];
    const int Cin   = A.dims()[3];
    const int Kh    = kernel.dims()[0];
    const int Kw    = kernel.dims()[1];
    const int Cout  = kernel.dims()[2];
    const int out_H = m_out.dims()[1];
    const int out_W = m_out.dims()[2];
    const int shift = kernel.quantizer + m_q;
    const int total = N * out_H * out_W * Cout;
    (void)total;

    int half_h = (Kh - 1) / 2;
    int half_w = (Kw - 1) / 2;

#ifdef __CUDACC__
    conv2dtranspose_kernel<<<gridDim1D(total, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
        A.dev_data(), kernel.dev_data(), m_out.dev_data(),
        N, in_H, in_W, Cin, Kh, Kw, Cout,
        out_H, out_W, m_strides[1], m_strides[2],
        half_h, half_w, shift);
#else
    for (int n = 0; n < N; ++n)
    for (int out_h = 0; out_h < out_H; ++out_h)
      for (int out_w = 0; out_w < out_W; ++out_w)
        for (int out_c = 0; out_c < Cout; ++out_c)
        {
          int32_t acc = 0;
          for (int ki = 0; ki < Kh; ++ki)
          {
            int nom_h = out_h + half_h - ki;
            if (nom_h < 0 || nom_h % m_strides[1] != 0) continue;
            int i1 = nom_h / m_strides[1];
            if (i1 >= in_H) continue;
            for (int kj = 0; kj < Kw; ++kj)
            {
              int nom_w = out_w + half_w - kj;
              if (nom_w < 0 || nom_w % m_strides[2] != 0) continue;
              int j1 = nom_w / m_strides[2];
              if (j1 >= in_W) continue;
              for (int ic = 0; ic < Cin; ++ic)
                acc += (int32_t)A(n, i1, j1, ic) * (int32_t)kernel(ki, kj, out_c, ic);
            }
          }
          acc >>= shift;
          m_out(n, out_h, out_w, out_c) = static_cast<T>(SADL_CUDA_SAT(acc));
        }
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2 || in[0]->dims().size() != 4 || in[1]->dims().size() != 4) return false;
    if (in[1]->dims()[0] != in[1]->dims()[1]) return false;
    int k = in[1]->dims()[0];
    if (k != 2 && k != 3 && k != 4 && k != 5) return false;

    Dimensions dim;
    dim.resize(4);
    dim[0] = in[0]->dims()[0];
    dim[1] = m_strides[1] * (in[0]->dims()[1] - 1) + m_out_pads[0] +
             ((in[1]->dims()[0] - 1) * 1 + 1) - (m_pads[0] + m_pads[2]);
    dim[2] = m_strides[2] * (in[0]->dims()[2] - 1) + m_out_pads[1] +
             ((in[1]->dims()[1] - 1) * 1 + 1) - (m_pads[1] + m_pads[3]);
    dim[3] = in[1]->dims()[2];
    m_out.resize(dim);
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
    if (!readDims(m_strides)) return false;
    if (m_strides.size() == 2)
      m_strides = Dimensions({1, m_strides[0], m_strides[1], 1});
    if (!readDims(m_pads)) return false;
    if (!readDims(m_out_pads)) return false;
    file.read(reinterpret_cast<char *>(&m_q), sizeof(m_q));
    return true;
  }

  Dimensions m_strides;
  Dimensions m_pads;
  Dimensions m_out_pads;
  int        m_q = 0;
};

} 
}  
