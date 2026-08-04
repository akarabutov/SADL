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

// dused Conv2D + BiasAdd, bit exact

#ifdef __CUDACC__
// spe
template<int BM, int BN, int BK, int TM, int TN>
__global__ static void conv1x1_biasadd_kernel(
    const int16_t * __restrict__ A,    // [M, Cin]   (M = N*H*W)
    const int16_t * __restrict__ K,    // [Cout, Cin]
    const int16_t * __restrict__ bias, // [Cout]
    int16_t       * __restrict__ out,  // [M, Cout]
    int M, int Cin, int Cout,
    int conv_shift, int bias_shift)
{
  __shared__ int16_t sA[BM][BK];
  __shared__ int16_t sK[BN][BK];

  constexpr int THREADS_M = BM / TM;
  constexpr int THREADS_N = BN / TN;
  constexpr int BTHREADS  = THREADS_M * THREADS_N;

  const int bx  = blockIdx.x;          // tile  Cout
  const int by  = blockIdx.y;          // tile  pixels
  const int tx  = threadIdx.x;         // 0..THREADS_N-1
  const int ty  = threadIdx.y;         // 0..THREADS_M-1
  const int tid = ty * THREADS_N + tx;

  // TM×TN outputs /thread
  int32_t acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0;

  for (int k0 = 0; k0 < Cin; k0 += BK)
  {
    // 
    #pragma unroll
    for (int i = tid; i < BM * BK; i += BTHREADS)
    {
      const int mm = i / BK;
      const int kk = i % BK;
      const int gm = by * BM + mm;
      const int gk = k0 + kk;
      sA[mm][kk] = (gm < M && gk < Cin) ? A[gm * Cin + gk] : (int16_t)0;
    }
   
    #pragma unroll
    for (int i = tid; i < BN * BK; i += BTHREADS)
    {
      const int nn = i / BK;
      const int kk = i % BK;
      const int gn = bx * BN + nn;
      const int gk = k0 + kk;
      sK[nn][kk] = (gn < Cout && gk < Cin) ? K[gn * Cin + gk] : (int16_t)0;
    }
    __syncthreads();

   
    #pragma unroll
    for (int k = 0; k < BK; ++k)
    {
      int16_t regA[TM], regK[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) regA[i] = sA[ty * TM + i][k];
      #pragma unroll
      for (int j = 0; j < TN; ++j) regK[j] = sK[tx * TN + j][k];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += (int32_t)regA[i] * (int32_t)regK[j];
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j)
    {
      const int m = by * BM + ty * TM + i;
      const int n = bx * BN + tx * TN + j;
      if (m < M && n < Cout)
      {
        int32_t v = acc[i][j];
        v >>= conv_shift;
        v = SADL_CUDA_SAT(v);
        int32_t va = (bias_shift < 0) ? (v << (-bias_shift)) : (v >> bias_shift);
        va = SADL_CUDA_SAT(va + (int32_t)bias[n]);
        out[m * Cout + n] = (int16_t)va;
      }
    }
}

__global__ static void conv2d_biasadd_kernel(
    const int16_t * __restrict__ A,
    const int16_t * __restrict__ K,
    const int16_t * __restrict__ bias,
    int16_t       * __restrict__ out,
    int N, int in_H, int in_W, int Cin,
    int Kh, int Kw, int Cout, int cin_by_g,
    int out_H, int out_W,
    int s_h, int s_w, int pad_h, int pad_w,
    int groups, int conv_shift, int bias_shift)
{
  int idx   = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * out_H * out_W * Cout;
  if (idx >= total) return;

  int out_c = idx % Cout;
  int tmp   = idx / Cout;
  int out_w = tmp % out_W;
  tmp       = tmp / out_W;
  int out_h = tmp % out_H;
  int n     = tmp / out_H;

  int cout_by_g = Cout / groups;
  int g         = out_c / cout_by_g;
  int ic_offset = g * cin_by_g;

  const int16_t *A_batch = A + n * in_H * in_W * Cin;

  // conv 
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
  // sat
  acc >>= conv_shift;
  acc = SADL_CUDA_SAT(acc);

  // add
  int32_t va;
  if (bias_shift < 0)
    va = acc << (-bias_shift);
  else
    va = acc >> bias_shift;
  va = SADL_CUDA_SAT(va + (int32_t)bias[out_c]);

  out[idx] = (int16_t)va;
}
#endif

template<typename T> class Conv2DBias : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  void setFusionParams(Dimensions strides, Dimensions pads, int groups,
                       int conv_q, int bias_q)
  {
    m_strides = strides;
    m_pads    = pads;
    m_groups  = groups;
    m_conv_q  = conv_q;
    m_bias_q  = bias_q;
  }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 3);
    const Tensor<T> &A    = *in[0];
    const Tensor<T> &K    = *in[1];
    const Tensor<T> &bias = *in[2];

    const int conv_shift = K.quantizer + m_conv_q;
    const int bias_shift = (A.quantizer - m_conv_q) - m_bias_q;
    m_out.quantizer      = m_bias_q;

    const int N        = A.dims()[0];
    const int in_H     = A.dims()[1];
    const int in_W     = A.dims()[2];
    const int Cin      = A.dims()[3];
    const int Kh       = K.dims()[0];
    const int Kw       = K.dims()[1];
    const int Cout     = K.dims()[2];
    const int cin_by_g = K.dims()[3];
    const int out_H    = m_out.dims()[1];
    const int out_W    = m_out.dims()[2];
    const int s_h      = m_strides[1];
    const int s_w      = m_strides[2];
    const int pad_h    = m_pads[0];
    const int pad_w    = m_pads[1];
    const int total    = N * out_H * out_W * Cout;
    (void)Cin; (void)total;

#ifdef __CUDACC__
    // spe
    if (Kh == 1 && Kw == 1 && m_groups == 1 && s_h == 1 && s_w == 1
        && pad_h == 0 && pad_w == 0)
    {
     
      constexpr int BM = 16, BN = 16, BK = 32, TM = 1, TN = 1;
      const int M = N * out_H * out_W;
      dim3 block(BN / TN, BM / TM);                 
      dim3 grid((Cout + BN - 1) / BN, (M + BM - 1) / BM);
      conv1x1_biasadd_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
          A.dev_data(), K.dev_data(), bias.dev_data(), m_out.dev_data(),
          M, Cin, Cout, conv_shift, bias_shift);
    }
    else
    {
      conv2d_biasadd_kernel<<<gridDim1D(total, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
          A.dev_data(), K.dev_data(), bias.dev_data(),
          m_out.dev_data(),
          N, in_H, in_W, Cin, Kh, Kw, Cout, cin_by_g,
          out_H, out_W, s_h, s_w, pad_h, pad_w,
          m_groups, conv_shift, bias_shift);
    }
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
                       (int32_t)K(kh, kw, out_c, ic);
            }
          }
          acc >>= conv_shift;
          acc = SADL_CUDA_SAT(acc);

          int32_t va;
          if (bias_shift < 0) va = acc << (-bias_shift);
          else                va = acc >> bias_shift;
          va = SADL_CUDA_SAT(va + (int32_t)bias[out_c]);

          m_out(n, out_h, out_w, out_c) = static_cast<T>(va);
        }
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 3) return false;
    const int N    = in[0]->dims()[0];
    const int in_H = in[0]->dims()[1];
    const int in_W = in[0]->dims()[2];
    const int Kh   = in[1]->dims()[0];
    const int Kw   = in[1]->dims()[1];
    const int Cout = in[1]->dims()[2];
    const int s_h  = m_strides[1];
    const int s_w  = m_strides[2];

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
  virtual bool loadInternal(std::istream &, Version) override { return true; }

  Dimensions m_strides;
  Dimensions m_pads;
  int        m_groups = 1;
  int        m_conv_q = 0;
  int        m_bias_q = 0;
};

}  // namespace layers
} 
