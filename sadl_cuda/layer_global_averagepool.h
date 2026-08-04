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
static constexpr int GAP_BLOCK = 256;

__global__ static void global_avgpool_kernel(
    const int16_t * __restrict__ A, int16_t * __restrict__ out,
    int H, int W, int C, int N,
    int divq, int q, int r,
    int shift_bits, int r_shift, int q2, int r2, int two_pass)
{
  const int nc = blockIdx.x;               // one block per (n, c)
  if (nc >= N * C) return;
  const int n  = nc / C;
  const int c  = nc % C;
  const int HW = H * W;

  const int16_t *base = A + ((int64_t) n * HW) * C + c;

  int32_t local = 0;
  for (int hw = threadIdx.x; hw < HW; hw += blockDim.x)
    local += (int32_t) base[(int64_t) hw * C];

  __shared__ int32_t sdata[GAP_BLOCK];
  sdata[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    const int32_t sum = sdata[0];
    int32_t res;
    if (!two_pass)
      res = (int32_t)(((int64_t)sum * divq + r) >> q);
    else
    {
      int32_t temp = (sum + r_shift) >> shift_bits;
      res = (int32_t)(((int64_t)temp * divq + r2) >> q2);
    }
    out[n * C + c] = (int16_t)SADL_CUDA_SAT(res);
  }
}
#endif

template<typename T> class GlobalAveragePool : public Layer<T>
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
    const int N = in[0]->dims()[0];
    const int H = in[0]->dims()[1];
    const int W = in[0]->dims()[2];
    const int C = in[0]->dims()[3];
    const int total = N * C;
    (void)total;

#ifdef __CUDACC__
    global_avgpool_kernel<<<total, GAP_BLOCK>>>(
        in[0]->dev_data(), m_out.dev_data(), H, W, C, N,
        m_divq, m_q, m_r,
        m_shift_bits, m_r_shift, m_q2, m_r2, m_two_pass ? 1 : 0);
#else
    for (int n = 0; n < N; ++n)
      for (int c = 0; c < C; ++c)
      {
        int32_t sum = 0;
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
            sum += (int32_t)(*in[0])(n, h, w, c);
        int32_t res;
        if (!m_two_pass)
          res = (int32_t)(((int64_t)sum * m_divq + m_r) >> m_q);
        else
        {
          int32_t temp = (sum + m_r_shift) >> m_shift_bits;
          res = (int32_t)(((int64_t)temp * m_divq + m_r2) >> m_q2);
        }
        m_out(n, 0, 0, c) = static_cast<T>(SADL_CUDA_SAT(res));
      }
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 1 || in[0]->dims().size() != 4) return false;
    const int H = in[0]->dims()[1];
    const int W = in[0]->dims()[2];
    m_out.resize({in[0]->dims()[0], 1, 1, in[0]->dims()[3]});

    const int cnt      = H * W;
    const int N_bits   = (int)sizeof(T) * 8 - 1;          // 15 for int16
    const int N2_bits  = (int)sizeof(int32_t) * 8 - 1;    // 31
    const int acc_bits = (int)std::ceil(std::log2((double)std::max(cnt, 2)));
    const int minq     = std::max(N_bits, N2_bits - (N_bits + acc_bits));
    m_q    = std::min(N2_bits, minq + acc_bits - 1);
    m_divq = (int)(((1LL << m_q) + cnt / 2) / cnt);
    m_r    = (int)((1LL << m_q) / 2);

    const int divq_bits = (int)std::ceil(std::log2((double)std::max(m_divq, 2)));
    m_two_pass = (divq_bits + acc_bits + N_bits > N2_bits);
    if (m_two_pass)
    {
      m_shift_bits = divq_bits - std::max(N2_bits - (acc_bits + N_bits), 0);
      m_q2         = m_q - m_shift_bits;
      m_r_shift    = (int)((1LL << m_shift_bits) / 2);
      m_r2         = (int)((1LL << m_q2) / 2);
    }

    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
  int  m_divq      = 1;
  int  m_q         = 0;
  int  m_r         = 0;
  bool m_two_pass  = false;
  int  m_shift_bits = 0;
  int  m_q2        = 0;
  int  m_r_shift   = 0;
  int  m_r2        = 0;
};

}  // namespace layers
}  
