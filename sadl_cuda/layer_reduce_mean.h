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
#include <cmath>
#include <vector>

namespace sadl_cuda
{
namespace layers
{


template<typename T> class ReduceMean : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 1);
    m_out.quantizer = in[0]->quantizer;
    const Tensor<T> &A = *in[0];
    const int N = A.dims()[0];
    const int H = A.dims()[1];
    const int W = A.dims()[2];
    const int D = A.dims()[3];

    std::vector<int32_t> acc(m_out.size(), 0);
    for (int n = 0; n < N; ++n)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          for (int d = 0; d < D; ++d)
          {
            int ni = n / m_reduce_dims[0];
            int hi = h / m_reduce_dims[1];
            int wi = w / m_reduce_dims[2];
            int di = d / m_reduce_dims[3];
            int idx = ((ni * m_out.dims()[1] + hi) * m_out.dims()[2] + wi) * m_out.dims()[3] + di;
            acc[idx] += (int32_t)A(n, h, w, d);
          }
    // to see: batch issue?      
    const bool two_pass = (m_divq_bits + m_acc_bits + 1 > (int)sizeof(int32_t) * 8 - 1);
    (void)N;
    for (int i = 0; i < (int)m_out.size(); ++i)
    {
      int32_t res;
      if (!two_pass)
        res = (int32_t)(((int64_t)acc[i] * m_divq + m_r) >> m_q);
      else
      {
        int32_t temp = (acc[i] + m_r_shift) >> m_shift_bits;
        res = (int32_t)(((int64_t)temp * m_divq + m_r2) >> m_q2);
      }
      m_out[i] = static_cast<T>(SADL_CUDA_SAT(res));
    }
    m_out.setOnDevice(false);
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 1 || in[0]->dims().size() != 4) return false;

    Dimensions out_dim = in[0]->dims();
    for (int ax : m_axes) out_dim[ax] = 1;
    m_out.resize(out_dim);

    m_cnt = 1;
    m_reduce_dims = {1, 1, 1, 1};
    for (int ax : m_axes)
    {
      m_cnt *= in[0]->dims()[ax];
      m_reduce_dims[ax] = in[0]->dims()[ax];
    }

    const int N_bits  = (int)sizeof(T) * 8 - 1;
    const int N2_bits = (int)sizeof(int32_t) * 8 - 1;
    m_acc_bits = (int)std::ceil(std::log2((double)std::max(m_cnt, 2)));
    const int minq = std::max(N_bits, N2_bits - (N_bits + m_acc_bits));
    m_q    = std::min(N2_bits, minq + m_acc_bits - 1);
    m_divq = (int)(((1LL << m_q) + m_cnt / 2) / m_cnt);
    m_r    = (int)((1LL << m_q) / 2);

    m_divq_bits = (int)std::ceil(std::log2((double)std::max(m_divq, 2)));
    {
      m_shift_bits = m_divq_bits - std::max(N2_bits - (m_acc_bits + N_bits), 0);
      m_q2         = m_q - m_shift_bits;
      m_r_shift    = (int)((1LL << m_shift_bits) / 2);
      m_r2         = (int)((1LL << m_q2) / 2);
    }

    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t x = 0;
    file.read(reinterpret_cast<char *>(&x), sizeof(x));
    if (x <= 0 || x > Dimensions::MaxDim) return false;
    m_axes.resize(x);
    for (int k = 0; k < (int)m_axes.size(); ++k)
    {
      file.read(reinterpret_cast<char *>(&x), sizeof(x));
      m_axes[k] = x;
    }
    bool keepdims = false;
    file.read(reinterpret_cast<char *>(&keepdims), sizeof(keepdims));
    if (!keepdims) return false;
    return true;
  }

  std::vector<int> m_axes;
  Dimensions       m_reduce_dims;
  int  m_cnt        = 1;
  int  m_divq       = 1;
  int  m_q          = 0;
  int  m_r          = 0;
  int  m_divq_bits  = 0;
  int  m_acc_bits   = 0;
  int  m_shift_bits = 0;
  int  m_q2         = 0;
  int  m_r_shift    = 0;
  int  m_r2         = 0;
};

} 
}  // namespace sadl_cuda
