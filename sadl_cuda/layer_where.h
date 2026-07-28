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

namespace sadl_cuda
{
namespace layers
{

template<typename T> class Where : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 3);
    const Tensor<T> &cond = *in[0];
    const Tensor<T> &X    = *in[1];
    const Tensor<T> &Y    = *in[2];

    const int qx = X.quantizer, qy = Y.quantizer;
    const int qfinal  = std::max(qx, qy);
    const int shift_x = qfinal - qx;  
    const int shift_y = qfinal - qy; 
    m_out.quantizer = qfinal;

    const int n   = static_cast<int>(m_out.size());
    const int x_n = static_cast<int>(X.size());
    const int y_n = static_cast<int>(Y.size());

    if (cond.size() == 1)
    {
      const Tensor<T> &src = cond[0] ? X : Y;
      const int        sq  = cond[0] ? shift_x : shift_y;
      const int        sn  = static_cast<int>(src.size());
      for (int i = 0; i < n; ++i)
        m_out[i] = static_cast<T>(SADL_CUDA_SAT((int32_t)src[sn == 1 ? 0 : i] << sq));
      return true;
    }

    for (int i = 0; i < n; ++i)
    {
      int32_t xv = (int32_t)X[x_n == 1 ? 0 : i] << shift_x;
      int32_t yv = (int32_t)Y[y_n == 1 ? 0 : i] << shift_y;
      m_out[i]   = static_cast<T>(cond[i] ? xv : yv);
    }
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 3) return false;
    if (in[0]->size() == 1)
      m_out.resize(in[1]->dims());
    else
      m_out.resize(in[0]->dims());
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
};

}  // namespace layers
}  // namespace sadl_cuda
