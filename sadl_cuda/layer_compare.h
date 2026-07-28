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

template<typename T> class Compare : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 2);
    const Tensor<T> &A = *in[0];
    const Tensor<T> &B = *in[1];
    const int n   = static_cast<int>(m_out.size());
    const int b_n = static_cast<int>(B.size());

    // align
    const int shift_a = std::max(0, B.quantizer - A.quantizer);
    const int shift_b = std::max(0, A.quantizer - B.quantizer);

    m_out.quantizer = 0;
    for (int i = 0; i < n; ++i)
    {
      int32_t va = (int32_t)A[i] << shift_a;
      int32_t vb = (int32_t)B[b_n == 1 ? 0 : (i % b_n)] << shift_b;
      bool    r;
      switch (m_mode)
      {
      case 0:  r = (va <  vb); break;
      case 1:  r = (va >  vb); break;
      default: r = (va == vb); break;
      }
      m_out[i] = static_cast<T>(r ? 1 : 0);
    }
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
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t mode = 0;
    file.read(reinterpret_cast<char *>(&mode), sizeof(mode));
    m_mode = mode;
    return true;
  }
  int m_mode = 0;
};

}  // namespace layers
}  // namespace sadl_cuda
