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
#include <vector>

namespace sadl_cuda
{
namespace layers
{


template<typename T> class GatherElements : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    // cpu
    assert(in.size() == 2);
    const Tensor<T> &A  = *in[0];
    const Tensor<T> &IX = *in[1];
    m_out.quantizer = A.quantizer;

    const int r  = static_cast<int>(A.dims().size());
    const int a  = (m_axis < 0) ? m_axis + r : m_axis;
    const int Da = A.dims()[a];
    const int P  = static_cast<int>(m_out.size());

   
    std::vector<int64_t> sD(r, 1), sO(r, 1);
    for (int i = r - 2; i >= 0; --i) sD[i] = sD[i+1] * A.dims()[i+1];
    for (int i = r - 2; i >= 0; --i) sO[i] = sO[i+1] * m_out.dims()[i+1];

    int obuf[Dimensions::MaxDim];
    int jbuf[Dimensions::MaxDim];

    for (int p = 0; p < P; ++p)
    {
      int64_t tmp = p;
      for (int i = 0; i < r; ++i) obuf[i] = static_cast<int>((tmp / sO[i]) % m_out.dims()[i]);

      int kk = static_cast<int>(IX[p]);
      if (kk < 0) kk += Da;

      for (int i = 0; i < r; ++i) jbuf[i] = obuf[i];
      jbuf[a] = kk;

      int64_t linD = 0;
      for (int i = 0; i < r; ++i) linD += jbuf[i] * sD[i];
      m_out[p] = A[static_cast<int>(linD)];
    }
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    const int r = static_cast<int>(in[0]->dims().size());
    const int a = (m_axis < 0) ? m_axis + r : m_axis;
    if (a < 0 || a >= r) return false;
    if (in[0]->dims().size() != in[1]->dims().size()) return false;
    m_out.resize(in[1]->dims());
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t ax = 0;
    file.read(reinterpret_cast<char *>(&ax), sizeof(ax));
    m_axis = static_cast<int>(ax);
    return true;
  }
  int m_axis = 0;
};

}  
} 
