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


template<typename T> class Expand : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;
  // cpu
  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 2);
    const Tensor<T> &A = *in[0];
    m_out.quantizer = A.quantizer;

    if (A.size() == 1)
    {
      m_out.fill(A[0]);
      return true;
    }

  
    const int nd     = static_cast<int>(m_out.dims().size());
    const int nd_src = static_cast<int>(A.dims().size());
    const int n_out  = static_cast<int>(m_out.size());

    for (int flat = 0; flat < n_out; ++flat)
    {
      int tmp = flat;
      int src_flat = 0;
      int src_stride = 1;
      for (int k = nd - 1; k >= 0; --k)
      {
        int idx     = tmp % m_out.dims()[k];
        tmp /= m_out.dims()[k]; // to be removed
        int src_k   = k - (nd - nd_src);
        int src_dim = (src_k >= 0) ? A.dims()[src_k] : 1;
        int src_idx = (src_dim == 1) ? 0 : idx;
        src_flat += src_idx * src_stride;
        src_stride *= src_dim;
      }
      m_out[flat] = A[src_flat];
    }
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    const Tensor<T> &shape_t = *in[1];
    const int nd = static_cast<int>(shape_t.size());
    Dimensions dim;
    dim.resize(nd);
    for (int k = 0; k < nd; ++k)
      dim[k] = static_cast<int>(shape_t[k]);
    m_out.resize(dim);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
};

} 
}  
