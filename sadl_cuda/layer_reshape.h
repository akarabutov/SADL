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

template<typename T> class Reshape : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool     isGpuLayer() const override { return true; }
  ExecMode execMode()   const override { return ExecMode::Noop; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 2);
    assert(in[0]->size() == m_out.size());
    swapData(*in[0], m_out);
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    const Tensor<T> &shape = *in[1];
    Dimensions newdims;
    newdims.resize(static_cast<int>(shape.size()));
    int64_t known = 1;
    int     infer_idx = -1;
    for (int i = 0; i < newdims.size(); ++i)
    {
      int v = static_cast<int>(shape[i]);
      newdims[i] = v;
      if (v == -1) infer_idx = i;
      else         known *= v;
    }
    // to see: batch 
    const int in_N = (in[0]->dims().size() >= 1) ? in[0]->dims()[0] : 1;
    if (newdims.size() >= 1 && newdims[0] == 1 && in_N > 1 && infer_idx != 0)
    {
      const int64_t per_slot_in = in[0]->dims().nbElements() / in_N;
      const bool    has_minus_one = (infer_idx >= 0);
      if (has_minus_one || known == per_slot_in)
      {
        newdims[0] = in_N;
        known *= in_N;  
      }
    }
    if (infer_idx >= 0)
      newdims[infer_idx] = static_cast<int>(in[0]->dims().nbElements() / known);
    m_out.resize(newdims);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
};

}  
}  
