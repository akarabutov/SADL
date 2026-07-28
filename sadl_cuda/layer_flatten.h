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

// Flatten to [N, rest] or [1, total] depending on axis.
template<typename T> class Flatten : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool     isGpuLayer() const override { return true; }
  ExecMode execMode()   const override { return ExecMode::Noop; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 1);
    assert(in[0]->size() == m_out.size());
    swapData(*in[0], m_out);
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 1) return false;
    const Dimensions &d = in[0]->dims();
    int64_t left = 1, right = 1;
    for (int i = 0; i < m_axis && i < d.size(); ++i) left  *= d[i];
    for (int i = m_axis; i < d.size();            ++i) right *= d[i];
    m_flat_dims.resize(2);
    m_flat_dims[0] = static_cast<int>(left);
    m_flat_dims[1] = static_cast<int>(right);
    m_out.resize(m_flat_dims);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t x = 0;
    file.read(reinterpret_cast<char *>(&x), sizeof(x));
    m_axis = x;
    return true;
  }
  int        m_axis      = 1;
  Dimensions m_flat_dims;
};

} 
}  
