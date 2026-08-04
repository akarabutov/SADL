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
#include <cstring>

namespace sadl_cuda
{
namespace layers
{


template<typename T> class Tile : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 2);
    m_out.quantizer = in[0]->quantizer;

    const int in_N = in[0]->dims()[0], in_H = in[0]->dims()[1];
    const int in_W = in[0]->dims()[2], in_K = in[0]->dims()[3];
    const int out_H = m_out.dims()[1], out_W = m_out.dims()[2], out_K = m_out.dims()[3];
    const int out_N = m_out.dims()[0];

    for (int d0 = 0; d0 < out_N; ++d0)
      for (int d1 = 0; d1 < out_H; ++d1)
        for (int d2 = 0; d2 < out_W; ++d2)
        {
          int inputIdx  = (d0 % in_N) * (in_H * in_W * in_K)
                        + (d1 % in_H) * (in_W * in_K)
                        + (d2 % in_W) * in_K;
          int outputIdx = d0 * (out_H * out_W * out_K)
                        + d1 * (out_W * out_K)
                        + d2 * out_K;
          const int tile_k = static_cast<int>((*in[1])[3]);
          for (int idx = 0; idx < tile_k; ++idx)
            std::memcpy(&m_out[outputIdx + idx * in_K], &(*in[0])[inputIdx], sizeof(T) * in_K);
        }
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2 || in[0]->dims().size() != 4 || in[1]->dims().size() != 1)
      return false;
    Dimensions dims;
    dims.resize(static_cast<int>(in[1]->size()));
    for (int i = 0; i < static_cast<int>(in[1]->size()); ++i)
      dims[i] = static_cast<int>((*in[1])[i]) * in[0]->dims()[i];
    m_out.resize(dims);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
};

}  
} 
