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
#include <algorithm>

namespace sadl_cuda
{
namespace layers
{

template<typename T> class ScatterND : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 3);
    const Tensor<T> &data    = *in[0];
    const Tensor<T> &updates = *in[1];
    const Tensor<T> &indices = *in[2];
    assert(indices.dims().size() == 4);

    std::copy(data.begin(), data.end(), m_out.begin());
    m_out.quantizer = data.quantizer;

    const int N     = data.dims()[0];
    const int dim_H = indices.dims()[0];
    const int dim_W = indices.dims()[1];
    const int dim_C = indices.dims()[2];

    // to see : batch
    for (int n = 0; n < N; ++n)
      for (int h = 0; h < dim_H; ++h)
        for (int w = 0; w < dim_W; ++w)
          for (int c = 0; c < dim_C; ++c)
          {
            int iH = static_cast<int>(indices(h, w, c, 1));
            int iW = static_cast<int>(indices(h, w, c, 2));
            int iC = static_cast<int>(indices(h, w, c, 3));
            m_out(n, iH, iW, iC) = updates(n, h, w, c);
          }
    // CPU layer: invalidate device copy so next GPU consumer re-syncs.
    m_out.setOnDevice(false);
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 3) return false;
    m_out.resize(in[0]->dims());
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &, Version) override { return true; }
};

}  // namespace layers
}  // namespace sadl_cuda
