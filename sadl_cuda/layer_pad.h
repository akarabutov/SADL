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
#include <cstring>

namespace sadl_cuda
{
namespace layers
{


template<typename T> class Pad : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    if (m_mode == 0) assert(in.size() == 2);
    else             assert(in.size() == 1);

    m_out.quantizer = in[0]->quantizer;
    const Tensor<T> &A     = *in[0];
    const auto       &ind  = A.dims();
    const auto       &outd = m_out.dims();

    const int N   = outd[0], H = outd[1], W = outd[2], C = outd[3];
    const int inH = ind[1],  inW = ind[2];
    const int ph  = m_pads[0], pw = m_pads[1];

    for (int n = 0; n < N; ++n)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          for (int c = 0; c < C; ++c)
          {
            int ih = h - ph;
            int iw = w - pw;
            if (m_mode == 0)  // constant
            {
              m_out(n, h, w, c) = (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                      ? A(n, ih, iw, c)
                                      : (*in[1])[0];
            }
            else if (m_mode == 2)  // edge
            {
              ih = std::clamp(ih, 0, inH - 1);
              iw = std::clamp(iw, 0, inW - 1);
              m_out(n, h, w, c) = A(n, ih, iw, c);
            }
            else  // reflect
            {
              auto reflect = [](int x, int lim) -> int
              {
                if (lim <= 1) return 0;
                x = std::abs(x);
                int period = 2 * (lim - 1);
                x %= period;
                return (x < lim) ? x : period - x;
              };
              m_out(n, h, w, c) = A(n, reflect(ih, inH), reflect(iw, inW), c);
            }
          }
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (m_mode == 0) { if (in.size() != 2) return false; }
    else             { if (in.size() != 1) return false; }
    if (in[0]->dims().size() != 4) return false;

    Dimensions dims = in[0]->dims();
    dims[1] += m_pads[0] + m_pads[2];
    dims[2] += m_pads[1] + m_pads[3];
    m_out.resize(dims);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t mode = 0;
    file.read(reinterpret_cast<char *>(&mode), sizeof(mode));
    m_mode = mode;
    for (int i = 0; i < 4; ++i)
    {
      int32_t pad = 0;
      file.read(reinterpret_cast<char *>(&pad), sizeof(pad));
      m_pads[i] = pad;
    }
    return true;
  }

  int                m_mode = 0;
  std::array<int, 4> m_pads{};
};

} 
}  
