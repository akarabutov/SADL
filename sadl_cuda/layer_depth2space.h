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


template<typename T> class Depth2Space : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;
  // cpu
  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 1);
    const Tensor<T> &src = *in[0];
    m_out.quantizer = src.quantizer;

    const int N    = src.dims()[0];
    const int Hsrc = src.dims()[1];
    const int Wsrc = src.dims()[2];
    const int Cdst = src.dims()[3] / (m_blocksize * m_blocksize);
    const int bs   = m_blocksize;

    for (int n = 0; n < N; ++n)
      for (int h = 0; h < Hsrc; ++h)
        for (int w = 0; w < Wsrc; ++w)
          for (int c = 0; c < Cdst; ++c)
            for (int bh = 0; bh < bs; ++bh)
              for (int bw = 0; bw < bs; ++bw)
                m_out(n, h*bs + bh, w*bs + bw, c) =
                    src(n, h, w, c * bs * bs + bh * bs + bw);
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 1 || in[0]->dims().size() != 4) return false;
    Dimensions dim = in[0]->dims();
    dim[1] *= m_blocksize;
    dim[2] *= m_blocksize;
    dim[3] /= (m_blocksize * m_blocksize);
    m_out.resize(dim);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t x = 0;
    file.read(reinterpret_cast<char *>(&x), sizeof(x));
    if (x <= 0) return false;
    m_blocksize = x;
    return true;
  }
  int m_blocksize = 2;
};

}  
}  
