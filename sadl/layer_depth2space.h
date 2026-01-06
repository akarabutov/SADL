/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2025, ITU/ISO/IEC
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

namespace sadl
{
namespace layers
{
template<typename T> class Depth2Space : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  void process_depth2space(Tensor<T> &in, Tensor<T> &out, int factor);

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
  Dimensions   dim_;   // dims after Depth2Space
  int blocksize;
};

template<typename T> void Depth2Space<T>::process_depth2space(Tensor<T> &in, Tensor<T> &out, int factor)
{
    int N = in.dims()[0];
    int Hsrc = in.dims()[1];
    int Wsrc = in.dims()[2];

    int Cdst = in.dims()[3] / (factor * factor);

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < Hsrc; h++) {
            for (int w = 0; w < Wsrc; w++) {
                for (int c = 0; c < Cdst; c++) {
                    for (int bh = 0; bh < factor; bh++) {
                        for (int bw = 0; bw < factor; bw++) {
                            for (int i = 0; i < 4; ++i)
                            {
                              out(n, h*factor + bh, w*factor+bw, c) = in(n, h, w, c * factor * factor + bh * factor + bw);
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename T> bool Depth2Space<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 1);
  assert(in[0]->size() == m_out.size());

  process_depth2space(*in[0], m_out, blocksize);
  // resize done at init
  //swapData(*in[0], out_);

  return true;
}

template<typename T> bool Depth2Space<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1)
    return false;
  SADL_DBG(std::cout << "  - " << in[0]->dims() << std::endl);
  dim_ = in[0]->dims();
  assert( dim_.size() == 4 );
  dim_[1] *= blocksize;
  dim_[2] *= blocksize;
  assert( (dim_[3] % (blocksize * blocksize)) == 0 );
  dim_[3] /= (blocksize * blocksize);
  m_out.resize(dim_);
  m_initDone = true;
  return true;
}

template<typename T> bool Depth2Space<T>::loadInternal(std::istream &file, Version)
{
  // load values
  int32_t x = 0;
  file.read((char *) &x, sizeof(x));
  if (x <= 0 )
  {
    std::cerr << "[ERROR] invalid block size: " << x << std::endl;
    return false;
  }
  blocksize = x;
  SADL_DBG(std::cout << "  - block size: " << blocksize << std::endl);
  return true;
}

}   // namespace layers
}   // namespace sadl
