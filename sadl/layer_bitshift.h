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
#include <cmath>

namespace sadl
{
namespace layers
{
/*
  Support of BitShift operator from https://onnx.ai/onnx/operators/onnx__BitShift.html 
*/
template<typename T> class BitShift : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;


  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;

private:
  bool  m_rightDirection;
};


// TODO : check all dims, check loop for optiz
template<typename T> bool BitShift<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  if (in[0] == in[1])
  {
    std::cerr << "  input aliasing" << std::endl;
    return false;
  }

  swap(*in[0], m_out);
  // adapt output width to second input (which are the bias) in order to be able to rescale as desired the input
  //out_.quantizer =qfinal;

  if (( in[0]->dims().size() == 4) && (in[1]->dims().size() == 1) ) 
  {
    const Tensor<T> &B = *in[1];
    const int        N = in[0]->dims()[0];
    const int        H = in[0]->dims()[1];
    const int        W = in[0]->dims()[2];
    const int        K = in[0]->dims()[3];
    assert((B.dims().size() == 1 && B.dims()[0] == K) || 
          (B.dims().size() == 2 && B.dims()[0] == 1 && B.dims()[1] == K) || 
          (B.dims().size() == 3 && B.dims()[0] == 1 && B.dims()[1] == 1 && B.dims()[2] == K) ||
          (B.dims().size() == 4 && B.dims()[0] == 1 && B.dims()[1] == 1 && B.dims()[2] == 1 && B.dims()[3] == K));
    for (int n = 0; n < N; ++n)
      for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
          for (int k = 0; k < K; ++k)
          {
            T z = m_out(n, i, j, k);
            //ComputationType<T>::quantize(z, shift);
            assert(B[k] >= 0);
            if constexpr (std::is_same<T, float>::value)
            {
              // float branch
              T m = 
#ifdef _MSC_BUILD 
                    std::powf
#else
                    std::pow
#endif
                    (2., B[k]);
              if (m_rightDirection)
                z /= m;
              else
                z *= m;
              z = std::floor(z);    // Added to align it with integer operation
            } else {
              // integer branch
              if (m_rightDirection)
                z >>= B[k]; 
              else
                z <<= B[k];
            }
            COUNTERS(z);
            SATURATE(z);
            m_out(n, i, j, k) = z;
          }
  }

  return true;
}


// data in in[0]
// assume data shape [N,W,H,D]
template<typename T> bool BitShift<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  SADL_DBG(std::cout << "  - " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);

  // either broadcast from a tensor of size [n] (use when input is Const) or [1,n]
  // of BitShift if same dimensions
  if (
      (in[1]->dims().size()==4 && in[0]->dims()[3] != in[1]->dims()[3]) ||
      (in[1]->dims().size()==1 && in[0]->dims()[3] != in[1]->dims()[0])
    )
  {
    return false;
  }
  m_out.resize(in[0]->dims());
  m_initDone = true;
  return true;
}

template<typename T> bool BitShift<T>::loadInternal(std::istream &file, Version)
{
  int32_t mode;
  file.read((char *) &mode, sizeof(mode));
  m_rightDirection = mode == 1;
  return true;
}

}   // namespace layers
}   // namespace sadl
