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
template<typename T> class Clip : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;
  Tensor<T>       m_limits;


  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
  typename ComputationType<T>::type  getMinMaxValue(const Tensor<T>& v) const;
};


// TODO : check all dims, check loop for optiz
template<typename T> bool Clip<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 3);
  const int        shift_in{ in[0]->quantizer };
  const int        shift_min{ in[1]->quantizer };
  const int        shift_max{ in[2]->quantizer };

  swap(*in[0], m_out);

  typename ComputationType<T>::type  cur_min = getMinMaxValue(*in[1]);
  typename ComputationType<T>::type  cur_max = getMinMaxValue(*in[2]);
  
  if constexpr (!std::is_same<T,float>::value) {
    int shift_diff = shift_min - shift_in;
    if (shift_diff > 0)
    {
      ComputationType<T>::quantize(cur_min, shift_diff);
    } else {
      ComputationType<T>::shift_left(cur_min, -shift_diff);
    }

    shift_diff = shift_max - shift_in;
    if (shift_diff > 0)
    {
      ComputationType<T>::quantize(cur_max, shift_diff);
    } else {
      ComputationType<T>::shift_left(cur_max, -shift_diff);
    }
  }

  cur_min = std::max(-ComputationType<T>::max, cur_min);
  cur_max = std::min(cur_max, ComputationType<T>::max);

  for (auto it0 = m_out.begin(); it0 != m_out.end(); ++it0)
  {
    typename ComputationType<T>::type z = *it0;
    z = std::max(cur_min, std::min(z, cur_max));
    COUNTERS(z);
    *it0 = (T)z;
  }

  return true;
}


// data in in[0]
// bias in in[1]
// assume data shape [N,W,H,D]
// assume bias shape [D]
template<typename T> bool Clip<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 3)
    return false;
  SADL_DBG(std::cout << "  - " << in[0]->dims() << std::endl);

  // either broadcast from a tensor of size [n] (use when input is Const) or [1,n]
  // of add if same dimensions
  m_out.resize(in[0]->dims());
  m_limits.resize({2});
  m_initDone = true;
  return true;
}

template<typename T> bool Clip<T>::loadInternal(std::istream &file, Version)
{
  return true;
}

template<typename T> 
typename ComputationType<T>::type  Clip<T>::getMinMaxValue(const Tensor<T>& v) const
{
  return v[0];
}


}   // namespace layers
}   // namespace sadl
