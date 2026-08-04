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

template<typename T> class Placeholder : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  ExecMode execMode() const override { return ExecMode::Copy; }
  int        quantizer() const { return m_q; }
  Dimensions dims()      const { return m_dims; }

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
  int          m_q    = -1000;
  Dimensions   m_dims;
};

template<typename T> bool Placeholder<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 1);
  std::copy(in[0]->begin(), in[0]->end(), m_out.begin());
  m_out.quantizer = (m_q >= 0) ? m_q : in[0]->quantizer;
  m_out.setOnDevice(false);
  return true;
}

template<typename T> bool Placeholder<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1) return false;
  m_out.resize(in[0]->dims());
  m_dims     = in[0]->dims();
  m_initDone = true;
  return true;
}

template<typename T> bool Placeholder<T>::loadInternal(std::istream &file, Version)
{
  int32_t x = 0;
  file.read(reinterpret_cast<char *>(&x), sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim) return false;
  m_dims.resize(x);
  file.read(reinterpret_cast<char *>(m_dims.begin()), sizeof(int) * x);
  if (m_dims.size() == 1)
  {
    x = m_dims[0];
    m_dims.resize(2);
    m_dims[0] = 1;
    m_dims[1] = x;
  }
  file.read(reinterpret_cast<char *>(&m_q), sizeof(m_q));
  return true;
}

} 
}  
