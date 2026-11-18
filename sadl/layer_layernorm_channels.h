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
#include <cmath>
#include <type_traits>
#include <vector>

#include "layer.h"

namespace sadl
{
namespace layers
{
// Layer normalization over the channels of each pixel (the last, NHWC
// dimension), with a per-channel scale and shift:
//   y = (x - mean) / sqrt(var + eps) * weight + bias,  var biased.
// What a "b c h w -> b (h w) c" reshape + LayerNorm(c) + reshape back
// computes (the Restormer blocks of the JPEG AI high operating point);
// tools/onnx_fixups/fuse_hop.py folds that ONNX subgraph into one node.
// Float models only. Parameters: C (int32), eps (float), weight[C], bias[C].
template<typename T> class LayerNormChannels : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool       loadInternal(std::istream &file, Version v) override;
  float              m_eps = 1e-5f;
  std::vector<float> m_weight, m_bias;
};

template<typename T> bool LayerNormChannels<T>::apply(std::vector<Tensor<T> *> &in)
{
  auto scope = makeLayerExecutionScope(m_out, in);
  assert(in.size() == 1);
  assert(in[0]->dims() == m_out.dims());
  swap(*in[0], m_out);
  const int     C      = static_cast<int>(m_weight.size());
  const int64_t pixels = m_out.size() / C;
  T            *p      = m_out.data();
  for (int64_t i = 0; i < pixels; ++i, p += C)
  {
    double sum = 0;
    for (int c = 0; c < C; ++c)
      sum += p[c];
    const float mean = static_cast<float>(sum / C);
    double      sq   = 0;
    for (int c = 0; c < C; ++c)
    {
      const double d = static_cast<float>(p[c]) - mean;
      sq += d * d;
    }
    const float var = static_cast<float>(sq / C);
    const float den = std::sqrt(var + m_eps);
    for (int c = 0; c < C; ++c)
      p[c] = static_cast<T>((static_cast<float>(p[c]) - mean) / den * m_weight[c] + m_bias[c]);
  }
  return true;
}

template<typename T> bool LayerNormChannels<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1)
    return false;
  const Dimensions &d = in[0]->dims();
  if (d[d.size() - 1] != static_cast<int>(m_weight.size()))
  {
    std::cerr << "[ERROR] LayerNormChannels: " << d[d.size() - 1] << " channels, parameters for " << m_weight.size() << std::endl;
    return false;
  }
  m_out.resize(d);
  m_initDone = true;
  return true;
}

template<typename T> bool LayerNormChannels<T>::loadInternal(std::istream &file, Version)
{
  if constexpr (!std::is_same_v<T, float>)
  {
    std::cerr << "[ERROR] LayerNormChannels: float models only" << std::endl;
    return false;
  }
  int32_t C = 0;
  file.read((char *) &C, sizeof(C));
  if (!file || C <= 0)
  {
    std::cerr << "[ERROR] LayerNormChannels: invalid channel count " << C << std::endl;
    return false;
  }
  file.read((char *) &m_eps, sizeof(m_eps));
  m_weight.resize(C);
  m_bias.resize(C);
  file.read((char *) m_weight.data(), C * sizeof(float));
  file.read((char *) m_bias.data(), C * sizeof(float));
  SADL_DBG(std::cout << "  - channels: " << C << ", eps: " << m_eps << std::endl);
  return static_cast<bool>(file);
}

}   // namespace layers
}   // namespace sadl
