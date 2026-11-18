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
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "layer.h"

namespace sadl
{
namespace layers
{
// Transposed ("channel") multi-head self-attention: the core of the
// Restormer MDTA block as the JPEG AI high operating point uses it (TAM's
// Attention, between its dwconv and project_out). The input x [N,H,W,3C]
// (NHWC) is chunked into q, k, v of C channels each; with c = C / heads, for
// each head h, over the channel rows of that head (all H*W positions):
//   q, k <- L2-normalized over the positions: x / max(||x||, eps)
//   A    =  softmax_rows(q k^T * temperature[h])   (c x c)
//   out  =  A v                                    ([N,H,W,C])
// tools/onnx_fixups/fuse_hop.py folds that ONNX subgraph (dynamic Reshapes,
// ReduceL2, MatMul, Softmax) into one node. Float models only.
// Parameters: heads (int32), eps (float), temperature[heads].
template<typename T> class ChannelAttention : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return false; }

protected:
  virtual bool       loadInternal(std::istream &file, Version v) override;
  float              m_eps = 1e-12f;
  std::vector<float> m_temperature;
};

template<typename T> bool ChannelAttention<T>::apply(std::vector<Tensor<T> *> &in)
{
  auto scope = makeLayerExecutionScope(m_out, in);
  assert(in.size() == 1);
  const Dimensions &d     = in[0]->dims();
  const int         N     = d[0];
  const int64_t     P     = static_cast<int64_t>(d[1]) * d[2];   // positions
  const int         C3    = d[3];
  const int         C     = C3 / 3;
  const int         heads = static_cast<int>(m_temperature.size());
  const int         c     = C / heads;

  std::vector<float>  q(static_cast<size_t>(c) * P), k(q.size()), v(q.size());
  std::vector<double> a(static_cast<size_t>(c) * c);
  // The c channel rows of one head starting at channel0, L2-normalized over
  // the positions for q and k.
  auto gather = [&](const T *x, int channel0, std::vector<float> &dst, bool normalize) {
    for (int i = 0; i < c; ++i)
    {
      float *row = &dst[static_cast<size_t>(i) * P];
      double sq  = 0;
      for (int64_t p = 0; p < P; ++p)
      {
        row[p] = static_cast<float>(x[p * C3 + channel0 + i]);
        sq += static_cast<double>(row[p]) * row[p];
      }
      if (!normalize)
        continue;
      const float norm = std::max(static_cast<float>(std::sqrt(sq)), m_eps);
      for (int64_t p = 0; p < P; ++p)
        row[p] /= norm;
    }
  };
  for (int n = 0; n < N; ++n)
  {
    const T *x   = in[0]->data() + n * P * C3;
    T       *out = m_out.data() + n * P * C;
    for (int h = 0; h < heads; ++h)
    {
      gather(x, h * c, q, true);
      gather(x, C + h * c, k, true);
      gather(x, 2 * C + h * c, v, false);
      for (int i = 0; i < c; ++i)
      {
        const float *qi = &q[static_cast<size_t>(i) * P];
        double      *ai = &a[static_cast<size_t>(i) * c];
        double       mx = -INFINITY;
        for (int j = 0; j < c; ++j)
        {
          const float *kj = &k[static_cast<size_t>(j) * P];
          double       s  = 0;
          for (int64_t p = 0; p < P; ++p)
            s += static_cast<double>(qi[p]) * kj[p];
          ai[j] = static_cast<double>(static_cast<float>(s) * m_temperature[h]);
          mx    = std::max(mx, ai[j]);
        }
        double sum = 0;
        for (int j = 0; j < c; ++j)
        {
          ai[j] = std::exp(ai[j] - mx);
          sum += ai[j];
        }
        for (int j = 0; j < c; ++j)
          ai[j] = static_cast<float>(ai[j] / sum);
      }
      for (int64_t p = 0; p < P; ++p)
        for (int i = 0; i < c; ++i)
        {
          const double *ai = &a[static_cast<size_t>(i) * c];
          double        s  = 0;
          for (int j = 0; j < c; ++j)
            s += ai[j] * v[static_cast<size_t>(j) * P + p];
          out[p * C + h * c + i] = static_cast<T>(s);
        }
    }
  }
  return true;
}

template<typename T> bool ChannelAttention<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1)
    return false;
  Dimensions d = in[0]->dims();
  if (d.size() != 4 || d[3] % (3 * static_cast<int>(m_temperature.size())) != 0)
  {
    std::cerr << "[ERROR] ChannelAttention: expected an NHWC input with 3 * heads * c channels, got " << d << std::endl;
    return false;
  }
  d[3] /= 3;
  m_out.resize(d);
  m_initDone = true;
  return true;
}

template<typename T> bool ChannelAttention<T>::loadInternal(std::istream &file, Version)
{
  if constexpr (!std::is_same_v<T, float>)
  {
    std::cerr << "[ERROR] ChannelAttention: float models only" << std::endl;
    return false;
  }
  int32_t heads = 0;
  file.read((char *) &heads, sizeof(heads));
  if (!file || heads <= 0)
  {
    std::cerr << "[ERROR] ChannelAttention: invalid head count " << heads << std::endl;
    return false;
  }
  file.read((char *) &m_eps, sizeof(m_eps));
  m_temperature.resize(heads);
  file.read((char *) m_temperature.data(), heads * sizeof(float));
  SADL_DBG(std::cout << "  - heads: " << heads << ", eps: " << m_eps << std::endl);
  return static_cast<bool>(file);
}

}   // namespace layers
}   // namespace sadl
