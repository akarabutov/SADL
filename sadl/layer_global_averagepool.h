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
#include <cstring>
#include <type_traits>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sadl
{
namespace layers
{

template<typename T> class GlobalAveragePool : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  using T2 = typename ComputationType<T>::type;

  Tensor<T2> result;
  int        cnt = 0;   // H*W
  int        divq = 0, q = 0, r = 0;
  int        N = 0, N2 = 0;
  int        divq_bits = 0, acc_bits = 0;
  int        shift_bits = 0, r_shift = 0, q2 = 0, r2 = 0;

  void accumulateHwOuter(const T *base, int HW, int C, T2 *sump);
  void quantizeSums(const T2 *sump, T *outp, int C);

  virtual bool loadInternal(std::istream &file, Version v) override;
};

template<typename T> bool GlobalAveragePool<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1)
    return false;
  if (in[0]->dims().size() != 4)
    return false;

  const int Nn = in[0]->dims()[0];
  const int H  = in[0]->dims()[1];
  const int W  = in[0]->dims()[2];
  const int C  = in[0]->dims()[3];

  if (H <= 0 || W <= 0 || C <= 0)
    return false;

  m_out.resize({ Nn, 1, 1, C });
  result.resize(m_out.dims());
  result.acquire(0);

  cnt = H * W;

  acc_bits               = (int) std::ceil(std::log2((double) cnt));
  N                      = (int) sizeof(T) * 8 - 1;
  N2                     = (int) sizeof(T2) * 8 - 1;

  const int minAccuracydivQ  = N;
  const int bitdepth_acc     = N + acc_bits;
  const int maxbitdepth_divq = N2 - bitdepth_acc;
  const int minq             = std::max(minAccuracydivQ, maxbitdepth_divq);

  q         = std::min(N2, minq + acc_bits - 1);
  divq      = (int) (((1LL << q) + cnt / 2) / cnt);
  divq_bits = (int) std::ceil(std::log2((double) divq));

  shift_bits = divq_bits - std::max(N2 - (acc_bits + N), 0);
  q2         = q - shift_bits;

  r       = (int) ((1LL << q) / 2);
  r_shift = (int) ((1LL << shift_bits) / 2);
  r2      = (int) ((1LL << q2) / 2);

  m_initDone = true;
  return true;
}

template<typename T> void GlobalAveragePool<T>::accumulateHwOuter(const T *base, int HW, int C, T2 *sump)
{
  std::memset(sump, 0, (size_t) C * sizeof(T2));

  if constexpr (std::is_same<T, float>::value)
  {
    const float *p = reinterpret_cast<const float *>(base);
    for (int hw = 0; hw < HW; ++hw)
    {
      int c = 0;
#if defined(__AVX2__)
      for (; c + 8 <= C; c += 8)
      {
        const __m256 v = _mm256_loadu_ps(p + c);
        _mm256_storeu_ps(reinterpret_cast<float *>(sump + c),
                         _mm256_add_ps(_mm256_loadu_ps(reinterpret_cast<const float *>(sump + c)), v));
      }
#endif
      for (; c < C; ++c)
        sump[c] += (T2) p[c];
      p += C;
    }
    return;
  }

#if defined(__AVX2__)
  if constexpr (std::is_same<T, int16_t>::value && std::is_same<T2, int32_t>::value)
  {
    const int16_t *p = reinterpret_cast<const int16_t *>(base);
    for (int hw = 0; hw < HW; ++hw)
    {
      int c = 0;
      for (; c + 16 <= C; c += 16)
      {
        const __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p + c));
        const __m256i s0  = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sump + c));
        const __m256i s1  = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sump + c + 8));
        const __m128i lo  = _mm256_castsi256_si128(v16);
        const __m128i hi  = _mm256_extracti128_si256(v16, 1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sump + c), _mm256_add_epi32(s0, _mm256_cvtepi16_epi32(lo)));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sump + c + 8), _mm256_add_epi32(s1, _mm256_cvtepi16_epi32(hi)));
      }
      for (; c < C; ++c)
        sump[c] += (int32_t) p[c];
      p += C;
    }
    return;
  }
#endif

  const T *p = base;
  for (int hw = 0; hw < HW; ++hw)
  {
    for (int c = 0; c < C; ++c)
      sump[c] += (T2) p[c];
    p += C;
  }
}

template<typename T> void GlobalAveragePool<T>::quantizeSums(const T2 *sump, T *outp, int C)
{
  if constexpr (std::is_same<T, float>::value)
  {
    const float inv_cnt = 1.0f / (float) cnt;
    for (int c = 0; c < C; ++c)
      outp[c] = (T) (sump[c] * inv_cnt);
  }
  else
  {
    if (divq_bits + acc_bits + N <= N2)
    {
      for (int c = 0; c < C; ++c)
      {
        T2 num = sump[c];
        num    = (num * divq + r) >> q;
        COUNTERS(num);
        COUNTERS_ACC(num * divq + r);
        SATURATE(num);
        outp[c] = (T) num;
      }
    }
    else
    {
      for (int c = 0; c < C; ++c)
      {
        T2 num  = sump[c];
        T2 temp = (num + r_shift) >> shift_bits;
        num     = (temp * divq + r2) >> q2;
        COUNTERS_ACC(temp * divq + r2);
        COUNTERS(num);
        SATURATE(num);
        outp[c] = (T) num;
      }
    }
  }
}

template<typename T> bool GlobalAveragePool<T>::apply(std::vector<Tensor<T> *> &in)
{
  auto scope = makeLayerExecutionScope(m_out, in);
  assert(in.size() == 1);

  const Tensor<T> &A = *in[0];
  assert(A.dims().size() == 4);

  m_out.quantizer = A.quantizer;

  const int Nn = A.dims()[0];
  const int H  = A.dims()[1];
  const int W  = A.dims()[2];
  const int C  = A.dims()[3];
  const int HW = H * W;
  assert(HW == cnt);

  for (int n = 0; n < Nn; ++n)
  {
    const T *base = A.data() + (int64_t) n * (int64_t) HW * (int64_t) C;
    accumulateHwOuter(base, HW, C, &result(n, 0, 0, 0));
    quantizeSums(&result(n, 0, 0, 0), &m_out(n, 0, 0, 0), C);
  }
  return true;
}

template<typename T> bool GlobalAveragePool<T>::loadInternal(std::istream &, Version) { return true; }

}
}
