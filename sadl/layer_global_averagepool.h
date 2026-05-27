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
  int        cnt = 0;       // H*W
  int        divq = 0, q = 0, r = 0;
  int        N = 0, N2 = 0;
  int        divq_bits = 0, acc_bits = 0;
  int        shift_bits = 0, r_shift = 0, q2 = 0, r2 = 0;

  virtual bool loadInternal(std::istream &file, Version v) override;
};

template<typename T> bool GlobalAveragePool<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1) return false;
  if (in[0]->dims().size() != 4) return false;

  const int Nn = in[0]->dims()[0];
  const int H  = in[0]->dims()[1];
  const int W  = in[0]->dims()[2];
  const int C  = in[0]->dims()[3];

  if (H <= 0 || W <= 0 || C <= 0) return false;

  m_out.resize({ Nn, 1, 1, C });
  result.resize(m_out.dims());
  result.acquire(0);
  
  cnt = H * W;

  acc_bits = (int) std::ceil(std::log2((double)cnt));
  N  = (int)sizeof(T)  * 8 - 1;
  N2 = (int)sizeof(T2) * 8 - 1;

  const int minAccuracydivQ = N;
  const int bitdepth_acc    = N + acc_bits;
  const int maxbitdepth_divq = N2 - bitdepth_acc;
  const int minq            = std::max(minAccuracydivQ, maxbitdepth_divq);

  q    = std::min(N2, minq + acc_bits - 1);
  divq = (int)(((1LL << q) + cnt / 2) / cnt);
  divq_bits = (int) std::ceil(std::log2((double)divq));

  shift_bits = divq_bits - std::max(N2 - (acc_bits + N), 0);
  q2         = q - shift_bits;

  r       = (int)((1LL << q) / 2);
  r_shift = (int)((1LL << shift_bits) / 2);
  r2      = (int)((1LL << q2) / 2);

  m_initDone = true;
  return true;
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

  if constexpr (std::is_same<T, float>::value)
  {
    const float inv_cnt = 1.0f / (float)cnt;

    for (int n = 0; n < Nn; ++n)
    {
      float *outp = &m_out(n, 0, 0, 0);
      const float *base = A.data() + (int64_t)n * (int64_t)HW * (int64_t)C;

#if defined(__AVX2__)
      int c = 0;
      for (; c + 8 <= C; c += 8)
      {
        __m256 acc = _mm256_setzero_ps();
        const float *p = base + c;
        for (int hw = 0; hw < HW; ++hw)
        {
          acc = _mm256_add_ps(acc, _mm256_loadu_ps(p));
          p += C;
        }
        acc = _mm256_mul_ps(acc, _mm256_set1_ps(inv_cnt));
        _mm256_storeu_ps(outp + c, acc);
      }
      for (; c < C; ++c)
      {
        float s = 0.0f;
        const float *p = base + c;
        for (int hw = 0; hw < HW; ++hw) { s += *p; p += C; }
        outp[c] = s * inv_cnt;
      }
#else
      for (int c = 0; c < C; ++c)
      {
        float s = 0.0f;
        const float *p = base + c;
        for (int hw = 0; hw < HW; ++hw) { s += *p; p += C; }
        outp[c] = s * inv_cnt;
      }
#endif
    }
    return true;
  }
  else
  {
    result.fill(0);

    for (int n = 0; n < Nn; ++n)
    {
      T2 *sump = &result(n, 0, 0, 0);
      const T *base = A.data() + (int64_t)n * (int64_t)HW * (int64_t)C;

#if defined(__AVX2__)
      if constexpr (std::is_same<T, int16_t>::value && std::is_same<T2, int32_t>::value)
      {
        int c = 0;
        for (; c + 16 <= C; c += 16)
        {
          __m256i acc0 = _mm256_setzero_si256();
          __m256i acc1 = _mm256_setzero_si256();
          const int16_t *p = reinterpret_cast<const int16_t *>(base + c);

          for (int hw = 0; hw < HW; ++hw)
          {
            const __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
            const __m128i lo  = _mm256_extracti128_si256(v16, 0);
            const __m128i hi  = _mm256_extracti128_si256(v16, 1);
            acc0 = _mm256_add_epi32(acc0, _mm256_cvtepi16_epi32(lo));
            acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi16_epi32(hi));
            p += C;
          }

          _mm256_storeu_si256(reinterpret_cast<__m256i *>(sump + c),     acc0);
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(sump + c + 8), acc1);
        }

        for (; c < C; ++c)
        {
          int32_t s = 0;
          const T *p = base + c;
          for (int hw = 0; hw < HW; ++hw) { s += (int32_t)(*p); p += C; }
          sump[c] = (T2)s;
        }
      }
      else
#endif
      {
        for (int c = 0; c < C; ++c)
        {
          T2 s = 0;
          const T *p = base + c;
          for (int hw = 0; hw < HW; ++hw) { s += (T2)(*p); p += C; }
          sump[c] = s;
        }
      }
    }

    if (divq_bits + acc_bits + N <= N2)
    {
      for (int n = 0; n < Nn; ++n)
      {
        for (int c = 0; c < C; ++c)
        {
          T2 num = result(n, 0, 0, c);
          num = (num * divq + r) >> q;
          COUNTERS(num);
          COUNTERS_ACC(num * divq + r);
          SATURATE(num);
          m_out(n, 0, 0, c) = (T)num;
        }
      }
    }
    else
    {
      for (int n = 0; n < Nn; ++n)
      {
        for (int c = 0; c < C; ++c)
        {
          T2 num  = result(n, 0, 0, c);
          T2 temp = (num + r_shift) >> shift_bits;
          num     = (temp * divq + r2) >> q2;
          COUNTERS_ACC(temp * divq + r2);
          COUNTERS(num);
          SATURATE(num);
          m_out(n, 0, 0, c) = (T)num;
        }
      }
    }
    return true;
  }
}

template<typename T> bool GlobalAveragePool<T>::loadInternal(std::istream &, Version) { return true; }

}
}
