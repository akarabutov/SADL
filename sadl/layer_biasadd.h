/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2023, ITU/ISO/IEC
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
#include "layer_add.h"

namespace sadl
{
namespace layers
{
template<typename T> class BiasAdd : public Add<T>
{
public:
  using Add<T>::Add;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
};

template<typename T> bool BiasAdd<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  if (in[0] == in[1])
  {
    std::cerr << "  input aliasing" << std::endl;
    return false;
  }

  const int shift = in[0]->quantizer - in[1]->quantizer;
  swap(*in[0], m_out);
  // adapt output width to second input (which are the bias) in order to be able to rescale as desired the input
  m_out.quantizer = in[1]->quantizer;

  if (shift < 0)
  {
    if (in[0]->dims() == in[1]->dims())
    {
#if DEBUG_MODEL_ANALYZE
      std::cout << "\n[ANALYZE] add (in):\t" << m_out.size() << std::endl;
#endif
      for (auto it0 = m_out.begin(), it1 = in[1]->begin(); it0 != m_out.end(); ++it0, ++it1)
      {
        typename ComputationType<T>::type z = *it0;
        ComputationType<T>::shift_left(z, -shift);
        //        ComputationType<T>::quantize(z, shift);
        z += *it1;
        COUNTERS_ACC(z);
        COUNTERS(z);
        SATURATE(z);
        *it0 = static_cast<T> (z);
      }
    }
    else
    {
      if (in[1]->size() == 1)
      {   // ie in[0]->dims().size() == 1? happen if in[1] is a Const
        const Tensor<T> &B     = *in[1];
        const T          value = B[0];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << m_out.size() << std::endl;
#endif
        for (auto &x: m_out)
        {
          typename ComputationType<T>::type z = x;
          ComputationType<T>::shift_left(z, -shift);
          z += value;
          COUNTERS_ACC(z);
          COUNTERS(z);
          SATURATE(z);
          x = static_cast<T>(z);
        }
      }
      else if (in[0]->dims().size() == 2)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << H << std::endl;
#endif

#if __AVX2__
      if constexpr(std::is_same_v<T, int16_t>)
      {
        if (H % 16 == 0) 
        {
#if SATURATE_RESULT          
          const __m256i     max   = _mm256_set1_epi16(32767);
          const __m256i     min   = _mm256_set1_epi16(-32768);
#endif
          for (int n = 0; n < N; ++n) {
            T *a_ptr = &m_out[n];
            const T *b_ptr = B.data();
            for (int i = 0; i < H; i += 16, a_ptr+=16, b_ptr+=16)
            {
              __m256i a = _mm256_slli_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_ptr)), -shift);
              __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr));
              __m256i result = _mm256_adds_epi16(a, b);
#if SATURATE_RESULT
              result = _mm256_min_epi16(_mm256_max_epi16(result, min), max);
#endif
              _mm256_store_si256((__m256i *) a_ptr, result);
            }
          }
          return true;
        }
      }
#endif
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
          {
            typename ComputationType<T>::type z = m_out(n, i);
            ComputationType<T>::shift_left(z, -shift);
            z += B[i];
            COUNTERS_ACC(z);
            COUNTERS(z);
            SATURATE(z);
            m_out(n, i) = static_cast<T>(z);
          }
      }
      else if (in[0]->dims().size() == 3)
      {
        const Tensor<T> &B = *in[1];
        const int        N = in[0]->dims()[0];
        const int        H = in[0]->dims()[1];
        const int        W = in[0]->dims()[2];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << W << std::endl;
#endif
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
            {
              typename ComputationType<T>::type z = m_out(n, i, j);
              ComputationType<T>::shift_left(z, -shift);
              z += B[j];
              COUNTERS_ACC(z);
              COUNTERS(z);
              SATURATE(z);
              m_out(n, i, j) = static_cast<T>(z);
            }
      }
      else if (in[0]->dims().size() == 4)
      {
        const Tensor<T> &B = *in[1];
        const int        N = in[0]->dims()[0];
        const int        H = in[0]->dims()[1];
        const int        W = in[0]->dims()[2];
        const int        K = in[0]->dims()[3];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << K << std::endl;
#endif
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
#if __AVX2__
        if constexpr (std::is_same_v<T, int16_t>)
        {
          if (K % 16 == 0)
          {
#if !SATURATE_RESULT
            const __m256i     mask  = _mm256_set1_epi32(65535);
#endif       
            const T* b_base = B.data();
            for (int n = 0; n < N; ++n)
            {
              for (int i = 0; i < H; ++i) 
              {
                for (int j = 0; j < W; ++j)
                {
                  T       *a_ptr = &m_out(n, i, j, 0);
                  const T *b_ptr = b_base;
      
                  for (int k = 0; k < K; k += 16, a_ptr += 16, b_ptr += 16)
                  {
                    __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_ptr));
                    __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr));

                    __m128i a_first8 = _mm256_castsi256_si128(a16);
                    __m128i a_last8  = _mm256_extracti128_si256(a16, 1);
                    __m128i b_first8 = _mm256_castsi256_si128(b16);
                    __m128i b_last8  = _mm256_extracti128_si256(b16, 1);

                    // int16_t -> int32_t
                    __m256i a_first8_32 = _mm256_cvtepi16_epi32(a_first8);
                    __m256i a_last8_32  = _mm256_cvtepi16_epi32(a_last8);
                    __m256i b_first8_32 = _mm256_cvtepi16_epi32(b_first8);
                    __m256i b_last8_32  = _mm256_cvtepi16_epi32(b_last8);

                    a_first8_32 = _mm256_slli_epi32(a_first8_32, -shift);
                    a_last8_32  = _mm256_slli_epi32(a_last8_32,  -shift);
    
                    // z += B[k]
                    __m256i r_first8_32 = _mm256_add_epi32(a_first8_32, b_first8_32);
                    __m256i r_last8_32  = _mm256_add_epi32(a_last8_32,  b_last8_32);
  
                    //  Align with SATURATE() of the scalar implementation.
#if SATURATE_RESULT
                    // int32_t -> int16_t SATURATE in _mm256_packs_epi32
                    __m256i r16 = _mm256_packs_epi32(r_first8_32, r_last8_32);
#else
                    const __m256i y0p = _mm256_and_si256(r_first8_32, mask);
                    const __m256i y1p = _mm256_and_si256(r_last8_32, mask); 
                    __m256i r16 = _mm256_packus_epi32(y0p , y1p);
#endif  
                    
                    r16 = _mm256_permute4x64_epi64(r16, _MM_SHUFFLE(3, 1, 2, 0));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_ptr), r16);
                  }
                }
              }
            }
            return true;
          }
          
          else if (K % 8 == 0)
          {
#if !SATURATE_RESULT
            const __m128i mask = _mm_set1_epi32(65535);
#endif
            const T* b_base = B.data();
            for (int n = 0; n < N; ++n)
            {
              for (int i = 0; i < H; ++i)
              {
                for (int j = 0; j < W; ++j)
                {
                  T       *a_ptr = &m_out(n, i, j, 0);
                  const T *b_ptr = b_base;
      
                  for (int k = 0; k < K; k += 8, a_ptr += 8, b_ptr += 8)
                  {
                    __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_ptr));
                    __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr));

                    // int16_t -> int32_t
                    __m256i a32 = _mm256_cvtepi16_epi32(a16);
                    __m256i b32 = _mm256_cvtepi16_epi32(b16);

                    a32 = _mm256_slli_epi32(a32, -shift);
                    __m256i r32 = _mm256_add_epi32(a32, b32);

                    __m128i r32_lo = _mm256_castsi256_si128(r32);          
                    __m128i r32_hi = _mm256_extracti128_si256(r32, 1);
                    // Align with SATURATE() of the scalar implementation.
#if SATURATE_RESULT
                    __m128i r16_128 = _mm_packs_epi32(r32_lo, r32_hi);  // SATURATE in _mm_packs_epi32
#else
                    const __m128i y0p = _mm_and_si128(r32_lo, mask);
                    const __m128i y1p = _mm_and_si128(r32_hi, mask);
                    __m128i r16_128 = _mm_packus_epi32(y0p , y1p); 
#endif              
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(a_ptr), r16_128);

                  }
                }
              }
            }
            return true;
          }
        }
#endif // __AVX2__

        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
              for (int k = 0; k < K; ++k)
              {
                typename ComputationType<T>::type z = m_out(n, i, j, k);
                ComputationType<T>::shift_left(z, -shift);
                z += B[k];
                COUNTERS_ACC(z);
                COUNTERS(z);
                SATURATE(z);
                m_out(n, i, j, k) = static_cast<T>(z);
              }
      }
    }
  }
  else
  {
    if (in[0]->dims() == in[1]->dims())
    {
#if DEBUG_MODEL_ANALYZE
      std::cout << "\n[ANALYZE] add (in):\t" << m_out.size() << std::endl;
#endif
      for (auto it0 = m_out.begin(), it1 = in[1]->begin(); it0 != m_out.end(); ++it0, ++it1)
      {
        typename ComputationType<T>::type z = *it0;
        ComputationType<T>::quantize(z, shift);
        z += *it1;
        COUNTERS_ACC(z);
        COUNTERS(z);
        SATURATE(z);
        *it0 = static_cast<T>(z);
      }
    }
    else
    {
      if (in[1]->size() == 1)
      {   // for constant
        const Tensor<T> &B     = *in[1];
        const T          value = B[0];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << m_out.size() << std::endl;
#endif
        for (auto &x: m_out)
        {
          typename ComputationType<T>::type z = x;
          ComputationType<T>::quantize(z, shift);
          z += value;
          COUNTERS_ACC(z);
          COUNTERS(z);
          SATURATE(z);
          x = static_cast<T>(z);
        }
      }
      else if (in[0]->dims().size() == 2)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << H << std::endl;
#endif
#if __AVX2__
        if constexpr(std::is_same_v<T, int16_t>)
        {
          if (H % 16 == 0) 
          {
#if SATURATE_RESULT            
            const __m256i     max   = _mm256_set1_epi16(32767);
            const __m256i     min   = _mm256_set1_epi16(-32768);
#endif
            for (int n = 0; n < N; ++n) {
              T *a_ptr = &m_out[n];
              const T *b_ptr = B.data();
              for (int i = 0; i < H; i += 16, a_ptr+=16, b_ptr+=16)
              {
                __m256i a = _mm256_srai_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_ptr)), shift);
                __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr));
                __m256i result = _mm256_adds_epi16(a, b);
#if SATURATE_RESULT
                result = _mm256_min_epi16(_mm256_max_epi16(result, min), max);
#endif
                _mm256_store_si256((__m256i *) a_ptr, result);
              }
            }
            return true;
          }
        }
#endif
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
          {
            typename ComputationType<T>::type z = m_out(n, i);
            ComputationType<T>::quantize(z, shift);
            z += B[i];
            COUNTERS_ACC(z);
            COUNTERS(z);
            SATURATE(z);
            m_out(n, i) = static_cast<T>(z);
          }
      }
      else if (in[0]->dims().size() == 3)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
        const int W = in[0]->dims()[2];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << W << std::endl;
#endif
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
            {
              typename ComputationType<T>::type z = m_out(n, i, j);
              ComputationType<T>::quantize(z, shift);
              z += B[j];
              COUNTERS_ACC(z);
              COUNTERS(z);
              SATURATE(z);
              m_out(n, i, j) = static_cast<T>(z);
            }
      }
      else if (in[0]->dims().size() == 4)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
        const int W = in[0]->dims()[2];
        const int K = in[0]->dims()[3];
#if DEBUG_MODEL_ANALYZE
      std::cout << "\n[ANALYZE] add (in):\t" << K << std::endl;
#endif
#if __AVX2__
        if constexpr (std::is_same_v<T, int16_t>)
        {
          if (K % 16 == 0)
          {
#if !SATURATE_RESULT
            const __m256i     mask  = _mm256_set1_epi32(65535);
#endif
            for (int n = 0; n < N; ++n)
            {
              for (int i = 0; i < H; ++i)
              {
                for (int j = 0; j < W; ++j)
                {
                  T       *a_ptr = &m_out(n, i, j, 0);
                  const T *b_ptr = B.data();

                  for (int k = 0; k < K; k += 16, a_ptr += 16, b_ptr += 16)
                  {
                    __m256i a = _mm256_srai_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_ptr)), shift);
                    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr));
                    
#if SATURATE_RESULT
                    __m256i result = _mm256_adds_epi16(a, b); // SATURATE in _mm256_adds_epi16
#else
                    __m128i a_first8 = _mm256_castsi256_si128(a);
                    __m128i a_last8  = _mm256_extracti128_si256(a, 1);
                    __m128i b_first8 = _mm256_castsi256_si128(b);
                    __m128i b_last8  = _mm256_extracti128_si256(b, 1);

                    // int16_t -> int32_t
                    __m256i a_first8_32 = _mm256_cvtepi16_epi32(a_first8);
                    __m256i a_last8_32  = _mm256_cvtepi16_epi32(a_last8);
                    __m256i b_first8_32 = _mm256_cvtepi16_epi32(b_first8);
                    __m256i b_last8_32  = _mm256_cvtepi16_epi32(b_last8);

                    // add on int32_t
                    __m256i r_first8_32 = _mm256_add_epi32(a_first8_32, b_first8_32);
                    __m256i r_last8_32  = _mm256_add_epi32(a_last8_32,  b_last8_32);
                    // mask 
                    const __m256i y0p = _mm256_and_si256(r_first8_32, mask);
                    const __m256i y1p = _mm256_and_si256(r_last8_32, mask); 
                    __m256i result = _mm256_packus_epi32(y0p , y1p);
                    result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
#endif
                    _mm256_store_si256(reinterpret_cast<__m256i *>(a_ptr), result);
                  }
                }
              }
            }
            return true;
          }
          else if (K % 8 == 0)
          {
#if !SATURATE_RESULT
            const __m128i mask = _mm_set1_epi32(65535);
#endif
            const T *b_base = B.data();
            
            for (int n = 0; n < N; ++n)
            {
              for (int i = 0; i < H; ++i)
              {
                for (int j = 0; j < W; ++j)
                {
                  T       *a_ptr = &m_out(n, i, j, 0);
                  const T *b_ptr = b_base;
                  for (int k = 0; k < K; k += 8, a_ptr += 8, b_ptr += 8)
                  {
                    __m128i a = _mm_srai_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a_ptr)), shift);
                    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr));                  
                     
#if SATURATE_RESULT
                    __m128i result = _mm_adds_epi16(a, b); // saturate in _mm_adds_epi16 
#else
                    // int16_t -> int32_t
                    __m256i a32 = _mm256_cvtepi16_epi32(a);
                    __m256i b32 = _mm256_cvtepi16_epi32(b);
                    // add on int32_t
                    __m256i r32 = _mm256_add_epi32(a32, b32);

                    __m128i r32_lo = _mm256_castsi256_si128(r32);          
                    __m128i r32_hi = _mm256_extracti128_si256(r32, 1);
                    const __m128i y0p = _mm_and_si128(r32_lo, mask);
                    const __m128i y1p = _mm_and_si128(r32_hi, mask);
                    __m128i result = _mm_packus_epi32(y0p , y1p);                   
#endif
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(a_ptr), result);
                  }
                }
              }
            }
            return true;
          }
        }
#endif // __AVX2__
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
              for (int k = 0; k < K; ++k)
              {
                typename ComputationType<T>::type z = m_out(n, i, j, k);
                ComputationType<T>::quantize(z, shift);
                z += B[k];
                COUNTERS_ACC(z);
                COUNTERS(z);
                SATURATE(z);
                m_out(n, i, j, k) = static_cast<T>(z);
              }
      }
    }
  }
  return true;
}

template<typename T> bool BiasAdd<T>::init(const std::vector<Tensor<T> *> &in)
{
  // convervative check
  if (in.size() != 2)
    return false;
  if (in[1]->dims().size() != 1)
    return false;
  if (in[0]->dims()[in[0]->dims().size() - 1] != in[1]->dims()[0])
    return false;

  m_out.resize(in[0]->dims());
  m_initDone = true;
  return true;
}

}   // namespace layers
}   // namespace sadl
