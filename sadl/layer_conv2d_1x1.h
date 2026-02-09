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
#include <cmath>
#include "layer.h"
#if __AVX2__
#include <immintrin.h>
#endif

namespace sadl
{
namespace layers
{
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1x1
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> template<int s_h, int s_w> void Conv2D<T>::conv2d_1x1_s_dispatch(const Tensor<T> &A, const Tensor<T> &kernel)
{
#if __AVX2__
#define CONV_MOD8 simd8_conv2d_1x1_s_d
#define CONV_MOD16 simd16_conv2d_1x1_s_d
#define CONV_MOD32 simd32_conv2d_1x1_s_d
#define CONV_OUT32 simd32_out32_conv2d_1x1_s
#define CONV_MIXED simd_mixed_conv2d_1x1_s
#define CONV_OUT16_D2 simd16_out16_conv2d_1x1_s_d2
#define CONV_OUT16_D1 simd16_out16_conv2d_1x1_s_d1
#else
#define CONV_MOD8 conv2d_1x1_s_d
#define CONV_MOD16 conv2d_1x1_s_d
#define CONV_MOD32 conv2d_1x1_s_d
#define CONV_OUT32 conv2d_1x1_s
#define CONV_MIXED conv2d_1x1_s
#define CONV_OUT16_D2 conv2d_1x1_s
#define CONV_OUT16_D1 conv2d_1x1_s
#endif

  const int in_D{ A.dims()[3] };
  const int nb_filters{ kernel.dims()[2] };
  if(transformedKernel.size() != 0)
  {
      if(in_D % 2 == 0){
          if(nb_filters % 32 == 0)
             CONV_OUT32<s_h, s_w>(A, kernel);
          else if(nb_filters % 16 == 0)
             CONV_OUT16_D2<s_h, s_w>(A, kernel);
      }
      else{
             CONV_OUT16_D1<s_h, s_w>(A, kernel);
      }
  }
  else
  {
    switch (in_D)
    {
    case 1:
      conv2d_1x1_s_d<1, s_h, s_w>(A, kernel);
      break;
    case 2:
      conv2d_1x1_s_d<2, s_h, s_w>(A, kernel);
      break;
    case 4:
      conv2d_1x1_s_d<4, s_h, s_w>(A, kernel);
      break;
    case 6:
      conv2d_1x1_s_d<6, s_h, s_w>(A, kernel);
      break;
    case 17:
      conv2d_1x1_s_d<17, s_h, s_w>(A, kernel);
      break;
    case 34:
      conv2d_1x1_s_d<34, s_h, s_w>(A, kernel);
      break;
    case 8:
      CONV_MOD8<8, s_h, s_w>(A, kernel);
      break;
    case 16:
      CONV_MOD16<16, s_h, s_w>(A, kernel);
      break;
    case 24:
      CONV_MOD8<24, s_h, s_w>(A, kernel);
      break;
    case 32:
      CONV_MOD32<32, s_h, s_w>(A, kernel);
      break;
    case 48:
      CONV_MOD16<48, s_h, s_w>(A, kernel);
      break;
    case 64:
      CONV_MOD32<64, s_h, s_w>(A, kernel);
      break;
    case 72:
      // better do 64 and than 8
      CONV_MOD8<72, s_h, s_w>(A, kernel);
      break;
    case 96:
      CONV_MOD32<96, s_h, s_w>(A, kernel);
      break;
    case 112:
      CONV_MOD16<112, s_h, s_w>(A, kernel);
      break;
    case 128:
      CONV_MOD32<128, s_h, s_w>(A, kernel);
      break;
    case 144:
      CONV_MOD16<144, s_h, s_w>(A, kernel);
      break;
    case 160:
      CONV_MOD32<160, s_h, s_w>(A, kernel);
      break;
    case 176:
      CONV_MOD16<176, s_h, s_w>(A, kernel);
      break;
    case 192:
      CONV_MOD32<192, s_h, s_w>(A, kernel);
      break;
    case 272:
      CONV_MOD16<272, s_h, s_w>(A, kernel);
      break;
    case 288:
      CONV_MOD32<288, s_h, s_w>(A, kernel);
      break;
    case 384:
      CONV_MOD32<384, s_h, s_w>(A, kernel);
      break;
    case 480:
      CONV_MOD32<480, s_h, s_w>(A, kernel);
      break;
    default:
      if(in_D < 16)
          conv2d_1x1_s<s_h, s_w>(A, kernel);
      else
          CONV_MIXED<s_h, s_w>(A, kernel);
      break;
    }
  }
#undef CONV_MOD8
#undef CONV_MOD16
#undef CONV_MOD32
#undef CONV_OUT32
#undef CONV_MIXED
#undef CONV_OUT16_D1
#undef CONV_OUT16_D2
}

template<typename T> template<int s_h, int s_w> void Conv2D<T>::conv2d_1x1_s(const Tensor<T> &A, const Tensor<T> &kernel)
{
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     in_D{ A.dims()[3] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };

#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv2d_1x1_s inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + m_q;
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter_nb = 0; filter_nb < nb_filters; ++filter_nb)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_d = 0; filter_d < in_D; ++filter_d)
        {
          {
            x += (typename ComputationType<T>::type) A(im_nb, im_i, im_j, filter_d) * kernel(0, 0, filter_nb, filter_d);
            COUNTERS_MAC(kernel(0, 0, filter_nb, filter_d));
          }
        }
        COUNTERS_ACC(x);
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(im_nb, im_i / s_w, im_j / s_h, filter_nb) = static_cast<T>(x);
      }
    }
  }
}

template<typename T> template<int in_D, int s_h, int s_w> void Conv2D<T>::conv2d_1x1_s_d(const Tensor<T> &A, const Tensor<T> &kernel)
{ 
//  if (conv2d_core_alongJ<s_h, s_w,0,0>(A, kernel))
//  {
//    return;
//  }
  
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };

#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv2d_1x1_s_d inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + m_q;
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter_nb = 0; filter_nb < nb_filters; ++filter_nb)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_d = 0; filter_d < in_D; ++filter_d)
        {
          x += (typename ComputationType<T>::type) A(im_nb, im_i, im_j, filter_d) * kernel(0, 0, filter_nb, filter_d);
          COUNTERS_MAC(kernel(0, 0, filter_nb, filter_d));
        }
        COUNTERS_ACC(x);
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
      }
    }
  }
}


#if __AVX2__
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1x1
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
template<> template<int in_D, int s_h, int s_w> inline void Conv2D<float>::simd8_conv2d_1x1_s_d(const Tensor<float> &A, const Tensor<float> &kernel)
{
  static_assert(in_D % 8 == 0, "Should be used with mod8 filters.");
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };
#if DEBUG_SIMD && __AVX512F__
  if (in_D >= 16)
  {
    std::cout << "\n[WARN] suboptimal SIMD8 version conv 1x1 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  constexpr int im_nb = 0;
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m256 s = _mm256_setzero_ps();
        for (int filter_d = 0; filter_d < in_D; filter_d += 8)
        {
          const float *kptr = kernel.addr(0, 0, filter, filter_d);
          const float *aptr = A.addr(im_nb, im_i, im_j, filter_d);
          const __m256 k0   = _mm256_load_ps(kptr);
#if __FMA__
          s = _mm256_fmadd_ps(k0, _mm256_load_ps(aptr), s);
#else
          const __m256 m0 = _mm256_mul_ps(k0, _mm256_load_ps(aptr));
          s               = _mm256_add_ps(s, m0);
          // s + m0; // s = _mm256_hadd_ps(s, m0);
#endif
        }
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = sum8_float(s);
      }
    }
  }
}

#if __AVX512F__
template<> template<int in_D, int s_h, int s_w> inline void Conv2D<float>::simd16_conv2d_1x1_s_d(const Tensor<float> &A, const Tensor<float> &kernel)
{
  static_assert(in_D % 16 == 0, "Should be used with mod16 filters.");
  constexpr int im_nb = 0;
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m512 s = _mm512_setzero_ps();
        for (int filter_d = 0; filter_d < in_D; filter_d += 16)
        {
          const float *kptr = kernel.addr(0, 0, filter, filter_d);
          const float *aptr = A.addr(im_nb, im_i, im_j, filter_d);
          const __m512 k0   = _mm512_load_ps(kptr);
#if __FMA__
          s = _mm512_fmadd_ps(k0, _mm512_load_ps(aptr), s);
#else
          const __m512 m0 = _mm512_mul_ps(k0, _mm512_load_ps(aptr));
          s               = _mm512_add_ps(s, m0);
#endif
        }
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = sum16_float(s);
      }
    }
  }
}
#endif

// int16
template<> template<int in_D, int s_h, int s_w> void Conv2D<int16_t>::simd8_conv2d_1x1_s_d(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{   // should be sse42
  using T = int16_t;
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };
  static_assert(in_D % 8 == 0, "Should be used with mod16 filters.");
#if DEBUG_SIMD && __AVX2__
  if (in_D >= 8)
  {
    std::cout << "\n[WARN] suboptimal SIMD8 version simd8_conv2d_1x1_s_d inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + m_q;
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m128i s = _mm_setzero_si128();
        for (int filter_d = 0; filter_d < in_D; filter_d += 8)
        {
          const __m128i *kptr = (const __m128i *) kernel.addr(0, 0, filter, filter_d);
          const __m128i  k0   = _mm_load_si128(kptr);   // or loadu ?
          const __m128i *aptr = (const __m128i *) A.addr(im_nb, im_i, im_j, filter_d);
          const __m128i  v0   = _mm_load_si128(aptr);

          const __m128i mad0 = _mm_madd_epi16(k0, v0);   // res in si32
          s                  = _mm_add_epi32(s, mad0);
        }
        typename ComputationType<T>::type z = (sum32_int16(s) >> shift);
        SATURATE(z);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<int16_t>(z);
      }
    }
  }
}



template<> template<int in_D, int s_h, int s_w> void Conv2D<int16_t>::simd16_conv2d_1x1_s_d(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  using T = int16_t;
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };
  static_assert(in_D % 16 == 0, "Should be used with mod16 filters.");
#if DEBUG_SIMD && __AVX512BW__
  if (in_D >= 32)
  {
    std::cout << "\n[WARN] suboptimal SIMD16 version simd16_conv2d_1x1_s_d inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + m_q;

  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m256i s = _mm256_setzero_si256();
        for (int filter_d = 0; filter_d < in_D; filter_d += 16)
        {
          const __m256i *kptr = (const __m256i *) kernel.addr(0, 0, filter, filter_d);
          const __m256i  k0   = _mm256_load_si256(kptr);   // or loadu ?
          const __m256i *aptr = (const __m256i *) A.addr(im_nb, im_i, im_j, filter_d);
          const __m256i  v0   = _mm256_load_si256(aptr);

          const __m256i mad0 = _mm256_madd_epi16(k0, v0);   // res in si32
          s                  = _mm256_add_epi32(s, mad0);
        }
        typename ComputationType<T>::type z = (sum32_int16(s) >> shift);
        SATURATE(z);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<int16_t>(z);
      }
    }
  }  
}

template<> template<int s_h, int s_w> void Conv2D<int16_t>::simd_mixed_conv2d_1x1_s(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  using T = int16_t;
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     in_D{ A.dims()[3] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + m_q;
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        int filter_d = 0;
#if __AVX512BW__
        __m512i s512 = _mm512_setzero_si512();
        for (; filter_d < in_D - 31; filter_d += 32)
        {
          const __m512i *kptr = (const __m512i *) kernel.addr(0, 0, filter, filter_d);
          const __m512i  k0   = _mm512_loadu_si512(kptr);
          const __m512i *aptr = (const __m512i *) A.addr(im_nb, im_i, im_j, filter_d);
          const __m512i  v0   = _mm512_loadu_si512(aptr);

          const __m512i mad0 = _mm512_madd_epi16(k0, v0);   // res in si32
          s512                 = _mm512_add_epi32(s512, mad0);
        }
#endif
        __m256i s256 = _mm256_setzero_si256();
        for (; filter_d < in_D - 15; filter_d += 16)
        {
          const __m256i *kptr = (const __m256i *) kernel.addr(0, 0, filter, filter_d);
          const __m256i  k0   = _mm256_loadu_si256(kptr);   // or loadu ?
          const __m256i *aptr = (const __m256i *) A.addr(im_nb, im_i, im_j, filter_d);
          const __m256i  v0   = _mm256_loadu_si256(aptr);

          const __m256i mad0 = _mm256_madd_epi16(k0, v0);   // res in si32
          s256                 = _mm256_add_epi32(s256, mad0);
        }
        __m128i s128 = _mm_setzero_si128();
        for (; filter_d < in_D - 7; filter_d += 8)
        {
          const __m128i *kptr = (const __m128i *) kernel.addr(0, 0, filter, filter_d);
          const __m128i  k0   = _mm_loadu_si128(kptr);   // or loadu ?
          const __m128i *aptr = (const __m128i *) A.addr(im_nb, im_i, im_j, filter_d);
          const __m128i  v0   = _mm_loadu_si128(aptr);

          const __m128i mad0 = _mm_madd_epi16(k0, v0);   // res in si32
          s128                 = _mm_add_epi32(s128, mad0);
        }
#if __AVX512BW__
        typename ComputationType<int32_t>::type z = ((sum32_int16(s256) +  sum32_int16(s128) + _mm512_reduce_add_epi32(s512)));
#else
        typename ComputationType<int32_t>::type z = ((sum32_int16(s256) +  sum32_int16(s128)));
#endif
        for (; filter_d < in_D; ++filter_d)
        {
          z += (typename ComputationType<int32_t>::type) A(im_nb, im_i, im_j, filter_d) * kernel(0, 0, filter, filter_d);
        }
        ComputationType<int32_t>::quantize(z, shift);
        COUNTERS(z);
        SATURATE(z);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(z);
      }
    }
  }
}

template<> template<int s_h, int s_w> void  inline Conv2D<int16_t>::simd16_out16_conv2d_1x1_s_d2(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     in_D{ A.dims()[3] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };
  constexpr int  N = 8, hN = 4;
  assert(nb_filters % 16 == 0);
  assert(in_D % 2 == 0);
  assert(transformedKernel.size() != 0);
#if DEBUG_SIMD && __AVX512BW__
  if (in_D >= 32)
  {
    std::cout << "\n[WARN] suboptimal SIMD16 version simd16_conv2d_1x1_s_d inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
#if !SATURATE_RESULT
  const __m256i mask  = _mm256_set1_epi32(65535);
#endif
  constexpr int im_nb = 0;
  const int  shift = kernel.quantizer + m_q;

  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      const int16_t* a_base = A.addr(im_nb, im_i, im_j, 0);
      const int16_t* k_base = transformedKernel.addr(0, 0, 0, 0);
      for (int filter = 0; filter < nb_filters; filter += 16)
      {
        __m256i s_lo = _mm256_setzero_si256();
        __m256i s_hi = _mm256_setzero_si256();
        int filter_d = 0;
        __m256i k[N], v[hN];
        for (; filter_d < in_D - N + 1; filter_d+=N)
        {
            for (int i=0; i < N; i += 2)
            {
                auto* k_addr = k_base + (filter_d + i) * nb_filters + filter * 2;
                k[i] = _mm256_load_si256((const __m256i*)(k_addr));
                k[i+1] = _mm256_load_si256((const __m256i*)(k_addr + 16));
                v[i>>1] = _mm256_set1_epi32(*(const int32_t*)(a_base + filter_d + i));
            }
            for (int i=0; i < N; i += 2)
            {
                const __m256i mad0 = _mm256_madd_epi16(k[i], v[i>>1]);
                const __m256i mad1 = _mm256_madd_epi16(k[i+1], v[i>>1]);
                s_lo               = _mm256_add_epi32(s_lo, mad0);
                s_hi               = _mm256_add_epi32(s_hi, mad1);
            }
        }
        for (; filter_d < in_D; filter_d+=2)
        {
            auto* k_addr = k_base + filter_d * nb_filters + filter * 2;
            const __m256i k0 = _mm256_load_si256((const __m256i*)(k_addr));
            const __m256i k1 = _mm256_load_si256((const __m256i*)(k_addr + 16));
            const __m256i v0 = _mm256_set1_epi32(*(const int32_t*)(a_base + filter_d));
            const __m256i mad0 = _mm256_madd_epi16(k0, v0);
            const __m256i mad1 = _mm256_madd_epi16(k1, v0);
            s_lo               = _mm256_add_epi32(s_lo, mad0);
            s_hi               = _mm256_add_epi32(s_hi, mad1);
        }
        s_lo = _mm256_srai_epi32(s_lo, shift);
        s_hi = _mm256_srai_epi32(s_hi, shift);

#if SATURATE_RESULT
        const __m256i sf16= _mm256_permute4x64_epi64(_mm256_packs_epi32(s_lo, s_hi), 0xD8);
#else
        const __m256i y0p = _mm256_and_si256(s_lo, mask);
        const __m256i y1p = _mm256_and_si256(s_hi, mask);
        const __m256i sf16= _mm256_permute4x64_epi64(_mm256_packus_epi32(y0p , y1p), 0xD8);
#endif
        _mm256_store_si256((__m256i *) m_out.addr(im_nb, im_i / s_h, im_j / s_w, filter), sf16 );
      }
    }
  }
}

template<> template<int s_h, int s_w> void  inline Conv2D<int16_t>::simd16_out16_conv2d_1x1_s_d1(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     in_D{ A.dims()[3] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };
  constexpr int  N = 8;
  assert(nb_filters % 16 == 0);
  assert(transformedKernel.size() != 0);
#if DEBUG_SIMD && __AVX512BW__
  if (in_D >= 32)
  {
    std::cout << "\n[WARN] suboptimal SIMD16 version simd16_conv2d_1x1_s_d inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
#if !SATURATE_RESULT
  const __m256i mask  = _mm256_set1_epi32(65535);
#endif
  constexpr int im_nb = 0;
  const int  shift = kernel.quantizer + m_q;

  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      const int16_t* a_base = A.addr(im_nb, im_i, im_j, 0);
      const int16_t* k_base = transformedKernel.addr(0, 0, 0, 0);
      for (int filter = 0; filter < nb_filters; filter += 16)
      {
        __m256i s_lo = _mm256_setzero_si256();
        __m256i s_hi = _mm256_setzero_si256();
        int filter_d = 0;
        __m256i k[N], v[N];
        for (; filter_d < in_D - N + 1; filter_d+=N)
        {
            for (int i = 0; i < N; i ++)
            {
                k[i] = _mm256_load_si256((const __m256i*)(k_base + (filter_d + i) * nb_filters + filter));
                v[i] = _mm256_set1_epi16(*(const int16_t*)(a_base + filter_d + i));
            }
            for (int i = 0; i < N; i ++)
            {
                const __m256i mul0_hi = _mm256_mulhi_epi16(k[i], v[i]);
                const __m256i mul0_lo = _mm256_mullo_epi16(k[i], v[i]);

                const __m256i mul0_up_lo = _mm256_unpacklo_epi16(mul0_lo, mul0_hi);
                const __m256i mul0_up_hi = _mm256_unpackhi_epi16(mul0_lo, mul0_hi);
                s_lo                = _mm256_add_epi32(s_lo, mul0_up_lo);
                s_hi                = _mm256_add_epi32(s_hi, mul0_up_hi);
            }
        }
        for (; filter_d < in_D; filter_d++)
        {
            const __m256i k0 = _mm256_load_si256((const __m256i*)(k_base + filter_d * nb_filters + filter));
            const __m256i v0 = _mm256_set1_epi16(*(const int16_t*)(a_base + filter_d));
            const __m256i mul0_hi = _mm256_mulhi_epi16(k0, v0);
            const __m256i mul0_lo = _mm256_mullo_epi16(k0, v0);

            const __m256i mul0_up_lo = _mm256_unpacklo_epi16(mul0_lo, mul0_hi);
            const __m256i mul0_up_hi = _mm256_unpackhi_epi16(mul0_lo, mul0_hi);
            s_lo                = _mm256_add_epi32(s_lo, mul0_up_lo);
            s_hi                = _mm256_add_epi32(s_hi, mul0_up_hi);
        }
        s_lo = _mm256_srai_epi32(s_lo, shift);
        s_hi = _mm256_srai_epi32(s_hi, shift);
#if SATURATE_RESULT
        const __m256i sf16= _mm256_packs_epi32(s_lo, s_hi);
#else
        const __m256i y0p = _mm256_and_si256(s_lo, mask);
        const __m256i y1p = _mm256_and_si256(s_hi, mask);
        const __m256i sf16= _mm256_packus_epi32(y0p , y1p);
#endif
        _mm256_store_si256((__m256i *) m_out.addr(im_nb, im_i / s_h, im_j / s_w, filter), sf16 );
      }
    }
  }
}


#if __AVX512BW__

template<> template<int s_h, int s_w> void  inline Conv2D<int16_t>::simd32_out32_conv2d_1x1_s(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     in_D{ A.dims()[3] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };
  constexpr int  N = 8, hN = 4;
  assert(nb_filters % 32 == 0);
  assert(in_D % 2 == 0);
  assert(transformedKernel.size() != 0);
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + m_q;
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      const int16_t* a_base = A.addr(im_nb, im_i, im_j, 0);
      const int16_t* k_base = transformedKernel.addr(0, 0, 0, 0);
      for (int filter = 0; filter < nb_filters; filter += 32)
      {
        __m512i s_lo = _mm512_setzero_si512();
        __m512i s_hi = _mm512_setzero_si512();
        int filter_d = 0;
        __m512i k[N], v[hN];
        for (; filter_d < in_D - N + 1; filter_d+=N)
        {
            for (int i=0; i < N; i += 2)
            {
                auto* k_addr = k_base + (filter_d + i) * nb_filters + filter * 2;
                k[i] = _mm512_load_si512((const __m512i*)(k_addr));
                k[i+1] = _mm512_load_si512((const __m512i*)(k_addr + 32));
                v[i>>1] = _mm512_set1_epi32(*(const int32_t*)(a_base + filter_d + i));
            }
            for (int i=0; i < N; i += 2)
            {
                const __m512i mad0 = _mm512_madd_epi16(k[i], v[i>>1]);
                const __m512i mad1 = _mm512_madd_epi16(k[i+1], v[i>>1]);
                s_lo               = _mm512_add_epi32(s_lo, mad0);
                s_hi               = _mm512_add_epi32(s_hi, mad1);
            }
        }
        for (; filter_d < in_D; filter_d+=2)
        {
            auto* k_addr = k_base + filter_d * nb_filters + filter * 2;
            const __m512i k0 = _mm512_load_si512((const __m512i*)(k_addr));
            const __m512i k1 = _mm512_load_si512((const __m512i*)(k_addr + 32));
            const __m512i v0 = _mm512_set1_epi32(*(const int32_t*)(a_base + filter_d));
            const __m512i mad0 = _mm512_madd_epi16(k0, v0);
            const __m512i mad1 = _mm512_madd_epi16(k1, v0);
            s_lo               = _mm512_add_epi32(s_lo, mad0);
            s_hi               = _mm512_add_epi32(s_hi, mad1);
        }
#if SATURATE_RESULT
        __m256i r_lo = _mm512_cvtsepi32_epi16(_mm512_srai_epi32(s_lo, shift));
        __m256i r_hi = _mm512_cvtsepi32_epi16(_mm512_srai_epi32(s_hi, shift));
#else
        __m256i r_lo = _mm512_cvtsepi32_epu16(_mm512_srai_epi32(s_lo, shift));
        __m256i r_hi = _mm512_cvtsepi32_epu16(_mm512_srai_epi32(s_hi, shift));
#endif
        _mm256_store_si256((__m256i *) m_out.addr(im_nb, im_i / s_h, im_j / s_w, filter), r_lo );
        _mm256_store_si256((__m256i *) m_out.addr(im_nb, im_i / s_h, im_j / s_w, filter + 16), r_hi );
      }
    }
  }
}

template<> template<int in_D, int s_h, int s_w> void Conv2D<int16_t>::simd32_conv2d_1x1_s_d(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  static_assert(in_D % 32 == 0, "Should be used with mod32 filters.");
  using T = int16_t;
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int half_size_h{ 0 };
  constexpr int half_size_w{ 0 };
  assert(m_pads[0]==0);
  assert(m_pads[1]==0);
  constexpr int     top{ 0 };
  constexpr int     left{ 0 };
  constexpr int     start_h{ half_size_h - top };
  constexpr int     start_w{ half_size_w - left };
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + m_q;
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m512i s = _mm512_setzero_si512();
        for (int filter_d = 0; filter_d < in_D; filter_d += 32)
        {
          const __m512i *kptr = (const __m512i *) kernel.addr(0, 0, filter, filter_d);
          const __m512i  k0   = _mm512_load_si512(kptr);
          const __m512i *aptr = (const __m512i *) A.addr(im_nb, im_i, im_j, filter_d);
          const __m512i  v0   = _mm512_load_si512(aptr);

          const __m512i mad0 = _mm512_madd_epi16(k0, v0);   // res in si32
          s                  = _mm512_add_epi32(s, mad0);
        }
        typename ComputationType<int32_t>::type z = (_mm512_reduce_add_epi32(s) >> shift);
        COUNTERS(z);
        SATURATE(z);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(z);
      }
    }
  }
}
#endif
#endif

}   // namespace layers
}   // namespace sadl
