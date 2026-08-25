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
#include "layer_matmul.h"

#if __AVX2__
static inline __m256i avx2_clamp_epi32(__m256i x, int32_t int_shift, int32_t act_min, int32_t act_max)
{
  __m256i vmin = _mm256_set1_epi32(act_min);
  __m256i vmax = _mm256_set1_epi32(act_max);
  x            = _mm256_srai_epi32(x, int_shift);
  x            = _mm256_max_epi32(x, vmin);   // x = max(x, act_min)
  x            = _mm256_min_epi32(x, vmax);   // x = min(x, act_max)
  return x;
}

static inline void store_packed_i16_8(int16_t *dst, __m256i v32)
{
  __m128i lo128 = _mm256_castsi256_si128(v32);                 // lanes 0..3 (int32)
  __m128i hi128 = _mm256_extracti128_si256(v32, 1);            // lanes 4..7 (int32)
  __m128i v16x8 = _mm_packs_epi32(lo128, hi128);               // saturating pack to 8xint16
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), v16x8);   // write 8xint16
}

static inline __m256i apply_signed_shift(__m256i x, __m256i vshift)
{
  const __m256i zero = _mm256_setzero_si256();

  __m256i shiftR = _mm256_max_epi32(vshift, zero);
  __m256i shiftL = _mm256_max_epi32(_mm256_sub_epi32(zero, vshift), zero);

  __m256i y = _mm256_srav_epi32(x, shiftR);
  y         = _mm256_sllv_epi32(y, shiftL);
  return y;
}

static inline __m256i leaky_relu_saturate(__m256i x, int32_t act_min, int32_t act_max)
{
  __m256i extra = _mm256_srai_epi32(x, 4);
  __m256i mask  = _mm256_cmpgt_epi32(x, _mm256_setzero_si256());
  __m256i y     = _mm256_blendv_epi8(extra, x, mask);

  __m256i vmin = _mm256_set1_epi32(act_min);
  __m256i vmax = _mm256_set1_epi32(act_max);
  y            = _mm256_max_epi32(y, vmin);
  y            = _mm256_min_epi32(y, vmax);
  return y;
}
#endif

namespace sadl
{
namespace layers
{

template<typename T> class FusedMatMulAdaptQuant : public MatMul<T>
{
public:
  using MatMul<T>::MatMul;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;

  bool leakyReluFused() const { return m_leakyReluFused; }
  void setLeakyReluFused(bool value) { m_leakyReluFused = value; }

  static constexpr int32_t m_activationBitwidth = 14;
  static constexpr int32_t act_max              = (1 << (m_activationBitwidth - 1)) - 1;
  static constexpr int32_t act_min              = -(1 << (m_activationBitwidth - 1));

  bool quantizeInput(std::vector<Tensor<T> *> &in);

  void    setStaticInputBitShifts(const std::vector<int> &v) { m_staticInputBitShifts = v; }
  void    setStaticOutputBitShifts(const std::vector<int> &v) { m_staticOutputBitShifts = v; }
  const std::vector<int32_t> &getStaticInputBitShifts() const { return m_staticInputBitShifts; }
  const std::vector<int32_t> &getStaticOutputBitShifts() const { return m_staticOutputBitShifts; }
  int32_t getStaticInputBitShift(int blockIdx) const { return m_staticInputBitShifts[blockIdx]; }
  int32_t getStaticOutputBitShift(int blockIdx) const { return m_staticOutputBitShifts[blockIdx]; }
  bool    hasStaticInputBitShifts() const { return !m_staticInputBitShifts.empty(); }
  bool    hasStaticOutputBitShifts() const { return !m_staticOutputBitShifts.empty(); }

  void    setMRRSPerOut(const std::vector<int32_t> &mrrs) { m_MRRS_per_out = mrrs; }
  const std::vector<int32_t> &getMRRSPerOut() const { return m_MRRS_per_out; }

protected:
  using MatMul<T>::m_out;
  using MatMul<T>::m_initDone;
  using MatMul<T>::m_q;

  virtual bool loadInternal(std::istream &file, Version v) override;

  template<int NN> bool apply_dim2(std::vector<Tensor<T> *> &in);
#if __AVX2__
  bool apply_dim2_simd8(std::vector<Tensor<T> *> &in) { return apply_dim2<8>(in); }
  bool apply_dim2_simd16(std::vector<Tensor<T> *> &in) { return apply_dim2_simd8(in); }
#endif

#if SPARSE_SUPPORT
  bool apply_sparse_matmul(std::vector<Tensor<T> *> &in);
  bool apply_sparse_pack_matmul(std::vector<Tensor<T> *> &in);
#if __AVX2__
  bool apply_sparse_matmul_simd16(std::vector<Tensor<T> *> &in);
  bool apply_sparse_pack8_matmul_simd16(std::vector<Tensor<T> *> &in);
  bool apply_sparse_pack16_matmul_simd16(std::vector<Tensor<T> *> &in);
#endif
#endif

  std::vector<int>     m_staticInputBitShifts;
  std::vector<int>     m_staticOutputBitShifts;
  std::vector<int32_t> m_MRRS_per_out;
  bool                 m_leakyReluFused = false;
  DUMP_MODEL_EXT;
};

template<> inline bool FusedMatMulAdaptQuant<int16_t>::quantizeInput(std::vector<Tensor<int16_t> *> &in)
{
  Tensor<int16_t> &A{ *in[0] };
  const int        K = A.dims()[1];

  int32_t inputBitShift = 0;

#if __AVX2__
  for (int t = 0; t < K; t += 8)
  {
    __m128i v16 = _mm_loadu_si128((const __m128i *) &A.data()[t]);   // load 8xint16
    __m256i v   = _mm256_cvtepi16_epi32(v16);                        // widen to 8xint32

    inputBitShift = getStaticInputBitShift(t / 8);

    v = avx2_clamp_epi32(v, inputBitShift, act_min, act_max);
    v = _mm256_slli_epi32(v, inputBitShift);
    store_packed_i16_8(&A.data()[t], v);
  }
#else
  std::vector<int32_t> intOut(8, 0);
  for (int t = 0; t < K; t += 8)
  {
    std::copy_n(A.begin() + t, 8, intOut.begin());
    inputBitShift = getStaticInputBitShift(t / 8);

    for (int ii = 0; ii < 8; ii++)
    {
      int32_t v = std::max(std::min(intOut[ii] >> inputBitShift, act_max), act_min);
      v <<= inputBitShift;
      A.data()[t + ii] = static_cast<int16_t>(v);
    }
  }
#endif

  return true;
}

template<typename T> bool FusedMatMulAdaptQuant<T>::apply(std::vector<Tensor<T> *> &in)
{
  (void) in;
  std::cerr << "[ERROR] FusedMatMulAdaptQuant only supports int16" << std::endl;
  return false;
}

#if __AVX2__
template<> inline bool FusedMatMulAdaptQuant<int16_t>::apply_dim2_simd16(std::vector<Tensor<int16_t> *> &in)
{
  using T = int16_t;

  const Tensor<int16_t> &A = *in[0];   // bs x m
  const Tensor<int16_t> &B = *in[1];   //  m x n
  const int              M = A.dims()[0];
  const int              K = A.dims()[1];
  const int              N = B.dims()[1];

  assert(K == B.dims()[0]);
  assert(K % 16 == 0);

  const int16_t *a_ptr = A.data();
  const int16_t *b_ptr = B.data();
  int16_t       *c_ptr = m_out.data();

  const int baseShift = B.quantizer + m_q;

  const auto          &mrrs = m_MRRS_per_out;
  std::vector<int32_t> intOut(8, 0);
  std::vector<int32_t> shiftOut(8, 0);
  int32_t              outputBitShift = 0;

  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      __m256i sum = _mm256_setzero_si256();

      for (int k = 0; k < K; k += 16)
      {
        __m256i a1 = _mm256_loadu_si256((const __m256i *) (a_ptr + i * K + k));
        __m256i b1 = _mm256_loadu_si256((const __m256i *) (b_ptr + j * K + k));

        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(a1, b1));
      }

      sum                                      = _mm256_hadd_epi32(sum, sum);
      sum                                      = _mm256_hadd_epi32(sum, sum);
      typename ComputationType<T>::type result = _mm256_extract_epi32(sum, 0) + _mm256_extract_epi32(sum, 4);

      int shift = baseShift;
      shift -= mrrs[j];
      if (hasStaticOutputBitShifts())
      {
        const int lane = j & 7;
        intOut[lane]   = static_cast<int32_t>(result);
        shiftOut[lane] = shift;
        if (lane == 7)
        {
          __m256i v      = _mm256_loadu_si256((const __m256i *) intOut.data());
          __m256i vshift = _mm256_loadu_si256((const __m256i *) shiftOut.data());
          v              = apply_signed_shift(v, vshift);
          if (leakyReluFused())
            v = leaky_relu_saturate(v, -ComputationType<T>::max, ComputationType<T>::max);

          const int blockIdx = (i * N + (j - 7)) / 8;
          outputBitShift     = getStaticOutputBitShift(blockIdx);

          v = avx2_clamp_epi32(v, outputBitShift, act_min, act_max);
          v = _mm256_slli_epi32(v, outputBitShift);
          store_packed_i16_8(&c_ptr[i * N + (j - 7)], v);
        }
      }
      else
      {
        ComputationType<T>::quantize(result, shift);
        COUNTERS(result);
        SATURATE(result);
        c_ptr[i * N + j] = static_cast<T>(result);
      }
    }
  }
  return true;
}
#endif

template<> template<int NN> inline bool FusedMatMulAdaptQuant<int16_t>::apply_dim2(std::vector<Tensor<int16_t> *> &in)
{
  using T = int16_t;

  const Tensor<int16_t> &A{ *in[0] };
  const Tensor<int16_t> &B{ *in[1] };
  const int              last = A.dims().size() - 1;
  const int              N{ A.dims()[last - 1] };
  const int              H{ (A.dims()[last] / NN) * NN };
  const int              R{ B.dims().back() };
  const int              baseShift  = B.quantizer + m_q;
  const bool             isDim2 = (A.dims().size() == 2);
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
  const auto          &mrrs = m_MRRS_per_out;
  std::vector<int32_t> intOut(8, 0);
  std::vector<int32_t> shiftOut(8, 0);

  for (int b = 0; b < N; ++b)
  {
    const int16_t *aptr = A.data() + H * b;
    for (int t = 0; t < R; ++t)
    {
      typename ComputationType<T>::type x    = 0;
      const int16_t                    *bptr = B.data() + t * H;
      for (int i = 0; i < H; ++i)
      {
        x += (typename ComputationType<T>::type) aptr[i] * bptr[i];
        COUNTERS_MAC(bptr[i]);
      }
      COUNTERS_ACC(x);

      int shift = baseShift;
      shift -= mrrs[t];

      if (hasStaticOutputBitShifts())
      {
        const int lane = t & 7;
        intOut[lane]   = (int32_t) x;
        shiftOut[lane] = shift;
        if (lane == 7 || t == R - 1)
        {
          const int     blockIdx       = (b * R + (t - lane)) / 8;
          const int32_t outputBitShift = getStaticOutputBitShift(blockIdx);
          for (int ii = 0; ii <= lane; ++ii)
          {
            int32_t v = intOut[ii];
            if (shiftOut[ii] >= 0)
              v >>= shiftOut[ii];
            else
              v <<= (-shiftOut[ii]);
            if (leakyReluFused())
            {
              if (v <= 0)
                v >>= 4;
              v = std::max(std::min(v, (int32_t) ComputationType<T>::max), -(int32_t) ComputationType<T>::max);
            }
            v = std::max(std::min(v >> outputBitShift, act_max), act_min);
            v <<= outputBitShift;
            SATURATE(v);
            if (isDim2)
              m_out(b, t - lane + ii) = (int16_t) v;
            else
              m_out(0, b, t - lane + ii) = (int16_t) v;
          }
        }
        continue;
      }

      ComputationType<T>::quantize(x, shift);
      COUNTERS(x);
      SATURATE(x);
      if (isDim2)
        m_out(b, t) = (int16_t) x;
      else
        m_out(0, b, t) = (int16_t) x;
    }
  }
  return true;
}

#if SPARSE_SUPPORT

template<> inline bool FusedMatMulAdaptQuant<int16_t>::apply_sparse_matmul(std::vector<Tensor<int16_t> *> &in)
{
  using T = int16_t;

  const Tensor<int16_t> &A{ *in[0] };
  const Tensor<int16_t> &B{ *in[1] };
  const int              baseShift{ in[1]->quantizer + m_q };
  const int              last = A.dims().size() - 1;
  const int              N{ A.dims()[last - 1] };
  const int              H{ A.dims()[last] };
  const bool             isDim2 = (A.dims().size() == 2);
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
  const auto          &mrrs   = m_MRRS_per_out;
  const int            numOut = (int) B.getNbNonzerosCol().size();
  std::vector<int32_t> intOut(8, 0);
  std::vector<int32_t> shiftOut(8, 0);

  uint32_t offset_data = 0;
  for (int b = 0; b < N; ++b)
  {
    const int16_t *aptr   = A.data() + H * b;
    int            outIdx = 0;
    for (const auto &nb_nonzero: B.getNbNonzerosCol())
    {
      typename ComputationType<T>::type x = 0;
      for (uint16_t k = 0; k < nb_nonzero; ++k, ++offset_data)
      {
        uint16_t j = B.getIndices()[offset_data];
        x += (typename ComputationType<T>::type) aptr[j] * B.getDataSparse()[offset_data];
        COUNTERS_MAC(B.getDataSparse()[offset_data]);
      }
#if __AVX2__
      if (std::is_same<T, int16_t>::value)
        offset_data += ((((nb_nonzero + 15) >> 4) << 4) - nb_nonzero);   // pad to 16
#endif
      COUNTERS_ACC(x);

      int shift = baseShift;
      shift -= mrrs[outIdx];

      if (hasStaticOutputBitShifts())
      {
        const int lane = outIdx & 7;
        intOut[lane]   = (int32_t) x;
        shiftOut[lane] = shift;
        if (lane == 7 || outIdx == numOut - 1)
        {
          const int     blockIdx       = (b * numOut + (outIdx - lane)) / 8;
          const int32_t outputBitShift = getStaticOutputBitShift(blockIdx);
          for (int ii = 0; ii <= lane; ++ii)
          {
            int32_t v = intOut[ii];
            if (shiftOut[ii] >= 0)
              v >>= shiftOut[ii];
            else
              v <<= (-shiftOut[ii]);
            if (leakyReluFused())
            {
              if (v <= 0)
                v >>= 4;
              v = std::max(std::min(v, (int32_t) ComputationType<T>::max), -(int32_t) ComputationType<T>::max);
            }
            v = std::max(std::min(v >> outputBitShift, act_max), act_min);
            v <<= outputBitShift;
            SATURATE(v);
            if (isDim2)
              m_out(b, outIdx - lane + ii) = (int16_t) v;
            else
              m_out(0, b, outIdx - lane + ii) = (int16_t) v;
          }
        }
        outIdx++;
        continue;
      }

      ComputationType<T>::quantize(x, shift);
      COUNTERS(x);
      SATURATE(x);
      if (isDim2)
        m_out(b, outIdx) = (int16_t) x;
      else
        m_out(0, b, outIdx) = (int16_t) x;
      outIdx++;
    }
  }
  return true;
}

template<> inline bool FusedMatMulAdaptQuant<int16_t>::apply_sparse_pack_matmul(std::vector<Tensor<int16_t> *> &in)
{
  using T = int16_t;

  const Tensor<int16_t> &A{ *in[0] };
  const Tensor<int16_t> &B{ *in[1] };
  const int              baseShift{ in[1]->quantizer + m_q };
  const int              last = A.dims().size() - 1;
  const int              N{ A.dims()[last - 1] };
  const int              H{ A.dims()[last] };

  uint32_t offset_data = 0;
  int      t           = 0;
#if __AVX2__
  const int pad      = 16;
  const int padShift = 4;
#elif __SSE4_2__
  const int pad      = 8;
  const int padShift = 3;
#endif
  const auto          &mrrs   = m_MRRS_per_out;
  std::vector<int32_t> intOut(8, 0);
  std::vector<int32_t> shiftOut(8, 0);
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
  int         packedSparsitySize = B.getPackedSparsitySize();
  const auto *idx                = B.getIndices().data();
  for (int b = 0; b < N; ++b)
  {
    const T *aptr   = A.data() + H * b;
    int      outIdx = 0;
    for (const auto &nb_nonzero: B.getNbNonzerosCol())
    {
      typename ComputationType<T>::type x = 0;
      for (auto k = 0; k < nb_nonzero / packedSparsitySize; k++, offset_data += packedSparsitySize)
      {
        uint16_t j = idx[k];
        for (auto p = 0; p < packedSparsitySize; ++p)
        {
          x += (typename ComputationType<T>::type) aptr[j + p] * B.getDataSparse()[offset_data + p];
          COUNTERS_MAC(B.getDataSparse()[offset_data + p]);
        }
      }
      COUNTERS_ACC(x);

      int shift = baseShift;
      shift -= mrrs[outIdx];
      outIdx++;

      if (hasStaticOutputBitShifts())
      {
        const int lane = t & 7;
        intOut[lane]   = (int32_t) x;
        shiftOut[lane] = shift;
        if (lane == 7)
        {
          const int     blockIdx       = (t - 7) / 8;
          const int32_t outputBitShift = getStaticOutputBitShift(blockIdx);
          for (int ii = 0; ii < 8; ++ii)
          {
            int32_t v = intOut[ii];
            if (shiftOut[ii] >= 0)
              v >>= shiftOut[ii];
            else
              v <<= (-shiftOut[ii]);
            if (leakyReluFused())
            {
              if (v <= 0)
                v >>= 4;
              v = std::max(std::min(v, (int32_t) ComputationType<T>::max), -(int32_t) ComputationType<T>::max);
            }
            v = std::max(std::min(v >> outputBitShift, act_max), act_min);
            v <<= outputBitShift;
            SATURATE(v);
            m_out[t - 7 + ii] = (int16_t) v;
          }
        }
      }
      else
      {
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out[t] = (int16_t) x;
      }
      t++;
      idx += nb_nonzero / packedSparsitySize;
#if (__SSE4_2__ || __AVX2__)
      if (std::is_same<T, int16_t>::value)
      {
        int aligning = (((nb_nonzero + (pad - 1)) >> padShift) << padShift) - nb_nonzero;
        idx += aligning / packedSparsitySize;
        offset_data += aligning;
      }
#endif
    }
  }
  return true;
}

#if __AVX2__
template<> inline bool FusedMatMulAdaptQuant<int16_t>::apply_sparse_matmul_simd16(std::vector<Tensor<int16_t> *> &in)
{
  using T = int16_t;

  const Tensor<int16_t> &A{ *in[0] };
  const Tensor<int16_t> &B{ *in[1] };
  const int              baseShift{ in[1]->quantizer + m_q };
  const int              last = A.dims().size() - 1;
  const int              N{ A.dims()[last - 1] };
  const int              H{ A.dims()[last] };
  const bool             isDim2 = (A.dims().size() == 2);

  std::vector<int32_t> intOut(8, 0);
  std::vector<int32_t> shiftOut(8, 0);
  int32_t              outputBitShift = 0;

  const auto &mrrs   = m_MRRS_per_out;
  const int   numOut = (int) B.getNbNonzerosCol().size();

#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
  for (int b = 0; b < N; ++b)
  {
    const int16_t *aptr   = A.data() + H * b;
    const int16_t *bptr   = B.getDataSparse().data();
    const auto    *idx    = B.getIndices().data();
    int            outIdx = 0;

    for (auto nb_nonzero: B.getNbNonzerosCol())
    {
      __m256i s = _mm256_setzero_si256();

      for (int j = 0; j < nb_nonzero; j += 16)
      {
        int16_t eA[16];

        for (int k = 0; k < 16; ++k)
        {
          eA[k] = *(aptr + *idx);
          idx++;
        }

        __m256i a1 = _mm256_loadu_si256((__m256i *) eA);
        __m256i b1 = _mm256_loadu_si256((const __m256i *) bptr);
        __m256i ab = _mm256_madd_epi16(a1, b1);

        s = _mm256_add_epi32(s, ab);

        bptr += 16;
      }

      int shift = baseShift;
      shift -= mrrs[outIdx];

      if (hasStaticOutputBitShifts())
      {
        const int lane = outIdx & 7;
        intOut[lane]   = sum32_int16(s);
        shiftOut[lane] = shift;

        if (lane == 7)
        {
          __m256i v      = _mm256_loadu_si256((const __m256i *) intOut.data());
          __m256i vshift = _mm256_loadu_si256((const __m256i *) shiftOut.data());

          v = apply_signed_shift(v, vshift);

          if (leakyReluFused())
            v = leaky_relu_saturate(v, -ComputationType<T>::max, ComputationType<T>::max);

          const int blockIdx = (b * numOut + (outIdx - 7)) / 8;
          outputBitShift     = getStaticOutputBitShift(blockIdx);

          v = avx2_clamp_epi32(v, outputBitShift, act_min, act_max);
          v = _mm256_slli_epi32(v, outputBitShift);

          int16_t tmp[8];
          store_packed_i16_8(tmp, v);

          for (int ii = 0; ii < 8; ii++)
          {
            if (isDim2)
              m_out(b, outIdx - 7 + ii) = tmp[ii];
            else
              m_out(0, b, outIdx - 7 + ii) = tmp[ii];
          }
        }
      }
      else
      {
        typename ComputationType<T>::type z = sum32_int16(s);
        ComputationType<T>::quantize(z, shift);
        SATURATE(z);
        if (isDim2)
          m_out(b, outIdx) = z;
        else
          m_out(0, b, outIdx) = z;
      }

      outIdx++;
    }

    if (hasStaticOutputBitShifts() && ((outIdx & 7) != 0))
    {
      const int validLanes = outIdx & 7;

      for (int ii = validLanes; ii < 8; ii++)
      {
        intOut[ii]   = 0;
        shiftOut[ii] = 0;
      }

      __m256i v      = _mm256_loadu_si256((const __m256i *) intOut.data());
      __m256i vshift = _mm256_loadu_si256((const __m256i *) shiftOut.data());

      v = apply_signed_shift(v, vshift);

      if (leakyReluFused())
        v = leaky_relu_saturate(v, -ComputationType<T>::max, ComputationType<T>::max);

      const int blockIdx = (b * numOut + (outIdx - validLanes)) / 8;
      outputBitShift     = getStaticOutputBitShift(blockIdx);

      v = avx2_clamp_epi32(v, outputBitShift, act_min, act_max);
      v = _mm256_slli_epi32(v, outputBitShift);

      int16_t tmp[8];
      store_packed_i16_8(tmp, v);

      for (int ii = 0; ii < validLanes; ii++)
      {
        if (isDim2)
          m_out(b, outIdx - validLanes + ii) = tmp[ii];
        else
          m_out(0, b, outIdx - validLanes + ii) = tmp[ii];
      }
    }
  }

  return true;
}

template<> inline bool FusedMatMulAdaptQuant<int16_t>::apply_sparse_pack8_matmul_simd16(std::vector<Tensor<int16_t> *> &in)
{
  using T = int16_t;

  const Tensor<int16_t> &A{ *in[0] };
  const Tensor<int16_t> &B{ *in[1] };
  const int              baseShift{ in[1]->quantizer + m_q };
  const int              last = A.dims().size() - 1;
  const int              N{ A.dims()[last - 1] };
  const int              H{ A.dims()[last] };

  std::vector<int32_t> intOut(8, 0);
  std::vector<int32_t> shiftOut(8, 0);
  int32_t              outputBitShift = 0;
  const auto          &mrrs           = m_MRRS_per_out;

  int t = 0;
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
  for (int b = 0; b < N; ++b)
  {
    const int16_t *aptr = A.data() + H * b;
    const int16_t *bptr = B.getDataSparse().data();
    const auto    *idx  = B.getIndices().data();

    int outIdx = 0;

    for (auto nb_nonzero: B.getNbNonzerosCol())
    {
      __m256i s = _mm256_setzero_si256();

      for (int j = 0; j < nb_nonzero; j += 16)
      {
#ifdef _mm256_loadu2_m128i
        __m256i a1 = _mm256_loadu2_m128i((const __m128i *) (aptr + *(idx + 1)), (const __m128i *) (aptr + *idx));
#else
        __m256i a1 = _mm256_set_m128i(_mm_loadu_si128((const __m128i *) (aptr + *(idx + 1))), _mm_loadu_si128((const __m128i *) (aptr + *idx)));
#endif
        __m256i b1 = _mm256_loadu_si256((const __m256i *) bptr);
        __m256i ab = _mm256_madd_epi16(a1, b1);

        s = _mm256_add_epi32(s, ab);

        bptr += 16;
        idx += 2;
      }

      int shift = baseShift;
      shift -= mrrs[outIdx];
      outIdx++;

      if (hasStaticOutputBitShifts())
      {
        const int lane = t & 7;
        intOut[lane]   = sum32_int16(s);
        shiftOut[lane] = shift;
        if (lane == 7)
        {
          __m256i v      = _mm256_loadu_si256((const __m256i *) intOut.data());
          __m256i vshift = _mm256_loadu_si256((const __m256i *) shiftOut.data());
          v              = apply_signed_shift(v, vshift);
          if (leakyReluFused())
            v = leaky_relu_saturate(v, -ComputationType<T>::max, ComputationType<T>::max);
          const int blockIdx = (t - 7) / 8;
          outputBitShift     = getStaticOutputBitShift(blockIdx);

          v = avx2_clamp_epi32(v, outputBitShift, act_min, act_max);
          v = _mm256_slli_epi32(v, outputBitShift);
          store_packed_i16_8(&m_out[t - 7], v);
        }
      }
      else
      {
        typename ComputationType<T>::type z = sum32_int16(s);
        ComputationType<T>::quantize(z, shift);
        SATURATE(z);
        m_out[t] = z;
      }

      t++;
    }
  }

  return true;
}

template<> inline bool FusedMatMulAdaptQuant<int16_t>::apply_sparse_pack16_matmul_simd16(std::vector<Tensor<int16_t> *> &in)
{
  using T = int16_t;

  const Tensor<int16_t> &A{ *in[0] };
  const Tensor<int16_t> &B{ *in[1] };
  const int              baseShift{ in[1]->quantizer + m_q };
  const int              last = A.dims().size() - 1;
  const int              N{ A.dims()[last - 1] };
  const int              H{ A.dims()[last] };

  std::vector<int32_t> intOut(8, 0);
  std::vector<int32_t> shiftOut(8, 0);
  int32_t              outputBitShift = 0;
  const auto          &mrrs           = m_MRRS_per_out;

  int t = 0;
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
  for (int b = 0; b < N; ++b)
  {
    const int16_t *aptr = A.data() + H * b;
    const int16_t *bptr = B.getDataSparse().data();
    const auto    *idx  = B.getIndices().data();

    int outIdx = 0;

    for (auto nb_nonzero: B.getNbNonzerosCol())
    {
      __m256i s = _mm256_setzero_si256();

      for (int j = 0; j < nb_nonzero; j += 16)
      {
        __m256i a1 = _mm256_loadu_si256((__m256i *) (aptr + *idx));
        __m256i b1 = _mm256_loadu_si256((const __m256i *) bptr);
        __m256i ab = _mm256_madd_epi16(a1, b1);

        s = _mm256_add_epi32(s, ab);

        bptr += 16;
        idx++;
      }

      int shift = baseShift;
      shift -= mrrs[outIdx];
      outIdx++;

      if (hasStaticOutputBitShifts())
      {
        const int lane = t & 7;
        intOut[lane]   = sum32_int16(s);
        shiftOut[lane] = shift;
        if (lane == 7)
        {
          __m256i v      = _mm256_loadu_si256((const __m256i *) intOut.data());
          __m256i vshift = _mm256_loadu_si256((const __m256i *) shiftOut.data());
          v              = apply_signed_shift(v, vshift);
          if (leakyReluFused())
            v = leaky_relu_saturate(v, -ComputationType<T>::max, ComputationType<T>::max);
          const int blockIdx = (t - 7) / 8;
          outputBitShift     = getStaticOutputBitShift(blockIdx);

          v = avx2_clamp_epi32(v, outputBitShift, act_min, act_max);
          v = _mm256_slli_epi32(v, outputBitShift);
          store_packed_i16_8(&m_out[t - 7], v);
        }
      }
      else
      {
        typename ComputationType<T>::type z = sum32_int16(s);
        ComputationType<T>::quantize(z, shift);
        SATURATE(z);
        m_out[t] = z;
      }

      t++;
    }
  }

  return true;
}
#endif   // __AVX2__
#endif   // SPARSE_SUPPORT

template<> inline bool FusedMatMulAdaptQuant<int16_t>::apply(std::vector<Tensor<int16_t> *> &in)
{
  using T    = int16_t;
  auto scope = makeLayerExecutionScope(m_out, in);
  assert(in.size() == 2);
#if __AVX2__
#define MULT8_DIM2 apply_dim2_simd8
#define MULT16_DIM2 apply_dim2_simd16
#else
#define MULT8_DIM2 apply_dim2<8>
#define MULT16_DIM2 apply_dim2<16>
#endif

#if SPARSE_SUPPORT
#if __AVX2__
#define SPARSE_MATMULT apply_sparse_matmul_simd16
#define SPARSE_MATMULT_PACK8 apply_sparse_pack8_matmul_simd16
#define SPARSE_MATMULT_PACK16 apply_sparse_pack16_matmul_simd16
#elif __SSE4_2__
#define SPARSE_MATMULT apply_sparse_matmul
#define SPARSE_MATMULT_PACK8 apply_sparse_pack_matmul
#define SPARSE_MATMULT_PACK16 apply_sparse_pack_matmul
#else
#define SPARSE_MATMULT apply_sparse_matmul
#define SPARSE_MATMULT_PACK8 apply_sparse_pack_matmul
#define SPARSE_MATMULT_PACK16 apply_sparse_pack_matmul
#endif
#endif

  const Tensor<T> &A{ *in[0] };
  const Tensor<T> &B{ *in[1] };
  m_out.quantizer = A.quantizer - m_q;
  assert(m_out.quantizer >= 0);
  assert(in[1]->quantizer + m_q >= 0);
  int dum = A.dims().size();
  if (A.dims().size() - 1 == B.dims().size())
    dum--;
  const int H{ A.dims().back() };

  if (hasStaticInputBitShifts())
  {
    quantizeInput(in);
  }

  switch (dum)
  {
  case 2:
#if SPARSE_SUPPORT
    if (in[1]->isSparsePack16())
      return SPARSE_MATMULT_PACK16(in);
    else if (in[1]->isSparsePack8())
      return SPARSE_MATMULT_PACK8(in);
    else if (in[1]->isSparse())
      return SPARSE_MATMULT(in);
    else
#endif
#if DEBUG_MODEL_ANALYZE
      std::cout << "[ANALYZE] matmult (in):\t" << H << std::endl;
#endif
    if (H % 16 == 0)
      return MULT16_DIM2(in);
    else if (H % 8 == 0)
      return MULT8_DIM2(in);
    else
      return apply_dim2<1>(in);
    break;
  default:
    std::cerr << "Logical error FusedMatMulAdaptQuant::apply(std::vector<Tensor<T> *> &in)" << A.dims() << ' ' << B.dims() << std::endl;
    return false;
  }
}

// ---------------------------------------------------------------------------
// loadInternal : q, leakyFused, static input shifts, static output shifts.
// (init() is inherited from MatMul.)
// ---------------------------------------------------------------------------
template<typename T> bool FusedMatMulAdaptQuant<T>::loadInternal(std::istream &file, Version v)
{
  (void) v;
  file.read((char *) &m_q, sizeof(m_q));
  SADL_DBG(std::cout << "  - q: " << m_q << std::endl);

  uint8_t leakyFused = 0;
  file.read((char *) &leakyFused, sizeof(uint8_t));
  m_leakyReluFused = (leakyFused != 0);

  int32_t inSize = 0;
  file.read((char *) &inSize, sizeof(int32_t));
  m_staticInputBitShifts.resize(inSize);
  for (int i = 0; i < inSize; ++i)
  {
    int16_t x = 0;
    file.read((char *) &x, sizeof(int16_t));
    m_staticInputBitShifts[i] = x;
  }

  int32_t outSize = 0;
  file.read((char *) &outSize, sizeof(int32_t));
  m_staticOutputBitShifts.resize(outSize);
  for (int i = 0; i < outSize; ++i)
  {
    int16_t x = 0;
    file.read((char *) &x, sizeof(int16_t));
    m_staticOutputBitShifts[i] = x;
  }

  int32_t mrrsSize = 0;
  file.read((char *) &mrrsSize, sizeof(int32_t));
  m_MRRS_per_out.resize(mrrsSize);
  for (int i = 0; i < mrrsSize; ++i)
  {
    int16_t x = 0;
    file.read((char *) &x, sizeof(int16_t));
    m_MRRS_per_out[i] = x;
  }

  return true;
}

}   // namespace layers
}   // namespace sadl
