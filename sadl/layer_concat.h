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
#include <cstring>   // for std::memcpy
#include "layer.h"

namespace sadl
{
namespace layers
{
template<typename T> class Concat : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;
  int m_axis = 0;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
};

template<typename T> bool Concat<T>::apply(std::vector<Tensor<T> *> &in)
{
  auto scope = makeLayerExecutionScope(m_out, in);
  assert(in.size() >= 3);
  const int nb_in = (int) in.size() - 1;                                                                        // without axis inputs
  const int axis  = m_axis;
  assert(axis >= 0 && axis < in[0]->dims().size());
  int shift[16] = {};
  int qmin      = in.front()->quantizer;
  if (!std::is_same<T, float>::value)
  {
    assert(nb_in < 16);
    for (int i = 0; i < nb_in; ++i)
    {
      if (in[i]->quantizer < qmin)
        qmin = in[i]->quantizer;
    }
    for (int i = 0; i < nb_in; ++i)
    {
      shift[i] = in[i]->quantizer - qmin;
    }
  }
  m_out.quantizer   = qmin;   // adapt output width to last input

  // Compute inner (product of dims after axis) and outer (product of dims before axis).
  int inner = 1;
  for (int k = axis + 1; k < in[0]->dims().size(); ++k)
   inner *= m_out.dims()[k];
  int outer = 1;
  for (int k = 0; k < axis; ++k) 
   outer *= m_out.dims()[k];
  const int out_axis_len = m_out.dims()[axis];
  // Main loop: per outer-block, append nb_in contiguous chunks along axis in order.
  for (int o = 0; o < outer; ++o)
  {
    int write_axis_cursor = 0;
    for (int n = 0; n < nb_in; ++n)
    {
      const int len_a      = in[n]->dims()[axis];
      const int copy_elems = len_a * inner;
      const int in_off     = o * copy_elems;
      const int out_off    = o * (out_axis_len * inner) + write_axis_cursor * inner;

      if (std::is_same<T, float>::value)
      {
        // Fast path for float: raw memcpy of a contiguous block.
        std::memcpy(m_out.data() + out_off, in[n]->data() + in_off, copy_elems * sizeof(T));
      }
      else
      {
        // Integer path: down-quantize to qmin before storing.
        const T *src = in[n]->data() + in_off;
        T *dst       = m_out.data()   + out_off;
        bool simd = false;
#if __AVX2__
        if constexpr (std::is_same_v<T, int16_t>)
        {
          const int s = shift[n];
          if (copy_elems % 16 == 0)
          {
            for (int e = 0; e < copy_elems; e += 16)
            {
              __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + e));
              __m256i r = _mm256_srai_epi16(v, s);
              _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + e), r);
            }
            simd = true;
          }
          else if (copy_elems % 8 == 0)
          {
            for (int e = 0; e < copy_elems; e += 8)
            {
              __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + e));
              __m128i r = _mm_srai_epi16(v, s);
              _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + e), r);
            }
            simd = true;
          }
        }
#endif // __AVX2__
        if (!simd)
        {
          for (int e = 0; e < copy_elems; ++e)
          {
            T z = src[e];
            ComputationType<T>::quantize(z, shift[n]);
            dst[e] = z;
          }
        }
      }
      write_axis_cursor += len_a;
    }
  }
  return true;
}

template<typename T> bool Concat<T>::init(const std::vector<Tensor<T> *> &in)
{
  /*
  The axis of the concatenation can be arbitrary
  */
  if (in.size() < 3)
    return false;
  if (in[0]->dims().size() < 1)
    return false;
  const int axis_idx = (int) in.size() - 1;
  if ((*in[axis_idx]).size() != 1)
    return false;
  int axis = (int)(*in[axis_idx])[0];
  if (axis < 0) 
    axis += in[0]->dims().size();
  if (axis < 0 || axis >= in[0]->dims().size())
    return false;
  m_axis = axis;

  // Check shape compatibility on non-axis dims and accumulate axis length.
  int sum_dim = 0;
  for (int i = 1; i < axis_idx; i++)
  {
    if (in[0]->dims().size() != in[i]->dims().size())
      return false;
    sum_dim += in[i]->dims()[axis];
    for (int k = 0; k < in[0]->dims().size(); ++k)
    {
      if (k == axis) 
       continue;
      if (in[0]->dims()[k] != in[i]->dims()[k])
        return false;
    }
  }
  Dimensions dim = in[0]->dims();
  dim[axis] += sum_dim;
  m_out.resize(dim);
  m_initDone = true;
  return true;
}

template<typename T> bool Concat<T>::loadInternal(std::istream &, Version) { return true; }

}   // namespace layers
}   // namespace sadl
