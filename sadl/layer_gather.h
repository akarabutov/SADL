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
#include <vector>
#include <cstdint>
#include <iostream>

namespace sadl
{
namespace layers
{

template<typename T> class Gather : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool apply(std::vector<Tensor<T> *> &in) override;
  bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  int        m_axis            = 0;
  bool       m_indicesIsScalar = false;
  Dimensions out_dims;

  bool loadInternal(std::istream &file, Version v) override;
  DUMP_MODEL_EXT;

private:
  int  normAxis(int a, int r);   // Normalize user axis 'a' into [0, r)
  T getND(const Tensor<T> &A, const int idx[], int r);   // Read A[coord] for rank r
  void setND(Tensor<T> &Y, const int idx[], int r, T v);   // Write Y[coord] = v for rank r
  void computeStrides(const Dimensions &dims, std::vector<int64_t> &strides);   // Compute row-major strides for given dims
  // Convert linear index to multidimensional coordinate (row-major)
  void unravelIndex(int64_t p, const Dimensions &dims, const std::vector<int64_t> &strides, int *obuf);
  // Convert multidimensional coordinate to linear index (row-major)
  int64_t ravelIndex(const int *coord, const std::vector<int64_t> &strides, int R);
};

template<typename T> bool Gather<T>::init(const std::vector<Tensor<T> *> &in)
{
  // Expect two inputs: data and indices
  if (in.size() != 2)
  {
    std::cerr << "[ERROR] Gather: requires 2 inputs (data, indices), got " << in.size() << std::endl;
    return false;
  }

  // Data dims
  const Dimensions &D = in[0]->dims();
  const int         r = (int) D.size();
  if (r <= 0 || r > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] Gather: invalid data rank\n";
    return false;
  }
  for (int i = 0; i < r; ++i)
  {
    if (D[i] <= 0)
    {
      std::cerr << "[ERROR] Gather: non-positive data dim\n";
      return false;
    }
  }

  const Tensor<T> *IX = in[1];
  if (!IX)
  {
    std::cerr << "[ERROR] Gather: indices input is null\n";
    return false;
  }

  if ((m_axis < -r) || (m_axis > (r - 1)))
  {
    std::cerr << "[ERROR] Gather: axis out of data rank\n";
    return false;
  }

  const Dimensions &I  = IX->dims();
  const int         ir = (int) I.size();
  if (ir <= 0 || ir > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] Gather: invalid indices rank (must be ≥1)\n";
    return false;
  }
  for (int i = 0; i < ir; ++i)
  {
    if (I[i] <= 0)
    {
      std::cerr << "[ERROR] Gather: non-positive indices dim\n";
      return false;
    }
  }

  const int a = normAxis(m_axis, r);

  // Output rank (depends on m_indicesIsScalar)
  const int R = m_indicesIsScalar ? (r - 1) : (r + ir - 1);
  if (R <= 0 || R > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] Gather: output rank exceeds MaxDim=" << Dimensions::MaxDim << std::endl;
    return false;
  }

  // Compose output dims
  out_dims.resize(R);
  if (!m_indicesIsScalar)
  {
    // O = D[:a] + I[:] + D[a+1:]
    for (int i = 0; i < a; ++i)
    {
      out_dims[i] = D[i];
    }
    for (int i = 0; i < ir; ++i)
    {
      out_dims[a + i] = I[i];
    }
    for (int i = a + 1, t = a + ir; i < r; ++i, ++t)
    {
      out_dims[t] = D[i];
    }
  }
  else
  {
    // O = D[:a] + D[a+1:]
    for (int i = 0; i < a; ++i)
    {
      out_dims[i] = D[i];
    }
    for (int i = a + 1, t = a; i < r; ++i, ++t)
    {
      out_dims[t] = D[i];
    }
  }

  m_out.resize(out_dims);
  m_initDone = true;
  return true;
}

template<typename T> bool Gather<T>::apply(std::vector<Tensor<T> *> &in)
{
  auto scope = makeLayerExecutionScope(m_out, in);
  if (!m_initDone)
  {
    std::cerr << "[ERROR] Gather: apply before init\n";
    return false;
  }
  if (in.size() != 2)
  {
    std::cerr << "[ERROR] Gather: requires 2 inputs\n";
    return false;
  }

  m_out.quantizer     = in[0]->quantizer;
  const Tensor<T> &A  = *in[0];
  const Tensor<T> &IX = *in[1];

  const Dimensions &D = A.dims();
  const Dimensions &I = IX.dims();

  const int r  = (int) D.size();
  const int ir = (int) I.size();
  const int a  = normAxis(m_axis, r);

  // Check expected output rank again
  const int R_expected = m_indicesIsScalar ? (r - 1) : (r + ir - 1);
  const int R_actual   = (int) out_dims.size();
  if (R_expected != R_actual)
  {
    std::cerr << "[ERROR] Gather: output rank mismatch\n";
    return false;
  }

  // Axis length
  const int Da = D[a];
  if (Da <= 0)
  {
    std::cerr << "[ERROR] Gather: invalid gathered axis size\n";
    return false;
  }

  // Strides
  std::vector<int64_t> sD, sI, sO;
  computeStrides(D, sD);
  computeStrides(I, sI);
  computeStrides(out_dims, sO);

  // indices size
  int64_t KI = in[1]->size();
  if (KI <= 0)
  {
    std::cerr << "[ERROR] Gather: indices size overflow\n";
    return false;
  }
  
  // Read & normalize indices
  std::vector<int> idx_norm((size_t) KI);
  
  int tcoord[Dimensions::MaxDim];
  for (int t = 0; t < KI; ++t)
  {
    unravelIndex(t, I, sI, tcoord);
    int k = static_cast<int>(getND(IX, tcoord, ir));
    if (k < 0)
    {
      k += Da;
    }
    if (k < 0 || k >= Da)
    {
      std::cerr << "[ERROR] Gather: index value " << (long long) k << " out of bounds [0," << (Da - 1) << "]" << std::endl;
      return false;
    }
    idx_norm[(size_t) t] = k;
  }
  
  // If scalar mode, runtime indices must contain exactly one element
  if (m_indicesIsScalar && KI != 1)
  {
    std::cerr << "[ERROR] Gather: indicesIsScalar=1 but indices size != 1\n";
    return false;
  }

  // Total output elements
  int64_t P = m_out.size();
  if (P <= 0)
  {
    std::cerr << "[ERROR] Gather: output size overflow\n";
    return false;
  }

  int obuf[Dimensions::MaxDim];   // output coord
  int jbuf[Dimensions::MaxDim];   // data   coord
  int tbuf[Dimensions::MaxDim];   // indices coord (only used in non-scalar path)

  if (!m_indicesIsScalar)
  {
    // Standard path: O = D[:a] + I[:] + D[a+1:]
    for (int64_t p = 0; p < P; ++p)
    {
      unravelIndex(p, out_dims, sO, obuf);

      for (int i = 0; i < a; ++i)
      {
        jbuf[i] = obuf[i];
      }
      for (int i = 0; i < ir; ++i)
      {
        tbuf[i] = obuf[a + i];
      }
      for (int i = a + 1, t = a + ir; i < r; ++i, ++t)
      {
        jbuf[i] = obuf[t];
      }

      const int64_t linI = ravelIndex(tbuf, sI, ir);
      const int     k    = idx_norm[(size_t) linI];
      jbuf[a]            = k;

      T v = getND(A, jbuf, r);
      COUNTERS(v);
      SATURATE(v);
      setND(m_out, obuf, R_actual, v);
    }
  }
  else
  {
    // Scalar path: O = D[:a] + D[a+1:]; use the single index k
    const int k = idx_norm[0];
    for (int64_t p = 0; p < P; ++p)
    {
      unravelIndex(p, out_dims, sO, obuf);

      for (int i = 0; i < a; ++i)
      {
        jbuf[i] = obuf[i];
      }
      jbuf[a] = k;
      for (int i = a + 1; i < r; ++i)
      {
        jbuf[i] = obuf[i - 1];
      }

      T v = getND(A, jbuf, r);
      COUNTERS(v);
      SATURATE(v);
      setND(m_out, obuf, R_actual, v);
    }
  }

  return true;
}

template<typename T> bool Gather<T>::loadInternal(std::istream &file, Version)
{
  int32_t ax = 0;
  file.read((char *) &ax, sizeof(ax));
  m_axis = (int) ax;

  file.read((char *) &m_indicesIsScalar, sizeof(m_indicesIsScalar));
  return true;
}

template<typename T> int Gather<T>::normAxis(int a, int r)
{
  int b = a;
  if (b < 0)
  {
    b = a + r;
  }
  return b;
}

template<typename T> void Gather<T>::computeStrides(const Dimensions &dims, std::vector<int64_t> &strides)
{
  const int R = (int) dims.size();
  strides.assign(R, 1);
  for (int i = R - 2; i >= 0; --i)
  {
    strides[i] = strides[i + 1] * (int64_t) dims[i + 1];
  }
}

template<typename T> void Gather<T>::unravelIndex(int64_t p, const Dimensions &dims, const std::vector<int64_t> &strides, int *obuf)
{
  const int R = (int) dims.size();
  for (int i = 0; i < R; ++i)
  {
    obuf[i] = (int) ((p / strides[i]) % dims[i]);
  }
}

template<typename T> int64_t Gather<T>::ravelIndex(const int *coord, const std::vector<int64_t> &strides, int R)
{
  int64_t sum = 0;
  for (int i = 0; i < R; ++i)
  {
    sum += (int64_t) coord[i] * strides[i];
  }
  return sum;
}

template<typename T> T Gather<T>::getND(const Tensor<T> &A, const int idx[], int r)
{
  switch (r)
  {
  case 1:
    return A(idx[0]);
  case 2:
    return A(idx[0], idx[1]);
  case 3:
    return A(idx[0], idx[1], idx[2]);
  case 4:
    return A(idx[0], idx[1], idx[2], idx[3]);
  default:
    std::cerr << "[ERROR] GatherElements: The requested dimension exceeds the maximum dimension supported by Tensor operator" << std::endl;
    throw std::runtime_error("unsupported tensor rank");
  }
}

template<typename T> void Gather<T>::setND(Tensor<T> &Y, const int idx[], int r, T v)
{
  switch (r)
  {
  case 1:
    Y(idx[0]) = v;
    break;
  case 2:
    Y(idx[0], idx[1]) = v;
    break;
  case 3:
    Y(idx[0], idx[1], idx[2]) = v;
    break;
  case 4:
    Y(idx[0], idx[1], idx[2], idx[3]) = v;
    break;
  default:
    std::cerr << "[ERROR] GatherElements: The requested dimension exceeds the maximum dimension supported by Tensor operator" << std::endl;
    throw std::runtime_error("unsupported tensor rank");
  }
}

}   // namespace layers
}   // namespace sadl
