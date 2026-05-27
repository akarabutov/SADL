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
template<typename T>
class GatherElements : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool apply(std::vector<Tensor<T> *> &in) override;
  bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  int        m_axis;
  Dimensions out_dims;

  bool loadInternal(std::istream &file, Version v) override;
  DUMP_MODEL_EXT;

private:
  // Normalize user axis 'a' into [0, r).
  int normAxis(int a, int r) const;

  // Compute row-major strides for given dims.
  void computeStrides(const Dimensions &dims, std::vector<int64_t> &strides) const;

  // Convert linear index to multidimensional coordinate (row-major).
  void unravelIndex(int64_t p, const Dimensions &dims, const std::vector<int64_t> &strides, Dimensions &obuf);

  // Read A[coord] for rank r (1..4 supported here).
  T getND(const Tensor<T> &A, const Dimensions &idx, int r);

  // Write Y[coord] = v for rank r (1..4 supported here).
  void setND(Tensor<T> &Y, const Dimensions &idx, int r, T v);
};

template<typename T>
bool GatherElements<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
  {
    std::cerr << "[ERROR] GatherElements: requires 2 inputs (data, indices), got " << in.size() << std::endl;
    return false;
  }

  const Tensor<T> *data = in[0];
  const Tensor<T> *idxs = in[1];
  if (!data || !idxs)
  {
    std::cerr << "[ERROR] GatherElements: null input tensor\n";
    return false;
  }

  const Dimensions &D = data->dims();
  const Dimensions &I = idxs->dims();

  const int rD = (int) D.size();
  const int rI = (int) I.size();

  if (rD <= 0 || rD > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] GatherElements: invalid data rank\n";
    return false;
  }
  if (rI <= 0 || rI > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] GatherElements: invalid indices rank\n";
    return false;
  }
  if (rD != rI)
  {
    std::cerr << "[ERROR] GatherElements: data and indices rank must match\n";
    return false;
  }
  if ((m_axis < -rD) || (m_axis > (rD - 1)))
  {
    std::cerr << "[ERROR] GatherElements: axis out of data rank\n";
    return false;
  }
  
  for (int i = 0; i < rD; ++i)
  {
    if (D[i] <= 0)
    {
      std::cerr << "[ERROR] GatherElements: non-positive data dim\n";
      return false;
    }
  }
  for (int i = 0; i < rI; ++i)
  {
    if (I[i] <= 0)
    {
      std::cerr << "[ERROR] GatherElements: non-positive indices dim\n";
      return false;
    }
  }

  const int a = normAxis(m_axis, rD);
  if (a < 0 || a >= rD)
  {
    std::cerr << "[ERROR] GatherElements: normalized axis out of range\n";
    return false;
  }

  // Output shape equals indices shape.
  out_dims = I;

  m_out.resize(out_dims);
  m_initDone = true;
  return true;
}

template<typename T>
bool GatherElements<T>::apply(std::vector<Tensor<T> *> &in)
{
  auto scope = makeLayerExecutionScope(m_out, in);
  if (!m_initDone)
  {
    std::cerr << "[ERROR] GatherElements: apply before init\n";
    return false;
  }
  if (in.size() != 2)
  {
    std::cerr << "[ERROR] GatherElements: requires 2 inputs\n";
    return false;
  }

  const Tensor<T> &A  = *in[0];
  const Tensor<T> &IX = *in[1];

  const Dimensions &D = A.dims();
  const Dimensions &I = IX.dims();

  const int r = (int) D.size();
  if ((int) I.size() != r)
  {
    std::cerr << "[ERROR] GatherElements: data and indices rank must match at apply\n";
    return false;
  }

  const int a = normAxis(m_axis, r);
  if (a < 0 || a >= r)
  {
    std::cerr << "[ERROR] GatherElements: normalized axis out of range at apply\n";
    return false;
  }

  for (int i = 0; i < r; ++i)
  {
    if (D[i] <= 0 || I[i] <= 0)
    {
      std::cerr << "[ERROR] GatherElements: non-positive dimension\n";
      return false;
    }
  }

  // Output dims were set to indices dims during init.
  if (out_dims.size() != I.size())
  {
    std::cerr << "[ERROR] GatherElements: output dims rank mismatch\n";
    return false;
  }
  for (int i = 0; i < out_dims.size(); ++i)
  {
    if (out_dims[i] != I[i])
    {
      std::cerr << "[ERROR] GatherElements: output dims differ from indices dims\n";
      return false;
    }
  }

  // Strides for unraveling output/indices coordinates.
  std::vector<int64_t> sO;
  computeStrides(out_dims, sO);

  // Total output elements.
  const int64_t P = m_out.size();
  if (P <= 0)
  {
    std::cerr << "[ERROR] GatherElements: invalid/overflow output size\n";
    return false;
  }

  // Axis length (for bounds check & negative index wrap).
  const int Da = D[a];
  if (Da <= 0)
  {
    std::cerr << "[ERROR] GatherElements: invalid axis size\n";
    return false;
  }

  Dimensions obuf;  // output/indices coord (rank r)
  obuf.resize(r);
  Dimensions jbuf;  // data coord (rank r)
  jbuf.resize(r);

  for (int64_t p = 0; p < P; ++p)
  {
    unravelIndex(p, out_dims, sO, obuf);

    int kk =  static_cast<int>(getND(IX, obuf, r));

    // Normalize negative indices.
    if (kk < 0)
    {
      kk += Da;
    }

    if (kk < 0 || kk >= Da)
    {
      std::cerr << "[ERROR] GatherElements: index value "
                << kk << " out of bounds [0," << (Da - 1) << "]\n";
      return false;
    }

    // Build data coord: copy all, replace axis with kk.
    for (int i = 0; i < r; ++i)
    {
      jbuf[i] = obuf[i];
    }
    jbuf[a] = kk;

    // Fetch and write.
    T v = getND(A, jbuf, r);
    COUNTERS(v);
    setND(m_out, obuf, r, v);
  }

  return true;
}

template<typename T>
bool GatherElements<T>::loadInternal(std::istream &file, Version)
{
  int32_t ax = 0;
  file.read((char *) &ax, sizeof(ax));
  m_axis = (int) ax;
  return true;
}

template<typename T>
int GatherElements<T>::normAxis(int a, int r) const
{
  int b = a;
  if (a < 0)
  {
    b = a + r;
  }
  return b;
}

template<typename T>
void GatherElements<T>::computeStrides(const Dimensions &dims, std::vector<int64_t> &strides) const
{
  const int R = (int) dims.size();
  strides.assign(R, 1);
  if (R >= 2)
  {
    for (int i = R - 2; i >= 0; --i)
    {
      strides[i] = strides[i + 1] * (int64_t) dims[i + 1];
    }
  }
}

template<typename T>
void GatherElements<T>::unravelIndex(int64_t p, const Dimensions &dims, const std::vector<int64_t> &strides, Dimensions &obuf)
{
  const int R = (int) dims.size();
  for (int i = 0; i < R; ++i)
  {
    obuf[i] = (int) ((p / strides[i]) % dims[i]);
  }
}

template<typename T>
T GatherElements<T>::getND(const Tensor<T> &A, const Dimensions &idx, int r)
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

template<typename T>
void GatherElements<T>::setND(Tensor<T> &Y, const Dimensions &idx, int r, T v)
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

}  // namespace layers
}  // namespace sadl
