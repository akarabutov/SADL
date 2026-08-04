/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
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

namespace sadl_cuda
{
namespace layers
{

// Gather: gathers slices from data along axis using indices.
// Output shape = data[:axis] + indices.shape + data[axis+1:]
// If indicesIsScalar=true, the index tensor has one element and the
// axis dimension is dropped: output shape = data[:axis] + data[axis+1:]

template<typename T> class Gather : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    // cpu
    assert(in.size() == 2);
    const Tensor<T> &A  = *in[0];
    const Tensor<T> &IX = *in[1];
    m_out.quantizer = A.quantizer;

    const int r    = static_cast<int>(A.dims().size());
    const int ir   = static_cast<int>(IX.dims().size());
    const int a    = (m_axis < 0) ? m_axis + r : m_axis;
    const int Da   = A.dims()[a];
    const int R    = m_indicesIsScalar ? r - 1 : r + ir - 1;
    const int P    = static_cast<int>(m_out.size());
    const int KI   = static_cast<int>(IX.size());

    // Compute strides
    std::vector<int64_t> sD(r, 1), sI(ir, 1), sO(R, 1);
    for (int i = r  - 2; i >= 0; --i) sD[i] = sD[i+1] * A.dims()[i+1];
    for (int i = ir - 2; i >= 0; --i) sI[i] = sI[i+1] * IX.dims()[i+1];
    for (int i = R  - 2; i >= 0; --i) sO[i] = sO[i+1] * m_out.dims()[i+1];

    // Normalize indices
    std::vector<int> idx_norm(KI);
    for (int t = 0; t < KI; ++t)
    {
      int k = static_cast<int>(IX[t]);
      if (k < 0) k += Da;
      idx_norm[t] = k;
    }

    int obuf[Dimensions::MaxDim];
    int jbuf[Dimensions::MaxDim];
    int tbuf[Dimensions::MaxDim];

    if (!m_indicesIsScalar)
    {
      for (int p = 0; p < P; ++p)
      {
        // unravel output index
        int64_t tmp = p;
        for (int i = 0; i < R; ++i) { obuf[i] = static_cast<int>((tmp / sO[i]) % m_out.dims()[i]); }
        // build data coord
        for (int i = 0; i < a;  ++i) jbuf[i] = obuf[i];
        for (int i = 0; i < ir; ++i) tbuf[i] = obuf[a + i];
        for (int i = a+1, t = a+ir; i < r; ++i, ++t) jbuf[i] = obuf[t];
        // linear index into IX
        int64_t linI = 0;
        for (int i = 0; i < ir; ++i) linI += tbuf[i] * sI[i];
        jbuf[a] = idx_norm[static_cast<int>(linI)];
        // read data
        int64_t linD = 0;
        for (int i = 0; i < r; ++i) linD += jbuf[i] * sD[i];
        m_out[p] = A[static_cast<int>(linD)];
      }
    }
    else
    {
      const int k = idx_norm[0];
      for (int p = 0; p < P; ++p)
      {
        int64_t tmp = p;
        for (int i = 0; i < R; ++i) { obuf[i] = static_cast<int>((tmp / sO[i]) % m_out.dims()[i]); }
        for (int i = 0; i < a;  ++i) jbuf[i] = obuf[i];
        jbuf[a] = k;
        for (int i = a+1; i < r; ++i) jbuf[i] = obuf[i-1];
        int64_t linD = 0;
        for (int i = 0; i < r; ++i) linD += jbuf[i] * sD[i];
        m_out[p] = A[static_cast<int>(linD)];
      }
    }
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    const Dimensions &D  = in[0]->dims();
    const Dimensions &I  = in[1]->dims();
    const int         r  = static_cast<int>(D.size());
    const int         ir = static_cast<int>(I.size());
    const int         a  = (m_axis < 0) ? m_axis + r : m_axis;
    if (a < 0 || a >= r) return false;

    const int R = m_indicesIsScalar ? r - 1 : r + ir - 1;
    if (R <= 0 || R > Dimensions::MaxDim) return false;

    Dimensions dim;
    dim.resize(R);
    if (!m_indicesIsScalar)
    {
      for (int i = 0; i < a;  ++i) dim[i] = D[i];
      for (int i = 0; i < ir; ++i) dim[a + i] = I[i];
      for (int i = a+1, t = a+ir; i < r; ++i, ++t) dim[t] = D[i];
    }
    else
    {
      for (int i = 0; i < a; ++i)   dim[i]   = D[i];
      for (int i = a+1; i < r; ++i) dim[i-1] = D[i];
    }
    m_out.resize(dim);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t ax = 0;
    file.read(reinterpret_cast<char *>(&ax), sizeof(ax));
    m_axis = static_cast<int>(ax);
    file.read(reinterpret_cast<char *>(&m_indicesIsScalar), sizeof(m_indicesIsScalar));
    return true;
  }
  int  m_axis            = 0;
  bool m_indicesIsScalar = false;
};

}  // namespace layers
}  // namespace sadl_cuda
