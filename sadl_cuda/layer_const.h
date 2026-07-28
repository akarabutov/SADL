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

namespace sadl_cuda
{
namespace layers
{

template<typename T> class Const : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override { (void)in; return true; }
  ExecMode execMode() const override { return ExecMode::Noop; }
  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (!in.empty()) return false;
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
  template<typename U> void readTensor(std::istream &file, int64_t n);
};

template<typename T> template<typename U> void Const<T>::readTensor(std::istream &file, int64_t n)
{
  if (std::is_same<T, U>::value)
  {
    file.read(reinterpret_cast<char *>(m_out.data()), sizeof(T) * n);
  }
  else
  {
    std::vector<U> tmp(n);
    file.read(reinterpret_cast<char *>(tmp.data()), sizeof(U) * n);
    for (int64_t k = 0; k < n; ++k)
      m_out[static_cast<int>(k)] = static_cast<T>(tmp[k]);
  }
}

template<typename T> bool Const<T>::loadInternal(std::istream &file, Version v)
{
  int32_t x = 0;
  file.read(reinterpret_cast<char *>(&x), sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim) return false;
  Dimensions d;
  d.resize(x);
  for (int k = 0; k < d.size(); ++k)
  {
    file.read(reinterpret_cast<char *>(&x), sizeof(x));
    d[k] = x;
  }
  if (d.nbElements() >= Tensor<T>::kMaxSize) return false;

  uint8_t isSparse = 0;
  int32_t sizeSparse = 0;
  int16_t packedSparsitySize = 1;
  bool    transposed = false;
  if (v >= Version::sadl05)
  {
    file.read(reinterpret_cast<char *>(&isSparse), sizeof(uint8_t));
    if (isSparse)
    {
      file.read(reinterpret_cast<char *>(&sizeSparse), sizeof(int32_t));
      file.read(reinterpret_cast<char *>(&packedSparsitySize), sizeof(packedSparsitySize));
    }
    file.read(reinterpret_cast<char *>(&transposed), sizeof(transposed));
    m_out.setTransposed(transposed);
  }

  m_out.resize(d);
  int64_t n = d.nbElements();

  file.read(reinterpret_cast<char *>(&x), sizeof(x));
  switch (x)
  {
  case TensorInternalType::Int32:
    file.read(reinterpret_cast<char *>(&m_out.quantizer), sizeof(m_out.quantizer));
    if (isSparse) // put back dense
    {
      std::vector<uint16_t> nbNonzeros(static_cast<size_t>(d[1]));
      std::vector<int32_t>  dataSparse(sizeSparse);
      std::vector<uint16_t> indices(sizeSparse / packedSparsitySize);
      file.read(reinterpret_cast<char *>(nbNonzeros.data()), sizeof(uint16_t) * nbNonzeros.size());
      file.read(reinterpret_cast<char *>(indices.data()),    sizeof(uint16_t) * indices.size());
      file.read(reinterpret_cast<char *>(dataSparse.data()), sizeof(int32_t)  * sizeSparse);
      m_out.fill(T{});
      uint32_t offset_data = 0;
      const auto *idx = indices.data();
      const int R = d[0];
      int idxSparse = 0;
      for (const auto &nb : nbNonzeros)
      {
        int i = 0;
        for (int k = 0; k < nb / packedSparsitySize; ++k)
        {
          int j = idx[k];
          if (i > j) idxSparse += i - j;
          while (i < j) m_out[offset_data + i++] = T{};
          while (i < j + packedSparsitySize)
            m_out[offset_data + i++] = static_cast<T>(dataSparse[idxSparse++]);
        }
        while (i < R) m_out[offset_data + i++] = T{};
        offset_data += R;
        idx += nb / packedSparsitySize;
      }
    }
    else
    {
      readTensor<int32_t>(file, n);
    }
    break;
  case TensorInternalType::Float:
    readTensor<float>(file, n);
    break;
  case TensorInternalType::Int16:
    file.read(reinterpret_cast<char *>(&m_out.quantizer), sizeof(m_out.quantizer));
    if (isSparse)
    {
      std::vector<uint16_t> nbNonzeros(static_cast<size_t>(d[1]));
      std::vector<int16_t>  dataSparse(sizeSparse);
      std::vector<uint16_t> indices(sizeSparse / packedSparsitySize);
      file.read(reinterpret_cast<char *>(nbNonzeros.data()), sizeof(uint16_t) * nbNonzeros.size());
      file.read(reinterpret_cast<char *>(indices.data()),    sizeof(uint16_t) * indices.size());
      file.read(reinterpret_cast<char *>(dataSparse.data()), sizeof(int16_t)  * sizeSparse);
      m_out.fill(T{});
      uint32_t offset_data = 0;
      const auto *idx = indices.data();
      const int R = d[0];
      int idxSparse = 0;
      for (const auto &nb : nbNonzeros)
      {
        int i = 0;
        for (int k = 0; k < nb / packedSparsitySize; ++k)
        {
          int j = idx[k];
          if (i > j) idxSparse += i - j;
          while (i < j) m_out[offset_data + i++] = T{};
          while (i < j + packedSparsitySize)
            m_out[offset_data + i++] = static_cast<T>(dataSparse[idxSparse++]);
        }
        while (i < R) m_out[offset_data + i++] = T{};
        offset_data += R;
        idx += nb / packedSparsitySize;
      }
    }
    else
    {
      readTensor<int16_t>(file, n);
    }
    break;
  default:
    std::cerr << "[ERROR] unknown internal type " << x << '\n';
    return false;
  }
  return true;
}

}  // namespace layers
}  // namespace sadl_cuda
