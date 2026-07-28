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
#include <vector>
#include <cstdint>
#include <array>
#include <iostream>
#include <string>
#include <algorithm>
#include "tensor.h"
#include "options.h"

namespace sadl_cuda
{
template<typename> class Model;  // fwd

namespace layers
{

struct OperationType
{
  enum Type
  {
    Copy               = -1,
    Const              = 1,
    Placeholder        = 2,
    Identity           = 3,
    BiasAdd            = 4,
    MaxPool            = 5,
    MatMul             = 6,
    Reshape            = 7,
    Relu               = 8,
    Conv2D             = 9,
    Add                = 10,
    Concat             = 11,
    Mul                = 12,
    Maximum            = 13,
    LeakyRelu          = 14,
    Transpose          = 15,
    Flatten            = 16,
    Shape              = 17,
    Expand             = 18,
    Conv2DTranspose    = 19,
    Slice              = 20,
    PReLU              = 21,
    ScatterND          = 22,
    GridSample         = 23,
    Resize             = 24,
    Compare            = 25,
    Where              = 26,
    Minimum            = 27,
    AveragePool        = 28,
    ReduceMean         = 29,
    Tile               = 30,
    Pad                = 31,
    Floor              = 32,
    Sub                = 33,
    Abs                = 34,
    Depth2Space        = 35,
    Gather             = 36,
    GatherElements     = 37,
    GlobalAveragePool  = 38,
    BitShift           = 39,
    Clip               = 40,
    OperationTypeCount = 41,
    OperationExperimentalStart = 65535,
    Conv2DBiasPreLU            = 65536,
    Conv2DBias                 = 65537,
    OperationExperimentalEnd   = 65538,
  };
};

struct TensorInternalType
{
  enum Type { Int32 = 0, Float = 1, Int16 = 2 };
};

enum class ExecMode { Noop, Copy, Cpu, Cuda };

// for dbg
inline const char *execModeName(ExecMode m)
{
  switch (m)
  {
  case ExecMode::Noop: return "noop";
  case ExecMode::Copy: return "copy";
  case ExecMode::Cpu:  return "cpu";
  case ExecMode::Cuda: return "cuda";
  }
  return "?";
}

template<typename T> class Layer
{
public:
  using Id         = int32_t;
  using value_type = T;

  Layer(Id iid, OperationType::Type iop) : m_id(iid), m_op(iop) {}
  virtual ~Layer() = default;

  virtual bool apply(std::vector<Tensor<T> *> &in)      = 0;
  virtual bool init(const std::vector<Tensor<T> *> &in) = 0;
  virtual bool isGpuLayer()                       const { return false; }
  virtual ExecMode execMode()                     const { return isGpuLayer() ? ExecMode::Cuda : ExecMode::Cpu; }
  bool         load(std::istream &file, Version v);

  bool                    initDone()       const { return m_initDone; }
  Tensor<T> &             output()               { return m_out; }
  const std::string &     name()           const { return m_name; }
  Id                      id()             const { return m_id; }
  const std::vector<Id> & inputsId()       const { return m_inputs_id; }
  OperationType::Type     op()             const { return m_op; }
  void                    replaceInputId(Id old_id, Id new_id)
  {
    std::replace(m_inputs_id.begin(), m_inputs_id.end(), old_id, new_id);
  }

protected:
  bool         loadPrefix(std::istream &file, Version v);
  virtual bool loadInternal(std::istream &file, Version v) = 0;

  Tensor<T>                 m_out;
  const Id                  m_id;
  const OperationType::Type m_op;
  std::string               m_name;
  std::vector<Id>           m_inputs_id;
  bool                      m_initDone = false;
  bool                      m_torun    = true;

  template<typename> friend class sadl_cuda::Model;
};

template<typename T> bool Layer<T>::load(std::istream &file, Version v)
{
  return loadPrefix(file, v) && loadInternal(file, v);
}

template<typename T> bool Layer<T>::loadPrefix(std::istream &file, Version v)
{
  m_initDone = false;
  int32_t L  = 0;
  file.read(reinterpret_cast<char *>(&L), sizeof(int32_t));
  constexpr int maxLength = 2048;
  assert(L > 0 && L + 1 < maxLength);
  char s[maxLength];
  file.read(s, L);
  s[L]   = '\0';
  m_name = s;
  SADL_DBG(std::cout << "  - name: " << m_name << '\n');

  file.read(reinterpret_cast<char *>(&L), sizeof(int32_t));
  assert(L >= 0 && L < 8);
  m_inputs_id.resize(L);
  for (auto &x : m_inputs_id)
    file.read(reinterpret_cast<char *>(&x), sizeof(int32_t));
  return static_cast<bool>(file);
}

}  
}  
