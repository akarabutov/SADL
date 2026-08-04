/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
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
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <cstdint>
#include <iostream>
#include <string>
#include <algorithm>
#include "layer.h"
#include "layers.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace sadl_cuda
{

template<typename T> std::unique_ptr<layers::Layer<T>> createLayer(int32_t id, layers::OperationType::Type op)
{
  using namespace layers;
  switch (op)
  {
  case OperationType::Copy:             return std::make_unique<Copy<T>>(id, op);
  case OperationType::Const:            return std::make_unique<Const<T>>(id, op);
  case OperationType::Placeholder:      return std::make_unique<Placeholder<T>>(id, op);
  case OperationType::Reshape:          return std::make_unique<Reshape<T>>(id, op);
  case OperationType::Identity:         return std::make_unique<Identity<T>>(id, op);
  case OperationType::MatMul:           return std::make_unique<MatMul<T>>(id, op);
  case OperationType::BiasAdd:          return std::make_unique<BiasAdd<T>>(id, op);
  case OperationType::Conv2D:           return std::make_unique<Conv2D<T>>(id, op);
  case OperationType::Conv2DTranspose:  return std::make_unique<Conv2DTranspose<T>>(id, op);
  case OperationType::Add:              return std::make_unique<Add<T>>(id, op);
  case OperationType::Relu:             return std::make_unique<Relu<T>>(id, op);
  case OperationType::MaxPool:          return std::make_unique<MaxPool<T>>(id, op);
  case OperationType::Mul:              return std::make_unique<Mul<T>>(id, op);
  case OperationType::Concat:           return std::make_unique<Concat<T>>(id, op);
  case OperationType::Maximum:          return std::make_unique<Maximum<T>>(id, op);
  case OperationType::LeakyRelu:        return std::make_unique<LeakyRelu<T>>(id, op);
  case OperationType::Transpose:        return std::make_unique<Transpose<T>>(id, op);
  case OperationType::Flatten:          return std::make_unique<Flatten<T>>(id, op);
  case OperationType::Shape:            return std::make_unique<Shape<T>>(id, op);
  case OperationType::Expand:           return std::make_unique<Expand<T>>(id, op);
  case OperationType::Slice:            return std::make_unique<Slice<T>>(id, op);
  case OperationType::PReLU:            return std::make_unique<PReLU<T>>(id, op);
  case OperationType::ScatterND:        return std::make_unique<ScatterND<T>>(id, op);
  case OperationType::GridSample:       return std::make_unique<GridSample<T>>(id, op);
  case OperationType::Resize:           return std::make_unique<Resize<T>>(id, op);
  case OperationType::Compare:          return std::make_unique<Compare<T>>(id, op);
  case OperationType::Where:            return std::make_unique<Where<T>>(id, op);
  case OperationType::Minimum:          return std::make_unique<Minimum<T>>(id, op);
  case OperationType::AveragePool:      return std::make_unique<AveragePool<T>>(id, op);
  case OperationType::ReduceMean:       return std::make_unique<ReduceMean<T>>(id, op);
  case OperationType::Tile:             return std::make_unique<Tile<T>>(id, op);
  case OperationType::Pad:              return std::make_unique<Pad<T>>(id, op);
  case OperationType::Floor:            return std::make_unique<Floor<T>>(id, op);
  case OperationType::Sub:              return std::make_unique<Sub<T>>(id, op);
  case OperationType::Abs:              return std::make_unique<Abs<T>>(id, op);
  case OperationType::Depth2Space:      return std::make_unique<Depth2Space<T>>(id, op);
  case OperationType::BitShift:         return std::make_unique<BitShift<T>>(id, op);
  case OperationType::Gather:           return std::make_unique<Gather<T>>(id, op);
  case OperationType::GatherElements:   return std::make_unique<GatherElements<T>>(id, op);
  case OperationType::GlobalAveragePool:return std::make_unique<GlobalAveragePool<T>>(id, op);
  case OperationType::Clip:             return std::make_unique<Clip<T>>(id, op);
  case OperationType::Conv2DBiasPreLU:  return std::make_unique<Conv2DBiasPreLU<T>>(id, op);
  case OperationType::Conv2DBias:       return std::make_unique<Conv2DBias<T>>(id, op);
  case OperationType::OperationTypeCount:
  case OperationType::OperationExperimentalStart:
  case OperationType::OperationExperimentalEnd:
    std::cerr << "[ERROR] unknown layer " << op << std::endl;
    exit(-1);
  }
  std::cerr << "[ERROR] unknown layer " << op << std::endl;
  exit(-1);
}

template<typename T> class Model
{
public:
  struct LayerData
  {
    std::unique_ptr<layers::Layer<T>> layer;
    std::vector<Tensor<T> *>          inputs;
  };

  bool             load(std::istream &in);
  bool             init(std::vector<Tensor<T>> &in);
  bool             apply(std::vector<Tensor<T>> &in);
  void             activateOutput(const std::vector<char> &output_activated);

  const Tensor<T> &result(int idx_out = 0) const  {
    if (!m_results_sync)
    {
      for (auto &id: m_ids_output)
      {
        auto &out = getLayer(id).layer->m_out;
        if (out.isGpuActive() && out.isOnDevice())
        {
#if SADL_CUDA_PROFILE
          std::cout << "[PROF] [WARN] " << getLayer(id).layer->id() << " " << layers::opName(getLayer(id).layer->op()) << " Device2Host for output\n";
#endif
          out.d2h();
        }
        m_results_sync = true;
      }
    }

    return getLayer(m_ids_output[idx_out]).layer->output();
  }

  std::vector<Tensor<T>>                            getInputsTemplate() const;
  const std::vector<typename layers::Layer<T>::Id> &getIdsOutput() const { return m_ids_output; }
  size_t                                            nbOutputs() const { return m_ids_output.size(); }
  std::vector<typename layers::Layer<T>::Id>        getLayersId() const;
  const LayerData &                                 getLayer(const typename layers::Layer<T>::Id &id) const;
  LayerData &                                       getLayer(const typename layers::Layer<T>::Id &id);
  Version                                           version() const { return m_version; }
  const std::string &                               info() const { return m_info; }


private:
  std::vector<LayerData>                     m_data;
  int32_t                                    m_nb_inputs = 0;
  static constexpr int                       kMaxLayers  = 8192;
  Version                                    m_version   = Version::unknown;
  std::vector<typename layers::Layer<T>::Id> m_ids_input, m_ids_output;
  bool                                       m_initDone  = false;
  std::string                                m_info;
  mutable bool                               m_results_sync = false;
  void reshapeConv2DFilters();
  void reshapeMatrix();
  void fuseConv2DBiasPreLU();
  void fuseConv2DBias();
  void classifyTensorMemory();
  void activateLayer(typename layers::Layer<T>::Id id);
};


template<typename T> bool Model<T>::load(std::istream &file)
{
  if (!file) { std::cerr << "[ERROR] Pb reading model\n"; return false; }

  m_info.clear();
  m_data.clear();
  m_nb_inputs = 0;
  m_version   = Version::unknown;
  m_ids_input.clear();
  m_ids_output.clear();
  m_initDone  = false;

  char magic[9]; file.read(magic, 8); magic[8] = '\0';
  std::string ms = magic;
  if      (ms == "SADL0002") m_version = Version::sadl02;
  else if (ms == "SADL0003") m_version = Version::sadl03;
  else if (ms == "SADL0004") m_version = Version::sadl04;
  else if (ms == "SADL0005") m_version = Version::sadl05;
  else { std::cerr << "[ERROR] wrong magic " << ms << "\n"; return false; }

  {
    int32_t x = 0;
    file.read(reinterpret_cast<char *>(&x), sizeof(x));
    if (x != layers::TensorInternalType::Int16)
    {
      std::cerr << "[ERROR] sadl_cuda only supports Int16 models\n";
      return false;
    }
  }

  if (m_version >= Version::sadl05)
  {
    int16_t x = 0;
    file.read(reinterpret_cast<char *>(&x), sizeof(x));
    if (x > 0)
    {
      std::vector<char> v(x + 1, 0);
      file.read(v.data(), x);
      m_info = std::string{ v.data() };
    }
  }

  int32_t nb_layers = 0;
  file.read(reinterpret_cast<char *>(&nb_layers), sizeof(nb_layers));
  if (nb_layers <= 0 || nb_layers > kMaxLayers)
  {
    std::cerr << "[ERROR] invalid nb_layers " << nb_layers << "\n";
    return false;
  }
  m_data.clear();
  m_data.resize(nb_layers);

  {
    int32_t nb = 0;
    file.read(reinterpret_cast<char *>(&nb), sizeof(nb));
    m_ids_input.resize(nb);
    file.read(reinterpret_cast<char *>(m_ids_input.data()), sizeof(int32_t) * nb);
    file.read(reinterpret_cast<char *>(&nb), sizeof(nb));
    m_ids_output.resize(nb);
    file.read(reinterpret_cast<char *>(m_ids_output.data()), sizeof(int32_t) * nb);
  }

  if (m_version >= Version::sadl05)
  {
    bool dummy = false;
    file.read(reinterpret_cast<char *>(&dummy), sizeof(dummy));
  }

  for (int k = 0; k < nb_layers; ++k)
  {
    typename layers::Layer<T>::Id id = 0;
    file.read(reinterpret_cast<char *>(&id), sizeof(int32_t));
    int32_t op = 0;
    file.read(reinterpret_cast<char *>(&op), sizeof(int32_t));
    if (!(op > 0 && op < layers::OperationType::OperationTypeCount)
        && !(op > layers::OperationType::OperationExperimentalStart
             && op < layers::OperationType::OperationExperimentalEnd + 1))
    {
      std::cerr << "[ERROR] invalid layer op " << op << "\n";
      return false;
    }
    m_data[k].layer = createLayer<T>(id, static_cast<layers::OperationType::Type>(op));
    m_data[k].inputs.clear();
    if (!m_data[k].layer->load(file, m_version)) { m_data.clear(); return false; }
  }

  if (m_data.empty()) { std::cerr << "[ERROR] no layers\n"; return false; }
  return true;
}


template<typename T> bool Model<T>::init(std::vector<Tensor<T>> &in)
{
  if (m_data.empty()) { std::cerr << "[ERROR] empty model\n"; return false; }
  m_nb_inputs = static_cast<int>(in.size());
  if (m_nb_inputs != static_cast<int>(m_ids_input.size()))
  {
    std::cerr << "[ERROR] inconsistent input count\n"; return false;
  }

  if (m_initDone)
  {
    for (auto &L : m_data) { L.layer->m_initDone = false; L.layer->m_torun = true; }
  }
  else
  {
    reshapeConv2DFilters();
    reshapeMatrix();
  }

  bool ok = true;
  int  ph = 0;
  for (int k = 0; k < static_cast<int>(m_data.size()) && ok; ++k)
  {
    if (m_data[k].layer->op() != layers::OperationType::Placeholder) continue;
    if (ph >= static_cast<int>(in.size())) { std::cerr << "[ERROR] too many placeholders\n"; return false; }
    std::vector<Tensor<T> *> v = { &in[ph++] };
    ok &= m_data[k].layer->init(v);
  }
  if (!ok) return false;

  for (int k = 0; k < static_cast<int>(m_data.size()) && ok; ++k)
  {
    if (m_data[k].layer->op() == layers::OperationType::Placeholder) continue;
    int nb = static_cast<int>(m_data[k].layer->inputsId().size());
    m_data[k].inputs.resize(nb);
    for (int i = 0; i < nb; ++i)
    {
      auto &L = getLayer(m_data[k].layer->inputsId()[i]);
      if (!L.layer->initDone()) { std::cerr << "[ERROR] dependency not init\n"; return false; }
      m_data[k].inputs[i] = &L.layer->output();
    }
    ok &= m_data[k].layer->init(m_data[k].inputs);
    if (!ok) { std::cerr << "[ERROR] init failed layer " << m_data[k].layer->id() << "\n"; return false; }
  }

  fuseConv2DBiasPreLU();
  fuseConv2DBias();
  classifyTensorMemory();
  m_initDone = true;
  return true;
}

template<typename T> bool Model<T>::apply(std::vector<Tensor<T>> &in)
{
  assert(!m_data.empty());
  assert(static_cast<int>(in.size()) == m_nb_inputs);

  bool ok = true;
  int  ph = 0;
  for (int k = 0; k < static_cast<int>(m_data.size()) && ok; ++k)
  {
    if (m_data[k].layer->op() == layers::OperationType::Placeholder && m_data[k].layer->m_torun)
    {
      std::vector<Tensor<T> *> v = { &in[ph++] };
      ok &= m_data[k].layer->apply(v);
      m_data[k].layer->m_out.setOnDevice(false);
#if SADL_CUDA_PROFILE
      std::cout << "[PROF] " << m_data[k].layer->id()
                << " " << layers::opName(m_data[k].layer->op())
                << " " << layers::execModeName(m_data[k].layer->execMode()) << "\n";
#endif
    }
  }

  for (int k = 0; k < static_cast<int>(m_data.size()) && ok; ++k)
  {
    if (m_data[k].layer->op() == layers::OperationType::Placeholder || !m_data[k].layer->m_torun)
      continue;
    if (m_data[k].layer->op() == layers::OperationType::Const)
      continue;

    const bool gpu = m_data[k].layer->isGpuLayer();
#if SADL_CUDA_PROFILE
    int idx=-1;
#endif
    for (auto *inp : m_data[k].inputs)
    {
#if SADL_CUDA_PROFILE
     ++idx;
#endif
      if (!inp->isGpuActive()) continue;
      if (gpu && !inp->isOnDevice()) {
        inp->h2d();
#if SADL_CUDA_PROFILE
        std::cout << "[PROF] [WARN] " << m_data[k].layer->id() << " " << layers::opName(m_data[k].layer->op()) << " Host2Device input["<<idx<<"]\n";
#endif
      }
      else if (!gpu && inp->isOnDevice()) {
        inp->d2h();
#if SADL_CUDA_PROFILE
        std::cout << "[PROF] [WARN] " << m_data[k].layer->id() << " " << layers::opName(m_data[k].layer->op()) << " Device2Host input["<<idx<<"]\n";
#endif

      }
    }

    ok &= m_data[k].layer->apply(m_data[k].inputs);
    if (gpu) SADL_CUDA_LAUNCH_CHECK();
#if SADL_CUDA_PROFILE
    std::cout << "[PROF] " << ((m_data[k].layer->execMode() == layers::ExecMode::Cpu) ? "[WARN] ":"") << m_data[k].layer->id()
              << " " << layers::opName(m_data[k].layer->op())
              << " " << layers::execModeName(m_data[k].layer->execMode()) << "\n";
#endif

    if (gpu) m_data[k].layer->m_out.setOnDevice(true);
    else     m_data[k].layer->m_out.setOnDevice(false);
  }

  m_results_sync = false;
  
  return ok;
}


template<typename T>
typename Model<T>::LayerData &
Model<T>::getLayer(const typename layers::Layer<T>::Id &id)
{
  auto it = std::find_if(m_data.begin(), m_data.end(),
      [id](const LayerData &d) { return d.layer->id() == id; });
  if (it == m_data.end()) { std::cerr << "[ERROR] cannot find layer " << id << "\n"; assert(false); exit(-1); }
  return *it;
}

template<typename T>
const typename Model<T>::LayerData &
Model<T>::getLayer(const typename layers::Layer<T>::Id &id) const
{
  auto it = std::find_if(m_data.begin(), m_data.end(),
      [id](const LayerData &d) { return d.layer->id() == id; });
  if (it == m_data.end()) { std::cerr << "[ERROR] cannot find layer " << id << "\n"; assert(false); exit(-1); }
  return *it;
}

template<typename T>
std::vector<typename layers::Layer<T>::Id> Model<T>::getLayersId() const
{
  std::vector<typename layers::Layer<T>::Id> ids;
  for (const auto &L : m_data) ids.push_back(L.layer->id());
  return ids;
}

template<typename T> void Model<T>::activateLayer(typename layers::Layer<T>::Id id)
{
  auto it = std::find_if(m_data.begin(), m_data.end(),
      [id](const LayerData &d) { return d.layer->id() == id; });
  assert(it != m_data.end());
  if (it->layer->m_torun) return;
  it->layer->m_torun = true;
  for (auto idin : it->layer->m_inputs_id) activateLayer(idin);
}

template<typename T> void Model<T>::activateOutput(const std::vector<char> &output_activated)
{
  assert(!m_ids_output.empty());
  if (output_activated.size() != m_ids_output.size()) { assert(false); exit(-1); }
  for (auto &L : m_data) L.layer->m_torun = false;
  for (int i = 0; i < static_cast<int>(output_activated.size()); ++i)
    if (output_activated[i]) activateLayer(m_ids_output[i]);
}


template<typename T> void Model<T>::reshapeConv2DFilters()
{
  std::vector<int> seen;
  for (auto &v : m_data)
  {
    if (v.layer->op() != layers::OperationType::Conv2D) continue;
    int wid = v.layer->inputsId()[1];
    if (std::find(seen.begin(), seen.end(), wid) != seen.end()) continue;
    seen.push_back(wid);
    auto &L = getLayer(wid);
    if (L.layer->op() != layers::OperationType::Const)
    {
      continue;
    }
    auto &W = L.layer->output();
    Dimensions d = W.dims();
    if (d.size() != 4) { std::cerr << "[ERROR] invalid conv filter dims\n"; exit(-1); }
    std::swap(d[2], d[3]);
    Tensor<T> T2(d); T2.quantizer = W.quantizer;
    for (int i = 0; i < d[0]; ++i)
      for (int j = 0; j < d[1]; ++j)
        for (int k = 0; k < d[2]; ++k)
          for (int l = 0; l < d[3]; ++l)
            T2(i, j, k, l) = W(i, j, l, k);
    swap(W, T2);
  }
}

template<typename T> void Model<T>::reshapeMatrix()
{
  for (auto &v : m_data)
  {
    if (v.layer->op() != layers::OperationType::MatMul) continue;
    auto &R = getLayer(v.layer->inputsId()[1]).layer->output();
    if (R.getTransposed()) continue;
    Dimensions d = R.dims();
    if (d.size() == 2)
    {
      Tensor<T> T2(d); T2.quantizer = R.quantizer;
      for (int i = 0; i < d[0]; ++i)
        for (int j = 0; j < d[1]; ++j)
          T2[j * d[0] + i] = R(i, j);
      swap(R, T2);
      R.setTransposed(true);
    }
  }
}

template<typename T> void Model<T>::fuseConv2DBiasPreLU()
{
  using namespace layers;
  using Id = typename Layer<T>::Id;

  std::map<Id, int> refcount;
  for (auto &ld : m_data)
    for (auto id : ld.layer->inputsId())
      ++refcount[id];

  struct Candidate { Id conv_id, bias_id, prelu_id; };
  std::vector<Candidate> candidates;

  for (auto &ld : m_data)
  {
    if (ld.layer->op() != OperationType::Conv2D) continue;
    Id conv_id = ld.layer->id();
    if (refcount[conv_id] != 1) continue;
    if (std::find(m_ids_output.begin(), m_ids_output.end(), conv_id) != m_ids_output.end()) continue;

    Id bias_id = -1000;
    for (auto &ld2 : m_data)
      for (auto iid : ld2.layer->inputsId())
        if (iid == conv_id) { bias_id = ld2.layer->id(); break; }
    if (bias_id == -1000) continue;
    if (getLayer(bias_id).layer->op() != OperationType::BiasAdd) continue;
    if (refcount[bias_id] != 1) continue;
    if (std::find(m_ids_output.begin(), m_ids_output.end(), bias_id) != m_ids_output.end()) continue;

    Id prelu_id = -1000;
    for (auto &ld2 : m_data)
      for (auto iid : ld2.layer->inputsId())
        if (iid == bias_id) { prelu_id = ld2.layer->id(); break; }
    if (prelu_id == -1000) continue;
    auto prelu_op = getLayer(prelu_id).layer->op();
    if (prelu_op != OperationType::PReLU && prelu_op != OperationType::LeakyRelu) continue;

    candidates.push_back({ conv_id, bias_id, prelu_id });
  }

  if (candidates.empty()) return;

  std::map<Id, int> id_to_idx;
  for (int k = 0; k < static_cast<int>(m_data.size()); ++k)
    id_to_idx[m_data[k].layer->id()] = k;

  std::sort(candidates.begin(), candidates.end(),
    [&](const Candidate &a, const Candidate &b)
    { return id_to_idx[a.prelu_id] > id_to_idx[b.prelu_id]; });

  static int s_fuse_id = -100;

  for (auto &cand : candidates)
  {
    int ci = -1, bi = -1, pi = -1;
    for (int k = 0; k < static_cast<int>(m_data.size()); ++k)
    {
      Id id = m_data[k].layer->id();
      if (id == cand.conv_id)  ci = k;
      if (id == cand.bias_id)  bi = k;
      if (id == cand.prelu_id) pi = k;
    }
    if (ci < 0 || bi < 0 || pi < 0) continue;

    auto *conv = static_cast<Conv2D<T>*>(m_data[ci].layer.get());

    Id   fused_id = s_fuse_id--;
    auto fl       = std::make_unique<Conv2DBiasPreLU<T>>(fused_id, OperationType::Conv2DBiasPreLU);
    fl->setFusionParams(conv->strides(), conv->pads(), conv->groups(), conv->mq(),
                        m_data[bi].inputs[1]->quantizer,   // bias.quantizer
                        m_data[pi].inputs[1]->quantizer);  // alpha.quantizer

    LayerData fused_ld;
    fused_ld.layer    = std::move(fl);
    fused_ld.inputs   = { m_data[ci].inputs[0],   // act
                          m_data[ci].inputs[1],   // w
                          m_data[bi].inputs[1],   // b
                          m_data[pi].inputs[1] }; // alpha
    fused_ld.layer->init(fused_ld.inputs);
    
    Tensor<T> *prelu_out_ptr = &m_data[pi].layer->m_out;

   
    m_data[ci].layer->m_torun = false;
    m_data[bi].layer->m_torun = false;
    m_data[pi].layer->m_torun = false;

  
    m_data.insert(m_data.begin() + pi + 1, std::move(fused_ld));
   
    Tensor<T> *fused_out_ptr = &m_data[pi + 1].layer->m_out;

   
    for (auto &ld2 : m_data)
      for (auto *&inp : ld2.inputs)
        if (inp == prelu_out_ptr) inp = fused_out_ptr;

   
    for (auto &oid : m_ids_output)
      if (oid == cand.prelu_id) oid = fused_id;
  }
}

template<typename T> void Model<T>::fuseConv2DBias()
{
  using namespace layers;
  using Id = typename Layer<T>::Id;

  std::map<Id, int> refcount;
  for (auto &ld : m_data)
    for (auto id : ld.layer->inputsId())
      ++refcount[id];

  struct Candidate { Id conv_id, bias_id; };
  std::vector<Candidate> candidates;

  for (auto &ld : m_data)
  {
    if (!ld.layer->m_torun) continue;
    if (ld.layer->op() != OperationType::Conv2D) continue;
    Id conv_id = ld.layer->id();
    if (refcount[conv_id] != 1) continue;
    if (std::find(m_ids_output.begin(), m_ids_output.end(), conv_id) != m_ids_output.end()) continue;

    Id bias_id = -1000;
    for (auto &ld2 : m_data)
    {
      if (!ld2.layer->m_torun) continue;
      for (auto iid : ld2.layer->inputsId())
        if (iid == conv_id) { bias_id = ld2.layer->id(); break; }
      if (bias_id != -1000) break;
    }
    if (bias_id == -1000) continue;
    if (getLayer(bias_id).layer->op() != OperationType::BiasAdd) continue;

    candidates.push_back({ conv_id, bias_id });
  }

  if (candidates.empty()) return;

  std::map<Id, int> id_to_idx;
  for (int k = 0; k < static_cast<int>(m_data.size()); ++k)
    id_to_idx[m_data[k].layer->id()] = k;

  std::sort(candidates.begin(), candidates.end(),
    [&](const Candidate &a, const Candidate &b)
    { return id_to_idx[a.bias_id] > id_to_idx[b.bias_id]; });

  static int s_fuse_id = -1000;

  for (auto &cand : candidates)
  {
    
    int ci = -1, bi = -1;
    for (int k = 0; k < static_cast<int>(m_data.size()); ++k)
    {
      Id id = m_data[k].layer->id();
      if (id == cand.conv_id) ci = k;
      if (id == cand.bias_id) bi = k;
    }
    if (ci < 0 || bi < 0) continue;

    auto *conv = static_cast<Conv2D<T>*>(m_data[ci].layer.get());

   
    Id   fused_id = s_fuse_id--;
    auto fl       = std::make_unique<Conv2DBias<T>>(fused_id, OperationType::Conv2DBias);
    fl->setFusionParams(conv->strides(), conv->pads(), conv->groups(), conv->mq(),
                        m_data[bi].inputs[1]->quantizer);  // bias.quantizer

    LayerData fused_ld;
    fused_ld.layer  = std::move(fl);
    fused_ld.inputs = { m_data[ci].inputs[0],   // act
                        m_data[ci].inputs[1],   // w
                        m_data[bi].inputs[1] }; // b
    fused_ld.layer->init(fused_ld.inputs);
   
    Tensor<T> *bias_out_ptr = &m_data[bi].layer->m_out;

   
    m_data[ci].layer->m_torun = false;
    m_data[bi].layer->m_torun = false;

   
    m_data.insert(m_data.begin() + bi + 1, std::move(fused_ld));
    // m_data[bi+1]:fused layer.

    Tensor<T> *fused_out_ptr = &m_data[bi + 1].layer->m_out;

  
    for (auto &ld2 : m_data)
      for (auto *&inp : ld2.inputs)
        if (inp == bias_out_ptr) inp = fused_out_ptr;

    
    for (auto &oid : m_ids_output)
      if (oid == cand.bias_id) oid = fused_id;
  }
}

template<typename T> void Model<T>::classifyTensorMemory()
{
  // Mark which tensors need GPU buffers (to avoid costly h2d transfers during inference)
  std::set<Tensor<T>*> gpu_needed;

  // Step 1: Mark all GPU layer outputs as GPU-needed
  for (auto &ld : m_data)
    if (ld.layer->isGpuLayer())
      gpu_needed.insert(&ld.layer->m_out);

  // Step 2: Propagate GPU requirement backward through the dependency graph
  // For each tensor needed by GPU, mark all inputs of its producer as GPU-needed.
  // This ensures CPU-only operations (Reshape, Flatten) that feed GPU layers
  // get GPU buffers activated, avoiding per-inference h2d transfers.
  bool changed = true;
  while (changed)
  {
    changed = false;
    for (auto &ld : m_data)
    {
      if (gpu_needed.count(&ld.layer->m_out) == 0) continue;
      for (auto *inp : ld.inputs)
      {
        if (gpu_needed.count(inp) == 0)
        {
          gpu_needed.insert(inp);
          changed = true;
        }
      }
    }
  }

  // Step 3: Activate GPU buffers for all GPU-needed tensors
  for (auto &ld : m_data)
    if (ld.layer->isGpuLayer())
      ld.layer->m_out.activateGpu();

  for (auto *tensor : gpu_needed)
    tensor->activateGpu();

  // Step 4: Pre-upload Const layers to GPU (they never change)
  for (auto &ld : m_data)
  {
    if (ld.layer->op() != layers::OperationType::Const) continue;
    ld.layer->m_out.activateGpu();
    ld.layer->m_out.h2d();
  }
}

template<typename T> std::vector<Tensor<T>> Model<T>::getInputsTemplate() const
{
  assert(!m_data.empty());
  std::vector<Tensor<T>> v;
  for (auto &id : m_ids_input)
  {
    auto &Lt = getLayer(id);
    if (Lt.layer->op() == layers::OperationType::Placeholder)
    {
      const auto &L = dynamic_cast<const layers::Placeholder<T> &>(*Lt.layer);
      Tensor<T> t; t.resize(L.dims()); t.quantizer = L.quantizer();
      v.push_back(t);
    }
  }
  return v;
}

}  
