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
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include "dimensions.h"
#include "options.h"
#include "cuda_utils.h"

namespace sadl_cuda
{

template<typename T> struct cuda_managed_allocator
{
  using value_type      = T;
  using pointer         = T *;
  using const_pointer   = const T *;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;

  template<typename U> struct rebind { using other = cuda_managed_allocator<U>; };

  cuda_managed_allocator()                                  = default;
  cuda_managed_allocator(const cuda_managed_allocator &)    = default;
  template<typename U> cuda_managed_allocator(const cuda_managed_allocator<U> &) {}

  pointer allocate(size_type n) const
  {
    if (n == 0) return nullptr;
    void *ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, n * sizeof(T));
    if (err != cudaSuccess)
      throw std::bad_alloc();
    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, size_type) const
  {
    if (p) cudaFree(p);
  }

  bool operator==(const cuda_managed_allocator &) const { return true; }
  bool operator!=(const cuda_managed_allocator &) const { return false; }
};

// fwd
template<typename T> class Tensor;
template<typename T> void swap(Tensor<T> &t0, Tensor<T> &t1);
template<typename T> void swapData(Tensor<T> &t0, Tensor<T> &t1);

template<typename T> struct ComputationType {};

template<> struct ComputationType<int16_t>
{
  using type                = int32_t;
  static constexpr type max = std::numeric_limits<int16_t>::max();
  static void quantize(type &z, int q)  { z >>= q; }
  static void shift_left(type &z, int q){ z <<= q; }
  static void quantize(int16_t &z, int q)  { z >>= q; }
  static void shift_left(int16_t &z, int q){ z <<= q; }
};

template<typename T> class Tensor
{
  static_assert(std::is_same<T, int16_t>::value, "sadl_cuda only supports int16_t");
public:
  using value_type     = T;
  using Data           = std::vector<value_type, cuda_managed_allocator<value_type>>;
  using iterator       = typename Data::iterator;
  using const_iterator = typename Data::const_iterator;

  Tensor() = default;
  explicit Tensor(Dimensions d) { resize(d); }

  Tensor(const Tensor &o)
    : quantizer(o.quantizer), m_dims(o.m_dims), m_data(o.m_data), m_transposed(o.m_transposed),
      m_dev(nullptr), m_on_device(false) {}

  Tensor(Tensor &&o) noexcept
    : quantizer(o.quantizer), m_dims(std::move(o.m_dims)), m_data(std::move(o.m_data)),
      m_transposed(o.m_transposed), 
      m_dev(o.m_dev), m_on_device(o.m_on_device)
  {
    o.m_dev       = nullptr;
    o.m_on_device = false;
  }

  Tensor &operator=(const Tensor &o)
  {
    if (this != &o)
    {
      m_dims       = o.m_dims;
      m_data       = o.m_data;
      m_transposed = o.m_transposed;
      quantizer    = o.quantizer;
      if (m_dev) { cudaFree(m_dev); m_dev = nullptr; }
      m_on_device  = false;
    }
    return *this;
  }

  Tensor &operator=(Tensor &&o) noexcept
  {
    if (this != &o)
    {
      m_dims       = std::move(o.m_dims);
      m_data       = std::move(o.m_data);
      m_transposed = o.m_transposed;
      quantizer    = o.quantizer;
      if (m_dev) cudaFree(m_dev);
      m_dev        = o.m_dev;
      m_on_device  = o.m_on_device;
      o.m_dev       = nullptr;
      o.m_on_device = false;
    }
    return *this;
  }

  ~Tensor() { if (m_dev) cudaFree(m_dev); }

  void resize(Dimensions d)
  {
    m_dims = d;
    int64_t n = m_dims.nbElements();
    assert(n < kMaxSize);
    m_data.resize(static_cast<size_t>(n));
    if (m_dev)
    {
      cudaFree(m_dev);
      m_dev = nullptr;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_dev), n * sizeof(T)));
    }
    m_on_device = false;
  }

  void activateGpu()
  {
    if (m_dev) return;
    int64_t n = m_dims.nbElements();
    if (n > 0)
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_dev), n * sizeof(T)));
  }

  bool isGpuActive()  const { return m_dev != nullptr; }
  bool isOnDevice()   const { return m_on_device; }
  void setOnDevice(bool v)  { m_on_device = v; }

  void h2d()
  {
    if (!m_dev || m_data.empty()) return;
    CUDA_CHECK(cudaMemcpy(m_dev, m_data.data(),
                          m_data.size() * sizeof(T), cudaMemcpyHostToDevice));
    m_on_device = true;
  }

  void d2h()
  {
    if (!m_dev || m_data.empty()) return;
    CUDA_CHECK(cudaMemcpy(m_data.data(), m_dev,
                          m_data.size() * sizeof(T), cudaMemcpyDeviceToHost));
    m_on_device = false;
  }

  T *       dev_data()       { return m_dev ? m_dev : m_data.data(); }
  const T * dev_data() const { return m_dev ? m_dev : m_data.data(); }

  value_type &      operator[](int i)       { return m_data[i]; }
  value_type        operator[](int i) const { return m_data[i]; }

  value_type &operator()(int i)
  {
    assert(m_dims.size() == 1 && i >= 0 && i < m_dims[0]);
    return m_data[i];
  }
  value_type operator()(int i) const
  {
    assert(m_dims.size() == 1 && i >= 0 && i < m_dims[0]);
    return m_data[i];
  }
  bool in(int i) const { return m_dims.size() == 1 && i >= 0 && i < m_dims[0]; }

  value_type &operator()(int i, int j)
  {
    assert(m_dims.size() == 2 && i >= 0 && i < m_dims[0] && j >= 0 && j < m_dims[1]);
    return m_data[(int64_t)m_dims[1] * i + j];
  }
  value_type operator()(int i, int j) const
  {
    assert(m_dims.size() == 2 && i >= 0 && i < m_dims[0] && j >= 0 && j < m_dims[1]);
    return m_data[(int64_t)m_dims[1] * i + j];
  }
  bool in(int i, int j) const
  {
    return m_dims.size() == 2 && i >= 0 && i < m_dims[0] && j >= 0 && j < m_dims[1];
  }

  value_type &operator()(int i, int j, int k)
  {
    assert(m_dims.size() == 3);
    return m_data[(int64_t)m_dims[2] * (m_dims[1] * i + j) + k];
  }
  value_type operator()(int i, int j, int k) const
  {
    assert(m_dims.size() == 3);
    return m_data[(int64_t)m_dims[2] * (m_dims[1] * i + j) + k];
  }
  bool in(int i, int j, int k) const
  {
    return m_dims.size() == 3 && i >= 0 && i < m_dims[0] && j >= 0 && j < m_dims[1] && k >= 0 && k < m_dims[2];
  }

  value_type &operator()(int i, int j, int k, int l)
  {
    assert(m_dims.size() == 4);
    return m_data[(int64_t)m_dims[3] * (m_dims[2] * (m_dims[1] * i + j) + k) + l];
  }
  value_type operator()(int i, int j, int k, int l) const
  {
    assert(m_dims.size() == 4);
    return m_data[(int64_t)m_dims[3] * (m_dims[2] * (m_dims[1] * i + j) + k) + l];
  }
  bool in(int i, int j, int k, int l) const
  {
    return m_dims.size() == 4 && i >= 0 && i < m_dims[0] && j >= 0 && j < m_dims[1] && k >= 0 && k < m_dims[2] && l >= 0 && l < m_dims[3];
  }
  const value_type *addr(int i, int j, int k, int l) const
  {
    assert(m_dims.size() == 4);
    return &m_data[(int64_t)m_dims[3] * (m_dims[2] * (m_dims[1] * i + j) + k) + l];
  }

  void fill(value_type v)
  {
    std::fill(m_data.begin(), m_data.end(), v);
  }

  const Dimensions &dims()  const { return m_dims; }
  int64_t           size()  const { return static_cast<int64_t>(m_data.size()); }
  const value_type *data()  const { return m_data.data(); }
  value_type *      data()        { return m_data.data(); }

  iterator       begin()        { return m_data.begin(); }
  const_iterator begin()  const { return m_data.begin(); }
  iterator       end()          { return m_data.end(); }
  const_iterator end()    const { return m_data.end(); }

  int      quantizer   = 0;
  bool     getTransposed() const      { return m_transposed; }
  void     setTransposed(bool v)      { m_transposed = v; }

  Data &getData() { return m_data; }

  static constexpr int64_t kMaxSize = 32LL * 1024 * 1024 * 1024;

private:
  Dimensions m_dims;
  Data       m_data;
  bool       m_transposed = false;
  T *        m_dev        = nullptr;
  bool       m_on_device  = false;

  friend void swap<>(Tensor<T> &, Tensor<T> &);
  friend void swapData<>(Tensor<T> &, Tensor<T> &);
};

template<typename T> void swap(Tensor<T> &t0, Tensor<T> &t1)
{
  std::swap(t0.m_dims,       t1.m_dims);
  std::swap(t0.m_data,       t1.m_data);
  std::swap(t0.quantizer,    t1.quantizer);
  std::swap(t0.m_dev,        t1.m_dev);
  std::swap(t0.m_on_device,  t1.m_on_device);
}

template<typename T> void swapData(Tensor<T> &t0, Tensor<T> &t1)
{
  assert(t0.size() == t1.size());
  std::swap(t0.m_data,      t1.m_data);
  std::swap(t0.quantizer,   t1.quantizer);
  std::swap(t0.m_dev,       t1.m_dev);
  std::swap(t0.m_on_device, t1.m_on_device);
}

template<typename T> std::ostream &operator<<(std::ostream &out, const Tensor<T> &t)
{
  out << "Tensor" << t.dims();
  return out;
}

} 