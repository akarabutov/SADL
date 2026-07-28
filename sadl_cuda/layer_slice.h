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
#include "cuda_utils.h"
#include <cstring>
#include <limits>

namespace sadl_cuda
{
namespace layers
{

#ifdef __CUDACC__
__global__ static void slice_kernel(
    const int16_t * __restrict__ A,
    int16_t       * __restrict__ out,
    int N, int in_H, int in_W, int in_C,
    int out_H, int out_W, int out_C,
    int start_h, int start_w, int start_c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * out_H * out_W * out_C;
  if (idx >= total) return;
  int out_c = idx % out_C;
  int tmp   = idx / out_C;
  int out_w = tmp % out_W;
  tmp       = tmp / out_W;
  int out_h = tmp % out_H;
  int n     = tmp / out_H;
  out[idx]  = A[((n * in_H + out_h + start_h) * in_W + (out_w + start_w)) * in_C + (out_c + start_c)];
}
#endif

template<typename T> class Slice : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 1);
    const Tensor<T> &A     = *in[0];
    const int        C_out = m_end_c - m_start_c;
    m_out.quantizer = A.quantizer;

    const int N     = A.dims()[0];
    const int in_H  = (m_out.dims().size() == 4) ? A.dims()[1] : 1;
    const int in_W  = (m_out.dims().size() == 4) ? A.dims()[2] : 1;
    const int in_C  = (m_out.dims().size() == 4) ? A.dims()[3] : A.dims()[1];
    const int out_H = (m_out.dims().size() == 4) ? (m_end_h - m_start_h) : 1;
    const int out_W = (m_out.dims().size() == 4) ? (m_end_w - m_start_w) : 1;
    const int total = N * out_H * out_W * C_out;

#ifdef __CUDACC__
    slice_kernel<<<gridDim1D(total, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
        A.dev_data(), m_out.dev_data(),
        N, in_H, in_W, in_C, out_H, out_W, C_out,
        m_start_h, m_start_w, m_start_c);
#else
    if (m_out.dims().size() == 4)
    {
      for (int n = 0; n < N; ++n)
        for (int im_i = m_start_h; im_i < m_end_h; ++im_i)
          for (int im_j = m_start_w; im_j < m_end_w; ++im_j)
            std::memcpy(&m_out(n, im_i - m_start_h, im_j - m_start_w, 0),
                        A.addr(n, im_i, im_j, m_start_c),
                        C_out * sizeof(T));
    }
    else if (m_out.dims().size() == 2)
    {
      for (int n = 0; n < N; ++n)
        std::memcpy(&m_out(n, 0), A.data() + n * A.dims()[1] + m_start_c, C_out * sizeof(T));
    }
    (void)in_H; (void)in_W; (void)in_C; (void)out_H; (void)out_W; (void)total;
#endif
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    constexpr int pow2_31 = std::numeric_limits<int>::max();
    if (in.size() != 1) return false;

    m_start_h = m_init[0]; m_end_h = m_init[1];
    m_start_w = m_init[2]; m_end_w = m_init[3];
    m_start_c = m_init[4]; m_end_c = m_init[5];

    Dimensions dim;
    dim.resize(static_cast<int>(in[0]->dims().size()));
    dim[0] = in[0]->dims()[0];

    auto resolve = [&](int &val, int dim_size)
    {
      if (val == pow2_31) val = dim_size;
      else if (val < 0)   val = dim_size + val;
    };

    if (dim.size() == 4)
    {
      resolve(m_end_h, in[0]->dims()[1]);
      resolve(m_end_w, in[0]->dims()[2]);
      resolve(m_end_c, in[0]->dims()[3]);
      dim[1] = m_end_h - m_start_h;
      dim[2] = m_end_w - m_start_w;
      dim[3] = m_end_c - m_start_c;
    }
    else if (dim.size() == 2)
    {
      resolve(m_end_c, in[0]->dims()[1]);
      dim[1] = m_end_c - m_start_c;
    }
    else return false;

    m_out.resize(dim);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version v) override
  {
    constexpr int pow2_31 = std::numeric_limits<int>::max();
    m_init[0] = m_init[2] = m_init[4] = 0;
    m_init[1] = m_init[3] = m_init[5] = pow2_31;
    if ((int)v <= (int)Version::sadl03)
    {
      file.read(reinterpret_cast<char *>(&m_init[4]), sizeof(int32_t));
      file.read(reinterpret_cast<char *>(&m_init[5]), sizeof(int32_t));
    }
    else
    {
      for (int i = 0; i < 6; ++i)
        file.read(reinterpret_cast<char *>(&m_init[i]), sizeof(int32_t));
    }
    return true;
  }

  int32_t m_start_h{}, m_end_h{}, m_start_w{}, m_end_w{}, m_start_c{}, m_end_c{};
  int32_t m_init[6]{};
};

}  // namespace layers
}  // namespace sadl_cuda
