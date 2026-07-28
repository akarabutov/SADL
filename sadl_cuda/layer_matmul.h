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

namespace sadl_cuda
{
namespace layers
{


#ifdef __CUDACC__
// dim2 GEMM: A[M,K] x B_T[N,K] -> C[M,N]
__global__ static void matmul2d_kernel(
    const int16_t * __restrict__ A,
    const int16_t * __restrict__ B,
    int16_t       * __restrict__ C,
    int M, int K, int N, int shift)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int b = idx / N;
  int t = idx % N;
  int32_t acc = 0;
  const int16_t *ap = A + b * K;
  const int16_t *bp = B + t * K;
  for (int k = 0; k < K; ++k)
    acc += (int32_t)ap[k] * (int32_t)bp[k];
  acc >>= shift;
  C[idx] = (int16_t)SADL_CUDA_SAT(acc);
}
#endif

template<typename T> class MatMul : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }
  ExecMode execMode() const override { return m_exec_mode; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    assert(in.size() == 2);
    const Tensor<T> &A = *in[0];
    const Tensor<T> &B = *in[1];
    m_out.quantizer = A.quantizer - m_q;
    const int shift = B.quantizer + m_q;

    const int ndimA = static_cast<int>(A.dims().size());
    int       dum   = ndimA;
    if (ndimA - 1 == static_cast<int>(B.dims().size())) dum--;

    if (dum == 2)
    {
      m_exec_mode = ExecMode::Cuda;
      // A: [M, K] or [1, M, K];  B: [K, N] (stored transposed: b_ptr[t*K+k])
      const int K = A.dims().back();
      const int N = B.dims().back();
      const int M = (ndimA == 2) ? A.dims()[0] : A.dims()[ndimA - 2];
      const int total = M * N;
      (void)total;

#ifdef __CUDACC__
      const int16_t *a_base = A.dev_data() + (ndimA == 2 ? 0 : 0);
      matmul2d_kernel<<<gridDim1D(total, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
          a_base, B.dev_data(), m_out.dev_data(), M, K, N, shift);
#else
      for (int b = 0; b < M; ++b)
      {
        const T *ap = A.dev_data() + b * K;
        for (int t = 0; t < N; ++t)
        {
          int32_t acc = 0;
          const T *bp = B.dev_data() + t * K;
          for (int k = 0; k < K; ++k)
            acc += (int32_t)ap[k] * (int32_t)bp[k];
          acc >>= shift;
          if (ndimA == 2) m_out(b, t) = static_cast<T>(SADL_CUDA_SAT(acc));
          else            m_out(0, b, t) = static_cast<T>(SADL_CUDA_SAT(acc));
        }
      }
#endif
    }
    else if (dum == 3)
    {
      m_exec_mode = ExecMode::Cpu;
#ifdef __CUDACC__
      if (in[0]->isGpuActive() && in[0]->isOnDevice()) in[0]->d2h();
      if (in[1]->isGpuActive() && in[1]->isOnDevice()) in[1]->d2h();
#endif
      // Batched: A[N,H,K] x B[N,K,R] -> out[N,H,R]
      const int N = (ndimA == 3) ? A.dims()[0] : A.dims()[1];
      const int H = A.dims()[ndimA - 2];
      const int K = A.dims().back();
      const int R = B.dims().back();
      for (int b = 0; b < N; ++b)
        for (int i = 0; i < H; ++i)
          for (int t = 0; t < R; ++t)
          {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
            {
              int32_t av = (ndimA == 3) ? (int32_t)A(b, i, k) : (int32_t)A(0, b, i, k);
              int32_t bv = (int32_t)B(b, k, t);
              acc += av * bv;
            }
            acc >>= shift;
            if (ndimA == 3) m_out(b, i, t) = static_cast<T>(SADL_CUDA_SAT(acc));
            else            m_out(0, b, i, t) = static_cast<T>(SADL_CUDA_SAT(acc));
          }
#ifdef __CUDACC__
      if (m_out.isGpuActive()) m_out.h2d();
#endif
    }
    else if (dum == 4)
    {
      m_exec_mode = ExecMode::Cpu;
#ifdef __CUDACC__
      if (in[0]->isGpuActive() && in[0]->isOnDevice()) in[0]->d2h();
      if (in[1]->isGpuActive() && in[1]->isOnDevice()) in[1]->d2h();
#endif
      // A[Nb,Ni,H,K] x B[Nb,Ni,K,R]  (Nb = outer batch, Ni = inner batch)
      const int Nb = A.dims()[0];
      const int Ni = A.dims()[1];
      const int H  = A.dims()[2];
      const int K  = A.dims()[3];
      const int R  = B.dims().back();
      for (int nb = 0; nb < Nb; ++nb)
      for (int b = 0; b < Ni; ++b)
        for (int i = 0; i < H; ++i)
          for (int t = 0; t < R; ++t)
          {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
              acc += (int32_t)A(nb, b, i, k) * (int32_t)B(nb, b, k, t);
            acc >>= shift;
            m_out(nb, b, i, t) = static_cast<T>(SADL_CUDA_SAT(acc));
          }
#ifdef __CUDACC__
      if (m_out.isGpuActive()) m_out.h2d();
#endif
    }
    return true;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    const Tensor<T> &A = *in[0];
    const Tensor<T> &B = *in[1];

    Dimensions dim = A.dims();
    int        nd  = static_cast<int>(A.dims().size());
    if (nd - 1 == static_cast<int>(B.dims().size()))
    {
      dim[nd - 1] = B.dims().back();
    }
    else
    {
      dim[nd - 1] = B.dims().back();
    }
    m_out.resize(dim);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    file.read(reinterpret_cast<char *>(&m_q), sizeof(m_q));
    return true;
  }
  int      m_q         = 0;
  ExecMode m_exec_mode = ExecMode::Cuda;
};

}  
}  
