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

namespace sadl
{
namespace layers
{
template<typename T> class ReduceMean : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  using T2 = typename ComputationType<T>::type;
  Tensor<T2> result;
  bool keepdims;
  Dimensions axes, reduce_dims;
  int cnt, divq, q, r, N, divq_bits, acc_bits, shift_bits, r_shift, q2, r2, N2;
  virtual bool loadInternal(std::istream &file, Version v) override;
  DUMP_MODEL_EXT;
};

template<typename T> bool ReduceMean<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 1);
  result.fill(0);
  const Tensor<T> &A = *in[0];
  m_out.quantizer = A.quantizer;
  const int N = A.dims()[0];
  const int H = A.dims()[1];
  const int W = A.dims()[2];
  const int D = A.dims()[3];  

  for (int im_nb = 0; im_nb < N; ++im_nb)
  {
    for (int im_i = 0; im_i < H; ++im_i)
    {
      for (int im_j = 0; im_j < W; ++im_j)
      {
        for (int im_d = 0; im_d < D; ++im_d)
        {
          int ni = im_nb / reduce_dims[0];
          int hi = im_i / reduce_dims[1];
          int wi = im_j / reduce_dims[2];
          int ci = im_d / reduce_dims[3];

          T2 num = A(im_nb, im_i, im_j, im_d);
          num += result(ni, hi, wi, ci);
          COUNTERS(num);
          result(ni, hi, wi, ci) = num;
        }
      }
    }
  }
  // Compute the mean
  if constexpr (std::is_same<T, float>::value) 
  {
    for (int im_nb = 0; im_nb < result.dims()[0]; ++im_nb) {
      for (int im_i = 0; im_i < result.dims()[1]; ++im_i) {
        for (int im_j = 0; im_j < result.dims()[2]; ++im_j) {
          for (int im_d = 0; im_d < result.dims()[3]; ++im_d) {
            // average
            T2 num = result(im_nb, im_i, im_j, im_d);
            num /= cnt;
            COUNTERS(num);
            m_out(im_nb, im_i, im_j, im_d) = static_cast<T>(num);
          }
        }
      }
    }
  } 
  else 
  {
    if (divq_bits + acc_bits + N <= N2) { // can do in one step
      for (int im_nb = 0; im_nb < result.dims()[0]; ++im_nb) {
        for (int im_i = 0; im_i < result.dims()[1]; ++im_i) {
          for (int im_j = 0; im_j < result.dims()[2]; ++im_j) {
            for (int im_d = 0; im_d < result.dims()[3]; ++im_d) {
              T2 num = result(im_nb, im_i, im_j, im_d);
              num = (num * divq + r) >> q;
              COUNTERS(num);
              SATURATE(num);
              m_out(im_nb, im_i, im_j, im_d) = static_cast<T>(num);
            }
          }
        }
      }
    } 
    else 
    {
      for (int im_nb = 0; im_nb < result.dims()[0]; ++im_nb) {
        for (int im_i = 0; im_i < result.dims()[1]; ++im_i) {
          for (int im_j = 0; im_j < result.dims()[2]; ++im_j) {
            for (int im_d = 0; im_d < result.dims()[3]; ++im_d) {
              T2 num = result(im_nb, im_i, im_j, im_d); // give room for divq
              T2 temp = (num + r_shift) >> shift_bits;
              num = (temp * divq + r2) >> q2;
              COUNTERS(num);
              SATURATE(num);
              m_out(im_nb, im_i, im_j, im_d) = static_cast<T>(num);
            }
          }
        }
      }
    }
  }
  return true;
}

template<typename T> bool ReduceMean<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1)
    return false;

  SADL_DBG(std::cout << "  - input ReduceMean: " << in[0]->dims() << std::endl);
  Dimensions out_dim = in[0]->dims();
  for (int k = 0; k < axes.size(); ++k)
  {
    out_dim[axes[k]] = 1;
  }
  m_out.resize(out_dim);
  SADL_DBG(std::cout << "  - output: " << m_out.dims() << std::endl);
  result.resize(out_dim);
  result.fill(0);

  if (!keepdims)
  {
    std::cerr << "[ERROR] invalid keepdims: " << keepdims << std::endl;
    return false;
  }

  cnt = 1;
  reduce_dims = { 1, 1, 1, 1 };
  for (int axis : axes)
  {
    cnt *= in[0]->dims()[axis];
    reduce_dims[axis] = in[0]->dims()[axis];
  }
  assert(cnt != 0);

  // during init
  acc_bits = (int) ceil(log2(cnt)); // to transform with int operation
  N=sizeof(T)*8-1;   // latent bitdepth
  N2=sizeof(T2)*8-1; // accumulator bitdepth
  int minAccuracydivQ=N; // minimum bitdepth of divq
  const int bitdepth_acc=N+acc_bits;
  const int maxbitdepth_divq=N2-bitdepth_acc; // try to use maximum of bits
  const int minq=std::max(minAccuracydivQ,maxbitdepth_divq); 
    
  q=std::min(N2,minq+acc_bits-1);      // bitdepth multiplier before division
  divq = ((1LL << q) + cnt/2) / cnt;   // possible overflow here TBC 
  divq_bits=(int) ceil(log2(divq)); // to transform with int operation
  shift_bits = divq_bits-std::max(N2-(acc_bits+N),0); // for 2 steps
  q2=q-shift_bits;
  r=(1LL<<q)/2;
  r_shift= (1LL<<shift_bits)/2;
  r2=(1LL<<q2)/2;  

  m_initDone = true;
  return true;
}

template<typename T> bool ReduceMean<T>::loadInternal(std::istream &file, Version v)
{
  int32_t x = 0;
  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid nb of dimensions axes: " << x << std::endl;
    return false;
  }
  axes.resize(x);

  for (int k = 0; k < axes.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    axes[k] = x;
  }
  SADL_DBG(std::cout << "  - axes: " << axes << std::endl);

  file.read((char *)&keepdims, sizeof(keepdims));
  SADL_DBG(std::cout << "  - keepdims: " << keepdims << std::endl);

  return true;
}

}   // namespace layers
}   // namespace sadl
