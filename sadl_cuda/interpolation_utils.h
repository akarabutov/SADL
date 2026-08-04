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
#include "tensor.h"
#include "options.h"
#include <vector>
#include <cstdint>


namespace sadl_cuda
{
namespace layers
{

// bilinear: pos[0]=x_left, pos[1]=y_top, pos[2]=x_right, pos[3]=y_bottom
template<typename T>
void bilinear_in_channels(
    const Tensor<T> &data, const int32_t coeffs[], const int pos[],
    int shift, int n, int im_i, int im_j, Tensor<T> &out)
{
  const int in_D = data.dims()[3];
  const int pos_table[4][2] = {
      {pos[1], pos[0]}, {pos[1], pos[2]}, {pos[3], pos[0]}, {pos[3], pos[2]}};

  std::vector<int32_t> buf(in_D, 0);
  for (int ci = 0; ci < 4; ++ci)
  {
    const int32_t coeff = coeffs[ci];
    const int     py    = pos_table[ci][0];
    const int     px    = pos_table[ci][1];
    for (int c = 0; c < in_D; ++c)
      buf[c] += coeff * static_cast<int32_t>(data(n, py, px, c));
  }
  for (int c = 0; c < in_D; ++c)
  {
    int32_t v = buf[c] >> shift;
    out(n, im_i, im_j, c) = static_cast<T>(SADL_CUDA_SAT(v));
  }
}

// nearest: copy row from (y, x) to (im_i, im_j)
template<typename T>
void nearest_in_channels(
    const Tensor<T> &data, int n, int im_i, int im_j, int y, int x, Tensor<T> &out)
{
  const int in_D = out.dims()[3];
  for (int c = 0; c < in_D; ++c)
    out(n, im_i, im_j, c) = data(n, y, x, c);
}

// bicubic: coeffs[16], pos[16][2]
template<typename T>
void bicubic_in_channels(
    const Tensor<T> &data, const int32_t coeffs[], const int pos[][2],
    int shift, int n, int im_i, int im_j, Tensor<T> &out)
{
  const int in_D = data.dims()[3];
  std::vector<int32_t> buf(in_D, 0);
  for (int ci = 0; ci < 16; ++ci)
  {
    const int32_t coeff = coeffs[ci];
    const int     py    = pos[ci][0];
    const int     px    = pos[ci][1];
    for (int c = 0; c < in_D; ++c)
      buf[c] += coeff * static_cast<int32_t>(data(n, py, px, c));
  }
  for (int c = 0; c < in_D; ++c)
  {
    int32_t v = buf[c] >> shift;
    out(n, im_i, im_j, c) = static_cast<T>(SADL_CUDA_SAT(v));
  }
}

}  // namespace layers
}  
