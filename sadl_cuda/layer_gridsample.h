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
#include "interpolation_utils.h"
#include <algorithm>
#include <cstdint>

namespace sadl_cuda
{
namespace layers
{

template<typename T> class GridSample : public Layer<T>
{
  using T2 = int32_t;

public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    if (m_mode == 0) return gs_nearest(in);
    if (m_mode == 1) return gs_bilinear(in);
    if (m_mode == 2) return gs_bicubic(in);
    return false;
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    Dimensions dim;
    dim.resize(4);
    dim[0] = in[0]->dims()[0];
    dim[1] = in[1]->dims()[3];
    dim[2] = in[1]->dims()[1];
    dim[3] = in[0]->dims()[3];
    m_out.resize(dim);
    m_initDone = true;
    return true;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t x = 0;
    file.read(reinterpret_cast<char *>(&x), sizeof(x)); m_align_corners = x;
    file.read(reinterpret_cast<char *>(&x), sizeof(x)); m_mode = x;
    file.read(reinterpret_cast<char *>(&x), sizeof(x)); m_padding_mode = x;
    return true;
  }

private:
  void denormalize(T2 &v, int length)
  {
    const int shift = m_grid_quantizer;
    T2 bias = T2(1) << shift;
    if (m_align_corners)
      v = (v + bias) * (length - 1);
    else
      v = (v + bias) * length - bias;

    if (m_mode == 0)  // nearest: round to nearest integer
    {
      const int frac = shift + 1;
      const T2 half = T2(1) << shift;
      const T2 base = v >> frac;
      const T2 rem  = v - (base << frac);
      v = (rem < half) ? base : (rem > half ? (base + 1) : ((base & 1) ? (base + 1) : base));
    }
  }

  void clamp_addr(int H, int W, int &i, int &j)
  {
    i = std::max(0, std::min(H-1, i));
    j = std::max(0, std::min(W-1, j));
  }

  T2 cubic_coeff(T2 x, int i)
  {
    const int shift = m_grid_quantizer + 1;
    T2 x2 = x * x; x2 >>= shift;
    T2 x3 = x2 * x; x3 >>= shift;
    if (i == -1) return (-3*x3 + 6*x2 - 3*x) >> 2;
    if (i ==  0) return ( 5*x3 - 9*x2 + (T2(4) << shift)) >> 2;
    if (i ==  1) return (-5*x3 + 6*x2 + 3*x) >> 2;
    return (3*x3 - 3*x2) >> 2;
  }

  void get_bilinear_coeffs(T2 y, T2 x, int pos[], T2 coeffs[])
  {
    const int shift = m_grid_quantizer + 1;
    int &xl = pos[0], &yt = pos[1], &xr = pos[2], &yb = pos[3];
    xl = static_cast<int>(x >> shift);
    yt = static_cast<int>(y >> shift);
    xr = xl + 1; yb = yt + 1;
    T2 dy2 = (T2(yb) << shift) - y, dy1 = y - (T2(yt) << shift);
    T2 dx2 = (T2(xr) << shift) - x, dx1 = x - (T2(xl) << shift);
    coeffs[0] = (dy2 * dx2) >> shift;
    coeffs[1] = (dy2 * dx1) >> shift;
    coeffs[2] = (dy1 * dx2) >> shift;
    coeffs[3] = (dy1 * dx1) >> shift;
  }

  void calc_bicubic_positions(T2 y, T2 x, int H, int W, int pos[][2])
  {
    const int shift = m_grid_quantizer + 1;
    T2 xi = x >> shift, yi = y >> shift;
    int idx = 0;
    for (int dy = -1; dy <= 2; ++dy)
      for (int dx = -1; dx <= 2; ++dx)
      {
        pos[idx][0] = std::min(std::max(static_cast<int>(yi + dy), 0), H-1);
        pos[idx][1] = std::min(std::max(static_cast<int>(xi + dx), 0), W-1);
        ++idx;
      }
  }

  void get_bicubic_coeffs(T2 y, T2 x, T2 coeffs[])
  {
    const int shift = m_grid_quantizer + 1;
    T2 xi = (x >> shift) << shift, yi = (y >> shift) << shift;
    T2 xr = (x == xi) ? 0 : x - xi, yr = (y == yi) ? 0 : y - yi;
    T2 wx[4], wy[4];
    for (int i = -1; i <= 2; ++i) { wx[i+1] = cubic_coeff(xr, i); wy[i+1] = cubic_coeff(yr, i); }
    int idx = 0;
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i)
      {
        coeffs[idx] = wx[i] * wy[j] + (T2(1) << (shift-1));
        coeffs[idx] >>= shift;
        ++idx;
      }
  }

  bool gs_nearest(std::vector<Tensor<T> *> &in)
  {
    const Tensor<T> &data = *in[0];
    const Tensor<T> &grid = *in[1];
    m_out.quantizer = data.quantizer;
    m_grid_quantizer = grid.quantizer;
    const int N    = data.dims()[0];
    const int H_in = data.dims()[1], W_in = data.dims()[2];
    const int H_out = grid.dims()[3], W_out = grid.dims()[1];

    for (int n = 0; n < N; ++n)
      for (int im_j = 0; im_j < W_out; ++im_j)
        for (int im_i = 0; im_i < H_out; ++im_i)
        {
          T2 gx = static_cast<T2>(grid(n, im_j, 0, im_i));
          T2 gy = static_cast<T2>(grid(n, im_j, 1, im_i));
          denormalize(gx, W_in);
          denormalize(gy, H_in);
          int xi = static_cast<int>(gx), yi = static_cast<int>(gy);
          if (m_padding_mode == 1 && (yi < 0 || yi >= H_in || xi < 0 || xi >= W_in))
          {
            const int in_D = data.dims()[3];
            for (int c = 0; c < in_D; ++c) m_out(n, im_i, im_j, c) = T(0);
          }
          else
          {
            clamp_addr(H_in, W_in, yi, xi);
            nearest_in_channels(data, n, im_i, im_j, yi, xi, m_out);
          }
        }
    return true;
  }

  bool gs_bilinear(std::vector<Tensor<T> *> &in)
  {
    const Tensor<T> &data = *in[0];
    const Tensor<T> &grid = *in[1];
    m_out.quantizer = data.quantizer;
    m_grid_quantizer = grid.quantizer;
    const int shift = m_grid_quantizer + 1;
    const int N    = data.dims()[0];
    const int H_in = data.dims()[1], W_in = data.dims()[2];
    const int H_out = grid.dims()[3], W_out = grid.dims()[1];

    for (int n = 0; n < N; ++n)
      for (int im_j = 0; im_j < W_out; ++im_j)
        for (int im_i = 0; im_i < H_out; ++im_i)
        {
          T2 gx = static_cast<T2>(grid(n, im_j, 0, im_i));
          T2 gy = static_cast<T2>(grid(n, im_j, 1, im_i));
          denormalize(gx, W_in);
          denormalize(gy, H_in);
          T2 coeffs[4]; int pos[4];
          get_bilinear_coeffs(gy, gx, pos, coeffs);
          if (m_padding_mode == 1)
          {
            if (!(pos[1] >= 0 && pos[1] < H_in && pos[0] >= 0 && pos[0] < W_in)) coeffs[0] = 0;
            if (!(pos[1] >= 0 && pos[1] < H_in && pos[2] >= 0 && pos[2] < W_in)) coeffs[1] = 0;
            if (!(pos[3] >= 0 && pos[3] < H_in && pos[0] >= 0 && pos[0] < W_in)) coeffs[2] = 0;
            if (!(pos[3] >= 0 && pos[3] < H_in && pos[2] >= 0 && pos[2] < W_in)) coeffs[3] = 0;
          }
          clamp_addr(H_in, W_in, pos[1], pos[0]);
          clamp_addr(H_in, W_in, pos[1], pos[2]);
          clamp_addr(H_in, W_in, pos[3], pos[0]);
          clamp_addr(H_in, W_in, pos[3], pos[2]);
          bilinear_in_channels(data, coeffs, pos, shift, n, im_i, im_j, m_out);
        }
    return true;
  }

  bool gs_bicubic(std::vector<Tensor<T> *> &in)
  {
    const Tensor<T> &data = *in[0];
    const Tensor<T> &grid = *in[1];
    m_out.quantizer = data.quantizer;
    m_grid_quantizer = grid.quantizer;
    const int shift = m_grid_quantizer + 1;
    const int N    = data.dims()[0];
    const int H_in = data.dims()[1], W_in = data.dims()[2];
    const int H_out = grid.dims()[3], W_out = grid.dims()[1];

    for (int n = 0; n < N; ++n)
      for (int im_j = 0; im_j < W_out; ++im_j)
        for (int im_i = 0; im_i < H_out; ++im_i)
        {
          T2 gx = static_cast<T2>(grid(n, im_j, 0, im_i));
          T2 gy = static_cast<T2>(grid(n, im_j, 1, im_i));
          denormalize(gx, W_in);
          denormalize(gy, H_in);
          T2 coeffs[16]; int pos[16][2];
          calc_bicubic_positions(gy, gx, H_in, W_in, pos);
          get_bicubic_coeffs(gy, gx, coeffs);
          if (m_padding_mode == 1)
          {
            T2 xi = gx >> shift, yi = gy >> shift;
            int idx = 0;
            for (int dy = -1; dy <= 2; ++dy)
              for (int dx = -1; dx <= 2; ++dx)
              {
                if (!(yi + dy >= 0 && yi + dy < H_in && xi + dx >= 0 && xi + dx < W_in))
                  coeffs[idx] = 0;
                ++idx;
              }
          }
          bicubic_in_channels(data, coeffs, pos, shift, n, im_i, im_j, m_out);
        }
    return true;
  }

  int m_align_corners  = 0;
  int m_mode           = 1;
  int m_padding_mode   = 0;
  int m_grid_quantizer = 0;
};

}  // namespace layers
}  // namespace sadl_cuda
