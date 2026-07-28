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
#include "cuda_utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace sadl_cuda
{
namespace layers
{


#ifdef __CUDACC__
struct BilinearUp2xCoeffs { int32_t c[4][4]; };  // c[ph_h*2+ph_w][k], k=0..3

__global__ static void fast_bilinear_up2x_kernel(
    const int16_t * __restrict__ src,
    int16_t       * __restrict__ dst,
    int N, int H_in, int W_in, int C,
    int H_out, int W_out,
    int shift,
    BilinearUp2xCoeffs coeffs)
{
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat >= N * H_out * W_out) return;

  const int n    = flat / (H_out * W_out);
  const int rem  = flat - n * H_out * W_out;
  const int im_i = rem / W_out;
  const int im_j = rem % W_out;
  const int ph_h = im_i & 1;
  const int ph_w = im_j & 1;

  // y_delta[ph=0]=-1, y_delta[ph=1]=0  (half-pixel coord, scale=2)
  const int yt_unc = (im_i >> 1) + ph_h - 1;
  const int xl_unc = (im_j >> 1) + ph_w - 1;

  const int yt = yt_unc < 0          ? 0          : yt_unc;
  const int yb = yt_unc + 1 >= H_in  ? H_in - 1   : yt_unc + 1;
  const int xl = xl_unc < 0          ? 0          : xl_unc;
  const int xr = xl_unc + 1 >= W_in  ? W_in - 1   : xl_unc + 1;

  const int ph   = ph_h * 2 + ph_w;
  const int32_t co0 = coeffs.c[ph][0];
  const int32_t co1 = coeffs.c[ph][1];
  const int32_t co2 = coeffs.c[ph][2];
  const int32_t co3 = coeffs.c[ph][3];

  const int in_stride  = W_in  * C;
  const int out_stride = W_out * C;
  const int16_t *src_b = src + n * H_in  * in_stride;
  int16_t       *dst_b = dst + n * H_out * out_stride;

  const int16_t *s0 = src_b + yt * in_stride + xl * C;
  const int16_t *s1 = src_b + yt * in_stride + xr * C;
  const int16_t *s2 = src_b + yb * in_stride + xl * C;
  const int16_t *s3 = src_b + yb * in_stride + xr * C;
  int16_t       *d  = dst_b + im_i * out_stride + im_j * C;

  for (int c = 0; c < C; ++c)
  {
    int32_t v = (co0*(int32_t)s0[c] + co1*(int32_t)s1[c]
               + co2*(int32_t)s2[c] + co3*(int32_t)s3[c]) >> shift;
    d[c] = (int16_t)SADL_CUDA_SAT(v);
  }
}
#endif

template<typename T> class Resize : public Layer<T>
{
  using T2 = int32_t;

public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  bool isGpuLayer() const override { return true; }

  virtual bool apply(std::vector<Tensor<T> *> &in) override
  {
    if (m_is_fast_downsampling) return fast_bilinear_downsampling(in);
    switch (m_scale_H)
    {
    case  2: return applyScale< 2>(in);
    case -2: return applyScale<-2>(in);
    case  4: return applyScale< 4>(in);
    case -4: return applyScale<-4>(in);
    default: return false;
    }
  }

  virtual bool init(const std::vector<Tensor<T> *> &in) override
  {
    if (in.size() != 2) return false;
    const int N    = in[0]->dims()[0];
    const int H_in = in[0]->dims()[1];
    const int W_in = in[0]->dims()[2];
    const int C    = in[0]->dims()[3];

    int scale_N, scale_C, scale_H, scale_W;
    if (m_input_label == 1)
    {
      scale_C = static_cast<int>(std::round((*in[1])[1] / static_cast<float>(C)));
      scale_H = static_cast<int>(std::round((*in[1])[2] / static_cast<float>(H_in)));
      scale_W = static_cast<int>(std::round((*in[1])[3] / static_cast<float>(W_in)));
    }
    else
    {
      scale_C = static_cast<int>(std::round(static_cast<float>((*in[1])[1])));
      scale_H = static_cast<int>(std::round(static_cast<float>((*in[1])[2])));
      scale_W = static_cast<int>(std::round(static_cast<float>((*in[1])[3])));
    }
    // Resize never changes batch dim — output_N = input_N regardless of how
    // the size/scale tensor was authored (which assumed batch=1).
    scale_N = 1;
    m_scale_H   = scale_H;
    m_scale_W   = scale_W;
    m_quantizer = in[0]->quantizer;
    if (!(scale_N == 1
          && ((scale_H == 2 && scale_W == 2) || (scale_H == 4 && scale_W == 4) || (scale_H == -2 && scale_W == -2) || (scale_H == -4 && scale_W == -4))
          && scale_C == 1))
    {
      std::cerr << "[ERROR] invalid scale factor: input: " << in[0]->dims() << " scales: " << *in[1] << " result=(" << scale_N << ", " << scale_H << ", "
                << scale_W << ", " << scale_C << ")" << std::endl;
      return false;
    }

    Dimensions dim;
    dim.resize(4);
    dim[0] = N * scale_N;
    dim[1] = (scale_H == -2) ? H_in >> 1 : (scale_H == -4) ? H_in >> 2 : H_in * scale_H;
    dim[2] = (scale_W == -2) ? W_in >> 1 : (scale_W == -4) ? W_in >> 2 : W_in * scale_W;
    dim[3] = C * scale_C;
    m_out.resize(dim);

    m_H_in  = H_in;
    m_W_in  = W_in;
    m_H_out = (int)dim[1];
    m_W_out = (int)dim[2];

    precomputePhases();

    m_is_fast_downsampling = (m_scale_H == 2 && m_coordinate_mode == 0
                              && m_mode == 0 && m_nearest_mode == 0
                              && m_input_label == 2);

    m_initDone = true;
    return true;
  }

private:
  template<int Scale>
  bool applyScale(std::vector<Tensor<T> *> &in)
  {
#ifdef __CUDACC__
    if (in[0]->isGpuActive() && in[0]->isOnDevice()) in[0]->d2h();
#endif
    if (m_mode == 0) return interpolate_bilinear<Scale>(in);
    if (m_mode == 1) return interpolate_nearest<Scale>(in);
    if (m_mode == 2) return interpolate_bicubic<Scale>(in);
    return false;
  }

protected:
  virtual bool loadInternal(std::istream &file, Version) override
  {
    int32_t x = 0;
    file.read(reinterpret_cast<char *>(&x), sizeof(x)); m_input_label = x;
    file.read(reinterpret_cast<char *>(&x), sizeof(x)); m_coordinate_mode = x;
    file.read(reinterpret_cast<char *>(&x), sizeof(x)); m_mode = x;
    file.read(reinterpret_cast<char *>(&x), sizeof(x)); m_nearest_mode = x;
    return true;
  }

private:
  // ─── phase table entry ────────────────────────────────────────────────────

  // x_delta / y_delta interpretation (mode-dependent):
  //   bilinear + nearest: xl_unc = base_x + x_delta  (exact source → left-neighbor adjusted by -1)
  //   bicubic:            xi_unc = base_x + x_delta  (x_int = x_ori >> shift, no -1 adjustment)
  struct PhaseEntry
  {
    int y_delta;
    int x_delta;
    std::array<T2, 16> coeffs;  // bilinear uses [0..3]; bicubic uses [0..15]
    int nearest_y_add;          // 0 or 1: nearest source_y = yt_unc + nearest_y_add
    int nearest_x_add;
  };

  // ─── Q-format source coordinate (no integer extraction, no clamping) ──────

  void calc_ori(int y, int x, T2 &y_ori, T2 &x_ori) const
  {
    const int Sy = m_scale_H, Sx = m_scale_W;
    const int shift = m_quantizer;
    T2 xQ   = static_cast<T2>(x) << shift;
    T2 yQ   = static_cast<T2>(y) << shift;
    T2 half = T2(1) << (shift - 1);

    if ((Sy == 2 && Sx == 2) || (Sy == 4 && Sx == 4))
    {
      const int ky = (Sy == 2) ? 1 : 2;
      const int kx = (Sx == 2) ? 1 : 2;
      if (m_coordinate_mode == 0)
      {
        x_ori = ((xQ + half) >> kx) - half;
        y_ori = ((yQ + half) >> ky) - half;
      }
      else
      {
        x_ori = (xQ + (T2(1) << (kx - 1))) >> kx;
        y_ori = (yQ + (T2(1) << (ky - 1))) >> ky;
      }
    }
    else if (Sy == -2 && Sx == -2)
    {
      x_ori = m_coordinate_mode == 0 ? (xQ << 1) + half : (xQ << 1);
      y_ori = m_coordinate_mode == 0 ? (yQ << 1) + half : (yQ << 1);
    }
    else if (Sy == -4 && Sx == -4)
    {
      T2 bias15 = T2(3) << (shift - 1);
      x_ori = m_coordinate_mode == 0 ? (xQ << 2) + bias15 : (xQ << 2);
      y_ori = m_coordinate_mode == 0 ? (yQ << 2) + bias15 : (yQ << 2);
    }
  }

  // ─── coefficient helpers (used only by precomputePhases) ──────────────────

  T2 cubic_coeff(T2 x, int i)
  {
    const int shift = m_quantizer;
    T2 x2 = x * x; x2 >>= shift;
    T2 x3 = x2 * x; x3 >>= shift;
    if (i == -1) return (-3*x3 + 6*x2 - 3*x) >> 2;
    if (i ==  0) return ( 5*x3 - 9*x2 + (T2(4) << shift)) >> 2;
    if (i ==  1) return (-5*x3 + 6*x2 + 3*x) >> 2;
    return (3*x3 - 3*x2) >> 2;
  }

  void get_bilinear_coeffs(T2 y_ori, T2 x_ori, T2 coeffs[])
  {
    const int shift  = m_quantizer;
    T2 xi = x_ori >> shift, yi = y_ori >> shift;
    T2 xr = (x_ori == (xi << shift)) ? T2(1) << shift : x_ori - (xi << shift);
    T2 yr = (y_ori == (yi << shift)) ? T2(1) << shift : y_ori - (yi << shift);
    T2 xc1 = (T2(1) << shift) - xr, xc2 = xr;
    T2 yc1 = (T2(1) << shift) - yr, yc2 = yr;
    coeffs[0] = (yc1 * xc1) >> shift;
    coeffs[1] = (yc1 * xc2) >> shift;
    coeffs[2] = (yc2 * xc1) >> shift;
    coeffs[3] = (yc2 * xc2) >> shift;
  }

  void get_bicubic_coeffs(T2 y_ori, T2 x_ori, T2 coeffs[])
  {
    const int shift  = m_quantizer;
    T2 xi = (x_ori >> shift) << shift, yi = (y_ori >> shift) << shift;
    T2 xr = (x_ori == xi) ? T2(1) << shift : x_ori - xi;
    T2 yr = (y_ori == yi) ? T2(1) << shift : y_ori - yi;
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

  void get_nearest_coeffs(T2 ori, T2 &c1, T2 &c2)
  {
    const int shift = m_quantizer;
    T2 oi = ori >> shift;
    T2 r  = (ori == (oi << shift)) ? T2(1) << shift : ori - (oi << shift);
    if (r == (T2(1) << shift)) { c1 = 0; c2 = 1; return; }
    if (m_nearest_mode == 0)   { c1 = 1; c2 = 0; return; }
    c1 = (r < (T2(1) << (shift-1))) ? 1 : 0;
    c2 = 1 - c1;
  }

  // ─── precompute ───────────────────────────────────────────────────────────

  void precomputePhases()
  {
    const int Sy    = m_scale_H, Sx = m_scale_W;
    const int shift = m_quantizer;

    m_n_ph_H = (Sy > 0) ? Sy : 1;
    m_n_ph_W = (Sx > 0) ? Sx : 1;
    m_div_H  = (Sy > 0) ? Sy : 1;   // upsample: floor(im_i / div) * 1
    m_div_W  = (Sx > 0) ? Sx : 1;
    m_mul_H  = (Sy < 0) ? -Sy : 1;  // downsample: im_i * mul
    m_mul_W  = (Sx < 0) ? -Sx : 1;

    m_phase_table.resize(m_n_ph_H * m_n_ph_W);

    for (int ph_h = 0; ph_h < m_n_ph_H; ++ph_h)
    {
      for (int ph_w = 0; ph_w < m_n_ph_W; ++ph_w)
      {
        PhaseEntry &e = m_phase_table[ph_h * m_n_ph_W + ph_w];

        T2 y_ori = 0, x_ori = 0;
        calc_ori(ph_h, ph_w, y_ori, x_ori);

        // Raw integer parts (arithmetic right shift).
        const T2  x_int  = x_ori >> shift;
        const T2  y_int  = y_ori >> shift;
        // "Natural" base that (ph / div) * mul evaluates to for this phase index.
        const int base_x = (ph_w / m_div_W) * m_mul_W;
        const int base_y = (ph_h / m_div_H) * m_mul_H;

        if (m_mode == 2)  // bicubic: use x_int directly (no exact-source -1 adjustment)
        {
          e.x_delta       = static_cast<int>(x_int) - base_x;
          e.y_delta       = static_cast<int>(y_int) - base_y;
          e.nearest_x_add = 0;
          e.nearest_y_add = 0;
          get_bicubic_coeffs(y_ori, x_ori, e.coeffs.data());
        }
        else  // bilinear and nearest: exact source coord shifts integer left by 1
        {
          const int xl_unc = static_cast<int>((x_ori == (x_int << shift)) ? x_int - 1 : x_int);
          const int yt_unc = static_cast<int>((y_ori == (y_int << shift)) ? y_int - 1 : y_int);
          e.x_delta = xl_unc - base_x;
          e.y_delta = yt_unc - base_y;

          if (m_mode == 0)  // bilinear
          {
            e.nearest_x_add = 0;
            e.nearest_y_add = 0;
            get_bilinear_coeffs(y_ori, x_ori, e.coeffs.data());
          }
          else  // nearest
          {
            T2 xc1, xc2, yc1, yc2;
            get_nearest_coeffs(x_ori, xc1, xc2);
            get_nearest_coeffs(y_ori, yc1, yc2);
            e.nearest_x_add = (xc1 == 0) ? 1 : 0;
            e.nearest_y_add = (yc1 == 0) ? 1 : 0;
          }
        }
      }
    }

    // Compute safe row/col ranges where no min/max clamping is needed.
    // bilinear/nearest: margin_top=0, margin_bot=1  (need yt_unc in [0, H_in-2])
    // bicubic:          margin_top=1, margin_bot=2  (need yi_unc in [1, H_in-3])
    // y_delta only depends on ph_h; x_delta only depends on ph_w.
    const int margin_top = (m_mode == 2) ? 1 : 0;
    const int margin_bot = (m_mode == 2) ? 2 : 1;

    m_i_safe_start = m_H_out;
    m_i_safe_end   = 0;
    for (int im_i = 0; im_i < m_H_out; ++im_i)
    {
      const int ph_h   = im_i % m_n_ph_H;
      const int yt_unc = (im_i / m_div_H) * m_mul_H + m_phase_table[ph_h * m_n_ph_W].y_delta;
      if (yt_unc >= margin_top && yt_unc + margin_bot <= m_H_in - 1)
      {
        if (m_i_safe_start == m_H_out) m_i_safe_start = im_i;
        m_i_safe_end = im_i + 1;
      }
    }

    m_j_safe_start = m_W_out;
    m_j_safe_end   = 0;
    for (int im_j = 0; im_j < m_W_out; ++im_j)
    {
      const int ph_w   = im_j % m_n_ph_W;
      // x_delta is the same for any ph_h; use index ph_w (= ph_h=0 row)
      const int xl_unc = (im_j / m_div_W) * m_mul_W + m_phase_table[ph_w].x_delta;
      if (xl_unc >= margin_top && xl_unc + margin_bot <= m_W_in - 1)
      {
        if (m_j_safe_start == m_W_out) m_j_safe_start = im_j;
        m_j_safe_end = im_j + 1;
      }
    }

    m_precomp_quantizer = m_quantizer;
  }

  // ─── fast path: scale_H=2, scale_W=2, coordinate_mode=0, mode=0 ──────────
  //
  // All structural constants are unfolded at build time:
  //   n_ph=2, div=2, mul=1
  //   y_delta[ph=0]=-1, y_delta[ph=1]=0  (same for x, from half-pixel coord with scale=2)
  //   Safe region: rows [1, H_out-1), cols [1, W_out-1)
  //
  // The bilinear pixel accumulation is inlined with pointer arithmetic, avoiding
  // the per-pixel std::vector allocation in bilinear_in_channels.

  bool fast_bilinear_downsampling(std::vector<Tensor<T> *> &in)
  {
    const Tensor<T> &data = *in[0];
    if (data.quantizer != m_precomp_quantizer)
    {
      m_quantizer = data.quantizer;
      precomputePhases();
    }
    m_out.quantizer      = data.quantizer;
    const int shift      = data.quantizer;
    const int N          = data.dims()[0];
    const int H_in       = data.dims()[1];
    const int W_in       = data.dims()[2];
    const int C          = data.dims()[3];
    const int H_out      = m_out.dims()[1];
    const int W_out      = m_out.dims()[2];
#ifdef __CUDACC__
    BilinearUp2xCoeffs coeffs;
    for (int ph = 0; ph < 4; ++ph)
      for (int k = 0; k < 4; ++k)
        coeffs.c[ph][k] = m_phase_table[ph].coeffs[k];
    const int total = N * H_out * W_out;
    fast_bilinear_up2x_kernel<<<gridDim1D(total, SADL_CUDA_BLOCK_SIZE), SADL_CUDA_BLOCK_SIZE>>>(
        data.dev_data(), m_out.dev_data(),
        N, H_in, W_in, C, H_out, W_out, shift, coeffs);
#else
    const int in_stride  = W_in  * C;
    const int out_stride = W_out * C;

    const T2 *c00 = m_phase_table[0].coeffs.data();
    const T2 *c01 = m_phase_table[1].coeffs.data();
    const T2 *c10 = m_phase_table[2].coeffs.data();
    const T2 *c11 = m_phase_table[3].coeffs.data();

    for (int n = 0; n < N; ++n)
    {
    const T *src_base = data.data()  + n * H_in  * in_stride;
    T       *dst_base = m_out.data() + n * H_out * out_stride;

    auto pixel = [&](int im_i, int im_j, int yt, int yb, int xl, int xr, const T2 *co)
    {
      const T *s0 = src_base + yt * in_stride + xl * C;
      const T *s1 = src_base + yt * in_stride + xr * C;
      const T *s2 = src_base + yb * in_stride + xl * C;
      const T *s3 = src_base + yb * in_stride + xr * C;
      T       *d  = dst_base + im_i * out_stride + im_j * C;
      for (int c = 0; c < C; ++c)
      {
        int32_t v = (co[0] * (int32_t)s0[c] + co[1] * (int32_t)s1[c]
                   + co[2] * (int32_t)s2[c] + co[3] * (int32_t)s3[c]) >> shift;
        d[c] = static_cast<T>(SADL_CUDA_SAT(v));
      }
    };

    for (int im_j = 0; im_j < W_out; ++im_j)
    {
      const int ph_w   = im_j & 1;
      const int xl_unc = (im_j >> 1) + ph_w - 1;
      const int xl     = std::max(0, xl_unc);
      const int xr     = std::min(W_in - 1, xl_unc + 1);
      pixel(0, im_j, 0, 0, xl, xr, ph_w ? c01 : c00);
    }

    for (int im_i = 1; im_i < H_out - 1; ++im_i)
    {
      const int ph_h = im_i & 1;
      const int yt = (im_i >> 1) + ph_h - 1;
      const int yb = yt + 1;
      pixel(im_i, 0, yt, yb, 0, 0, ph_h ? c10 : c00);
      for (int im_j = 1; im_j < W_out - 1; ++im_j)
      {
        const int ph_w = im_j & 1;
        const int xl   = (im_j >> 1) + ph_w - 1;
        const int xr   = xl + 1;
        pixel(im_i, im_j, yt, yb, xl, xr, ph_h ? (ph_w ? c11 : c10) : (ph_w ? c01 : c00));
      }
      {
        const int xl = (W_out - 1) >> 1;
        pixel(im_i, W_out - 1, yt, yb, xl, std::min(W_in - 1, xl + 1), ph_h ? c11 : c01);
      }
    }

    {
      const int yt = H_in - 1;
      for (int im_j = 0; im_j < W_out; ++im_j)
      {
        const int ph_w   = im_j & 1;
        const int xl_unc = (im_j >> 1) + ph_w - 1;
        const int xl     = std::max(0, xl_unc);
        const int xr     = std::min(W_in - 1, xl_unc + 1);
        pixel(H_out - 1, im_j, yt, yt, xl, xr, ph_w ? c11 : c10);
      }
    }
    }  // for n
#endif
    return true;
  }

  // ─── interpolation kernels ────────────────────────────────────────────────

  template<int Scale>
  bool interpolate_bilinear(std::vector<Tensor<T> *> &in)
  {
    constexpr int n_ph = (Scale > 0) ? Scale : 1;
    constexpr int div  = n_ph;
    constexpr int mul  = (Scale < 0) ? -Scale : 1;

    const Tensor<T> &data = *in[0];
    if (data.quantizer != m_precomp_quantizer)
    {
      m_quantizer = data.quantizer;
      precomputePhases();
    }
    m_out.quantizer     = data.quantizer;
    const int N     = data.dims()[0];
    const int H_in  = data.dims()[1], W_in  = data.dims()[2];
    const int H_out = m_out.dims()[1], W_out = m_out.dims()[2];
    const int shift = data.quantizer;

    for (int n = 0; n < N; ++n) {
    auto row_full = [&](int im_i)
    {
      const int ph_h   = im_i % n_ph;
      const int yt_unc = (im_i / div) * mul + m_phase_table[ph_h * n_ph].y_delta;
      const int yt     = std::max(0, yt_unc);
      const int yb     = std::min(H_in - 1, yt_unc + 1);
      for (int im_j = 0; im_j < W_out; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xl_unc    = (im_j / div) * mul + e.x_delta;
        const int xl        = std::max(0, xl_unc);
        const int xr        = std::min(W_in - 1, xl_unc + 1);
        const int pos[4]    = {xl, yt, xr, yb};
        bilinear_in_channels(data, e.coeffs.data(), pos, shift, n, im_i, im_j, m_out);
      }
    };

    for (int im_i = 0; im_i < m_i_safe_start; ++im_i)
      row_full(im_i);

    for (int im_i = m_i_safe_start; im_i < m_i_safe_end; ++im_i)
    {
      const int ph_h   = im_i % n_ph;
      const int yt_unc = (im_i / div) * mul + m_phase_table[ph_h * n_ph].y_delta;
      const int yt     = yt_unc;
      const int yb     = yt_unc + 1;

      for (int im_j = 0; im_j < m_j_safe_start; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xl_unc    = (im_j / div) * mul + e.x_delta;
        const int xl        = std::max(0, xl_unc);
        const int xr        = std::min(W_in - 1, xl_unc + 1);
        const int pos[4]    = {xl, yt, xr, yb};
        bilinear_in_channels(data, e.coeffs.data(), pos, shift, n, im_i, im_j, m_out);
      }
      for (int im_j = m_j_safe_start; im_j < m_j_safe_end; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xl_unc    = (im_j / div) * mul + e.x_delta;
        const int pos[4]    = {xl_unc, yt, xl_unc + 1, yb};
        bilinear_in_channels(data, e.coeffs.data(), pos, shift, n, im_i, im_j, m_out);
      }
      for (int im_j = m_j_safe_end; im_j < W_out; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xl_unc    = (im_j / div) * mul + e.x_delta;
        const int xl        = std::max(0, xl_unc);
        const int xr        = std::min(W_in - 1, xl_unc + 1);
        const int pos[4]    = {xl, yt, xr, yb};
        bilinear_in_channels(data, e.coeffs.data(), pos, shift, n, im_i, im_j, m_out);
      }
    }

    for (int im_i = m_i_safe_end; im_i < H_out; ++im_i)
      row_full(im_i);
    }  // for n

#ifdef __CUDACC__
    if (m_out.isGpuActive()) m_out.h2d();
#endif
    return true;
  }

  template<int Scale>
  bool interpolate_nearest(std::vector<Tensor<T> *> &in)
  {
    constexpr int n_ph = (Scale > 0) ? Scale : 1;
    constexpr int div  = n_ph;
    constexpr int mul  = (Scale < 0) ? -Scale : 1;

    const Tensor<T> &data = *in[0];
    if (data.quantizer != m_precomp_quantizer)
    {
      m_quantizer = data.quantizer;
      precomputePhases();
    }
    m_out.quantizer     = data.quantizer;
    const int N     = data.dims()[0];
    const int H_in  = data.dims()[1], W_in  = data.dims()[2];
    const int H_out = m_out.dims()[1], W_out = m_out.dims()[2];

    for (int n = 0; n < N; ++n) {
    auto row_full = [&](int im_i)
    {
      const int ph_h   = im_i % n_ph;
      const int yt_unc = (im_i / div) * mul + m_phase_table[ph_h * n_ph].y_delta;
      const int yt     = std::max(0, yt_unc);
      const int yb     = std::min(H_in - 1, yt_unc + 1);
      for (int im_j = 0; im_j < W_out; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xl_unc    = (im_j / div) * mul + e.x_delta;
        const int xl        = std::max(0, xl_unc);
        const int xr        = std::min(W_in - 1, xl_unc + 1);
        const int yn        = e.nearest_y_add ? yb : yt;
        const int xn        = e.nearest_x_add ? xr : xl;
        nearest_in_channels(data, n, im_i, im_j, yn, xn, m_out);
      }
    };

    for (int im_i = 0; im_i < m_i_safe_start; ++im_i)
      row_full(im_i);

    for (int im_i = m_i_safe_start; im_i < m_i_safe_end; ++im_i)
    {
      const int ph_h   = im_i % n_ph;
      const int yt_unc = (im_i / div) * mul + m_phase_table[ph_h * n_ph].y_delta;
      const int yt     = yt_unc;
      const int yb     = yt_unc + 1;

      for (int im_j = 0; im_j < m_j_safe_start; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xl_unc    = (im_j / div) * mul + e.x_delta;
        const int xl        = std::max(0, xl_unc);
        const int xr        = std::min(W_in - 1, xl_unc + 1);
        const int yn        = e.nearest_y_add ? yb : yt;
        const int xn        = e.nearest_x_add ? xr : xl;
        nearest_in_channels(data, n, im_i, im_j, yn, xn, m_out);
      }
      for (int im_j = m_j_safe_start; im_j < m_j_safe_end; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xl_unc    = (im_j / div) * mul + e.x_delta;
        const int yn        = yt_unc + e.nearest_y_add;
        const int xn        = xl_unc + e.nearest_x_add;
        nearest_in_channels(data, n, im_i, im_j, yn, xn, m_out);
      }
      for (int im_j = m_j_safe_end; im_j < W_out; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xl_unc    = (im_j / div) * mul + e.x_delta;
        const int xl        = std::max(0, xl_unc);
        const int xr        = std::min(W_in - 1, xl_unc + 1);
        const int yn        = e.nearest_y_add ? yb : yt;
        const int xn        = e.nearest_x_add ? xr : xl;
        nearest_in_channels(data, n, im_i, im_j, yn, xn, m_out);
      }
    }

    for (int im_i = m_i_safe_end; im_i < H_out; ++im_i)
      row_full(im_i);
    }  // for n

#ifdef __CUDACC__
    if (m_out.isGpuActive()) m_out.h2d();
#endif
    return true;
  }

  template<int Scale>
  bool interpolate_bicubic(std::vector<Tensor<T> *> &in)
  {
    constexpr int n_ph = (Scale > 0) ? Scale : 1;
    constexpr int div  = n_ph;
    constexpr int mul  = (Scale < 0) ? -Scale : 1;

    const Tensor<T> &data = *in[0];
    if (data.quantizer != m_precomp_quantizer)
    {
      m_quantizer = data.quantizer;
      precomputePhases();
    }
    m_out.quantizer     = data.quantizer;
    const int N     = data.dims()[0];
    const int H_in  = data.dims()[1], W_in  = data.dims()[2];
    const int H_out = m_out.dims()[1], W_out = m_out.dims()[2];
    const int shift = data.quantizer;

    for (int n = 0; n < N; ++n) {
    auto row_full = [&](int im_i)
    {
      const int ph_h   = im_i % n_ph;
      const int yi_unc = (im_i / div) * mul + m_phase_table[ph_h * n_ph].y_delta;
      for (int im_j = 0; im_j < W_out; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xi_unc    = (im_j / div) * mul + e.x_delta;
        int pos[16][2];
        int idx = 0;
        for (int dy = -1; dy <= 2; ++dy)
          for (int dx = -1; dx <= 2; ++dx)
          {
            pos[idx][0] = std::min(std::max(yi_unc + dy, 0), H_in - 1);
            pos[idx][1] = std::min(std::max(xi_unc + dx, 0), W_in - 1);
            ++idx;
          }
        bicubic_in_channels(data, e.coeffs.data(), pos, shift, n, im_i, im_j, m_out);
      }
    };

    for (int im_i = 0; im_i < m_i_safe_start; ++im_i)
      row_full(im_i);

    for (int im_i = m_i_safe_start; im_i < m_i_safe_end; ++im_i)
    {
      const int ph_h   = im_i % n_ph;
      const int yi_unc = (im_i / div) * mul + m_phase_table[ph_h * n_ph].y_delta;

      for (int im_j = 0; im_j < m_j_safe_start; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xi_unc    = (im_j / div) * mul + e.x_delta;
        int pos[16][2];
        int idx = 0;
        for (int dy = -1; dy <= 2; ++dy)
          for (int dx = -1; dx <= 2; ++dx)
          {
            pos[idx][0] = std::min(std::max(yi_unc + dy, 0), H_in - 1);
            pos[idx][1] = std::min(std::max(xi_unc + dx, 0), W_in - 1);
            ++idx;
          }
        bicubic_in_channels(data, e.coeffs.data(), pos, shift, n, im_i, im_j, m_out);
      }
      for (int im_j = m_j_safe_start; im_j < m_j_safe_end; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xi_unc    = (im_j / div) * mul + e.x_delta;
        int pos[16][2];
        int idx = 0;
        for (int dy = -1; dy <= 2; ++dy)
          for (int dx = -1; dx <= 2; ++dx)
          {
            pos[idx][0] = yi_unc + dy;
            pos[idx][1] = xi_unc + dx;
            ++idx;
          }
        bicubic_in_channels(data, e.coeffs.data(), pos, shift, n, im_i, im_j, m_out);
      }
      for (int im_j = m_j_safe_end; im_j < W_out; ++im_j)
      {
        const int ph_w      = im_j % n_ph;
        const PhaseEntry &e = m_phase_table[ph_h * n_ph + ph_w];
        const int xi_unc    = (im_j / div) * mul + e.x_delta;
        int pos[16][2];
        int idx = 0;
        for (int dy = -1; dy <= 2; ++dy)
          for (int dx = -1; dx <= 2; ++dx)
          {
            pos[idx][0] = std::min(std::max(yi_unc + dy, 0), H_in - 1);
            pos[idx][1] = std::min(std::max(xi_unc + dx, 0), W_in - 1);
            ++idx;
          }
        bicubic_in_channels(data, e.coeffs.data(), pos, shift, n, im_i, im_j, m_out);
      }
    }

    for (int im_i = m_i_safe_end; im_i < H_out; ++im_i)
      row_full(im_i);
    }  // for n

#ifdef __CUDACC__
    if (m_out.isGpuActive()) m_out.h2d();
#endif
    return true;
  }

  
  int  m_input_label          = 1;
  int  m_coordinate_mode      = 0;
  int  m_mode                 = 1;
  int  m_nearest_mode         = 0;
  int  m_scale_H              = 1;
  int  m_scale_W              = 1;
  int  m_quantizer            = 0;
  bool m_is_fast_downsampling = false;

  int m_H_in  = 0, m_W_in  = 0;
  int m_H_out = 0, m_W_out = 0;

  int                     m_n_ph_H            = 1;
  int                     m_n_ph_W            = 1;
  int                     m_div_H             = 1;
  int                     m_div_W             = 1;
  int                     m_mul_H             = 1;
  int                     m_mul_W             = 1;
  std::vector<PhaseEntry> m_phase_table;
  int                     m_precomp_quantizer = -1;

  int m_i_safe_start = 0, m_i_safe_end = 0;
  int m_j_safe_start = 0, m_j_safe_end = 0;
};

}  
}  // namespace sadl_cuda
