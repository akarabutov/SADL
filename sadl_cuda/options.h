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

// CUDA backend for SADL: int16_t only.
// All source files must be compiled with nvcc.

// Saturate integer results to [-32768, 32767]
#ifndef SATURATE_RESULT
#define SATURATE_RESULT 1 // mandatory for bit exactnes
#endif

// element-wise operations
#ifndef SADL_CUDA_BLOCK_SIZE
#define SADL_CUDA_BLOCK_SIZE 256
#endif

// Thread block dimensions for 2D spatial kernels (conv, pool)
#ifndef SADL_CUDA_BLOCK_X
#define SADL_CUDA_BLOCK_X 16
#endif
#ifndef SADL_CUDA_BLOCK_Y
#define SADL_CUDA_BLOCK_Y 16
#endif




#if SATURATE_RESULT
#define SADL_CUDA_SAT(x) (((x) > 32767) ? (int32_t)32767 : (((x) < -32768) ? (int32_t)(-32768) : (x)))
#else
#define SADL_CUDA_SAT(x) (x)
#endif

#if DEBUG_PRINT
#define SADL_DBG(X) X
#else
#define SADL_DBG(X)
#endif

#ifndef SADL_CUDA_PROFILE
#define SADL_CUDA_PROFILE 0
#endif

namespace sadl_cuda
{
enum class Version
{
  unknown = -1,
  sadl01  = 1,
  sadl02  = 2,
  sadl03  = 3,
  sadl04  = 4,
  sadl05  = 5,
  sadl_new = 5
};
}  // namespace sadl_cuda
