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
// test_cuda_int16: load a binary file produced by test_sadl_int16, run the
// same model through sadl_cuda, and compare output tensors element by element.
//
// Usage: test_cuda_int16 model.sadl reference.bin [tol=0]
//
// Binary file format (written by test_sadl_int16):
//   nb_inputs  [int32]
//   for each input:
//     ndim          [int32]
//     dims[0..ndim) [int32 x ndim]
//     quantizer     [int32]
//     data          [int16 x total_elements]
//   nb_outputs [int32]
//   for each output:
//     ndim          [int32]
//     dims[0..ndim) [int32 x ndim]
//     quantizer     [int32]
//     data          [int16 x total_elements]
//
// Exit code: 0 = all outputs within tolerance, 1 = mismatch or error.

#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <sadl_cuda/model.h>


static bool readTensor(std::ifstream &f, sadl_cuda::Tensor<int16_t> &t)
{
  int32_t ndim = 0;
  f.read(reinterpret_cast<char *>(&ndim), sizeof(ndim));
  if (!f || ndim < 1 || ndim > 4) return false;

  sadl_cuda::Dimensions d;
  d.resize(ndim);
  for (int i = 0; i < ndim; ++i)
  {
    int32_t x = 0;
    f.read(reinterpret_cast<char *>(&x), sizeof(x));
    d[i] = x;
  }
  int32_t q = 0;
  f.read(reinterpret_cast<char *>(&q), sizeof(q));

  t.resize(d);
  t.quantizer = q;
  f.read(reinterpret_cast<char *>(t.data()), t.size() * sizeof(int16_t));
  return f.good();
}


int main(int argc, char **argv)
{
  if (argc < 3 || argc > 4)
  {
    std::cerr << "Usage: test_cuda_int16 model.sadl reference.bin [tol=0]\n";
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string ref_path   = argv[2];
  const int         tol        = (argc == 4) ? std::atoi(argv[3]) : 0;

  std::ifstream ref(ref_path, std::ios::binary);
  if (!ref) { std::cerr << "[ERROR] cannot open " << ref_path << "\n"; return 1; }

  int32_t nb_inputs = 0;
  ref.read(reinterpret_cast<char *>(&nb_inputs), sizeof(nb_inputs));
  std::vector<sadl_cuda::Tensor<int16_t>> inputs(nb_inputs);
  for (auto &t : inputs)
    if (!readTensor(ref, t)) { std::cerr << "[ERROR] reading input\n"; return 1; }

  int32_t nb_ref_outputs = 0;
  ref.read(reinterpret_cast<char *>(&nb_ref_outputs), sizeof(nb_ref_outputs));
  std::vector<sadl_cuda::Tensor<int16_t>> ref_outputs(nb_ref_outputs);
  for (auto &t : ref_outputs)
    if (!readTensor(ref, t)) { std::cerr << "[ERROR] reading reference output\n"; return 1; }

  sadl_cuda::Model<int16_t> model;
  {
    std::ifstream f(model_path, std::ios::binary);
    if (!f) { std::cerr << "[ERROR] cannot open " << model_path << "\n"; return 1; }
    if (!model.load(f)) { std::cerr << "[ERROR] load failed\n"; return 1; }
  }

  if (!model.init(inputs))  { std::cerr << "[ERROR] init failed\n";  return 1; }
  if (!model.apply(inputs)) { std::cerr << "[ERROR] apply failed\n"; return 1; }

  if (static_cast<int>(model.nbOutputs()) != nb_ref_outputs)
  {
    std::cerr << "[ERROR] output count mismatch: model=" << model.nbOutputs()
              << " ref=" << nb_ref_outputs << "\n";
    return 1;
  }

  bool all_pass = true;
  for (int i = 0; i < nb_ref_outputs; ++i)
  {
    const auto &cuda_out = model.result(i);
    const auto &ref_out  = ref_outputs[i];

    if (!(cuda_out.dims() == ref_out.dims()))
    {
      std::cerr << "[ERROR] output " << i << " shape mismatch\n";
      all_pass = false;
      continue;
    }
    if (cuda_out.quantizer != ref_out.quantizer)
    {
      std::cerr << "[WARN] output " << i << " quantizer mismatch: cuda="
                << cuda_out.quantizer << " ref=" << ref_out.quantizer << "\n";
    }

    int   nb_errors = 0;
    int   max_diff  = 0;
    int64_t first_err_idx = -1;
    for (int k = 0; k < static_cast<int>(ref_out.size()); ++k)
    {
      int diff = std::abs(static_cast<int>(cuda_out[k]) - static_cast<int>(ref_out[k]));
      
      if (diff > max_diff) max_diff = diff;
      if (diff > tol)
      {
        if (first_err_idx < 0) first_err_idx = k;
        ++nb_errors;
      }
    }

    if (nb_errors > 0)
    {
      std::cerr << "[FAIL] output " << i << ": " << nb_errors << "/"
                << ref_out.size() << " elements differ (max_diff=" << max_diff
                << ", first at index " << first_err_idx << ", tol=" << tol << ")\n";
      all_pass = false;
    }
    else
    {
      std::cout << "[INFO] output " << i << " OK (max_diff=" << max_diff << ")\n";
    }
  }

  return all_pass ? 0 : 1;
}
