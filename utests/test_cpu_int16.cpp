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
// test_cpu_int16: run SADL CPU (int16) inference on random inputs and write
// inputs + outputs to a binary file for later comparison with the CUDA version.
//
// Usage: test_cpu_int16 model.sadl output.bin [N=512]
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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sadl/model.h>

static void writeTensor(std::ofstream &f, const sadl::Tensor<int16_t> &t)
{
  int32_t ndim = static_cast<int32_t>(t.dims().size());
  f.write(reinterpret_cast<const char *>(&ndim), sizeof(ndim));
  for (int i = 0; i < ndim; ++i)
  {
    int32_t d = t.dims()[i];
    f.write(reinterpret_cast<const char *>(&d), sizeof(d));
  }
  int32_t q = t.quantizer;
  f.write(reinterpret_cast<const char *>(&q), sizeof(q));
  f.write(reinterpret_cast<const char *>(t.data()), t.size() * sizeof(int16_t));
}

int main(int argc, char **argv)
{
  if (argc < 3 || argc > 4)
  {
    std::cerr << "Usage: test_sadl_int16 model.sadl output.bin [N=512]\n";
    return 1;
  }
  const std::string model_path  = argv[1];
  const std::string output_path = argv[2];
  const int         N           = (argc == 4) ? std::atoi(argv[3]) : 512;

  sadl::Model<int16_t> model;
  {
    std::ifstream f(model_path, std::ios::binary);
    if (!f) { std::cerr << "[ERROR] cannot open " << model_path << "\n"; return 1; }
    if (!model.load(f)) { std::cerr << "[ERROR] load failed\n"; return 1; }
  }

  std::vector<sadl::Tensor<int16_t>> inputs = model.getInputsTemplate();
  if (inputs.empty()) { std::cerr << "[ERROR] no inputs\n"; return 1; }

  std::mt19937                            rng(42);
  std::uniform_int_distribution<int16_t>  dist(static_cast<int16_t>(-N),
                                                static_cast<int16_t>( N));
  for (auto &t : inputs) {
    // std::cout<<t.dims()<<std::endl;
    // auto d=t.dims();
    // d[1]=144;
    // d[2]=272;
    // t.resize(d);
    for (auto &x : t)
      x = dist(rng);
  }
  std::ofstream out(output_path, std::ios::binary);
  if (!out) { std::cerr << "[ERROR] cannot write " << output_path << "\n"; return 1; }

  int32_t nb = static_cast<int32_t>(inputs.size());
  out.write(reinterpret_cast<const char *>(&nb), sizeof(nb));
  for (const auto &t : inputs)
    writeTensor(out, t);

  if (!model.init(inputs))  { std::cerr << "[ERROR] init failed\n";  return 1; }
  if (!model.apply(inputs)) { std::cerr << "[ERROR] apply failed\n"; return 1; }

  nb = static_cast<int32_t>(model.nbOutputs());
  out.write(reinterpret_cast<const char *>(&nb), sizeof(nb));
  for (int i = 0; i < static_cast<int>(model.nbOutputs()); ++i)
    writeTensor(out, model.result(i));

  std::cout << "[INFO] wrote " << output_path << "\n";
  return 0;
}
