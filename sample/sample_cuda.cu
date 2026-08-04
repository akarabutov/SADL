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

#include <sadl_cuda/model.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;


static void infer(const string &filename, int override_size, int N)
{
  sadl_cuda::Model<int16_t> model;
  ifstream file(filename, ios::binary);
  cout << "[INFO] Model loading" << endl;
  if (!model.load(file))
  {
    cerr << "[ERROR] Unable to read model " << filename << endl;
    exit(-1);
  }

  vector<sadl_cuda::Tensor<int16_t>> inputs = model.getInputsTemplate();
  
  for (auto &t : inputs)
  {
    auto d = t.dims();
    if (override_size > 1 && d.size() >= 3)
    {
      d[1] = override_size;
      d[2] = override_size;
    }   
    t.resize(d);
  }
  if (!model.init(inputs))
  {
    cerr << "[ERROR] issue during initialization" << endl;
    exit(-1);
  }

  double step = (1. + 1.) / (inputs[0].size() - 1);
  double x0   = -1.;
  for (auto &t : inputs)
    for (auto &x : t)
    {
      x = static_cast<int16_t>(x0);
      x0 += step;
    }

  if (!model.apply(inputs))
  {
    cerr << "[ERROR] issue during inference" << endl;
    exit(-1);
  }


  #if !SADL_CUDA_PROFILE
  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);


  cudaEventRecord(ev_start);
  for (int i = 0; i < N; ++i)
  {
    if (!model.apply(inputs))
    {
      cerr << "[ERROR] issue during inference" << endl;
      exit(-1);
    }
  }
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  cout << "[INFO] " << ms / N << " ms (avg over " << N << " samples, "
      << endl;
  cout << "[INFO] " << ms     << " ms (total over " << N << " samples)" << endl;

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
#endif
}

int main(int argc, char **argv)
{
  // Usage: sample_cuda filename_model [override_size] [--n N] [--batch] [--demo-async]
  //   --n N         : number of samples to time over (default 16)
  string         filename_model;
  int            override_size = -1;
  int            N             = 16;
  vector<string> positional;

  for (int i = 1; i < argc; ++i)
  {
    string a = argv[i];
    if (a == "--n" && i + 1 < argc)        N          = atoi(argv[++i]);
    else                                   positional.push_back(a);
  }

  if (positional.empty() || positional.size() > 2)
  {
    cout << "[ERROR] sample_cuda filename_model [override_size] [--n N]" << endl;
    return 1;
  }
  filename_model = positional[0];
  if (positional.size() == 2) override_size = atoi(positional[1].c_str());
  if (N < 1) { cerr << "[ERROR] --n must be >= 1\n"; return 1; }

  infer(filename_model, override_size, N);
  return 0;
}
