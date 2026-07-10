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
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <sadl/model.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr const char *kAnchorMagic   = "SADL_INT16_ANCHOR";
constexpr int         kAnchorVersion = 1;

struct TensorSnapshot
{
  sadl::Dimensions     dims;
  int                  quantizer = 0;
  std::vector<int16_t> values;
};

void usage(const char *argv0)
{
  std::cerr << "usage:\n"
            << "  " << argv0 << " generate model.sadl anchor_file [min max seed]\n"
            << "  " << argv0 << " check model.sadl anchor_file\n";
}

int parseInt(const char *s, const char *name)
{
  char *end = nullptr;
  long  v   = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
  {
    throw std::runtime_error(std::string("invalid ") + name + ": " + s);
  }
  return static_cast<int>(v);
}

void expectToken(std::istream &file, const std::string &expected)
{
  std::string token;
  if (!(file >> token) || token != expected)
  {
    throw std::runtime_error("invalid anchor file: expected token '" + expected + "'");
  }
}

int readInt(std::istream &file, const std::string &name)
{
  int v = 0;
  if (!(file >> v))
  {
    throw std::runtime_error("invalid anchor file: missing " + name);
  }
  return v;
}

int64_t readInt64(std::istream &file, const std::string &name)
{
  int64_t v = 0;
  if (!(file >> v))
  {
    throw std::runtime_error("invalid anchor file: missing " + name);
  }
  return v;
}

TensorSnapshot captureTensor(const sadl::Tensor<int16_t> &tensor)
{
  TensorSnapshot snapshot;
  snapshot.dims      = tensor.dims();
  snapshot.quantizer = tensor.quantizer;
  snapshot.values.resize(static_cast<size_t>(tensor.size()));
  for (int64_t i = 0; i < tensor.size(); ++i)
  {
    snapshot.values[static_cast<size_t>(i)] = tensor[static_cast<int>(i)];
  }
  return snapshot;
}

std::vector<TensorSnapshot> captureTensors(const std::vector<sadl::Tensor<int16_t>> &tensors)
{
  std::vector<TensorSnapshot> snapshots;
  snapshots.reserve(tensors.size());
  for (const auto &tensor: tensors)
  {
    snapshots.push_back(captureTensor(tensor));
  }
  return snapshots;
}

std::vector<TensorSnapshot> captureOutputs(const sadl::Model<int16_t> &model)
{
  std::vector<TensorSnapshot> snapshots;
  snapshots.reserve(model.nbOutputs());
  for (size_t i = 0; i < model.nbOutputs(); ++i)
  {
    snapshots.push_back(captureTensor(model.result(static_cast<int>(i))));
  }
  return snapshots;
}

void writeTensor(std::ostream &file, int idx, const TensorSnapshot &tensor)
{
  file << "tensor " << idx << '\n';
  file << "dims " << tensor.dims.size();
  for (int d: tensor.dims)
  {
    file << ' ' << d;
  }
  file << '\n';
  file << "quantizer " << tensor.quantizer << '\n';
  file << "values " << tensor.values.size() << '\n';
  for (size_t i = 0; i < tensor.values.size(); ++i)
  {
    file << static_cast<int>(tensor.values[i]);
    if (i + 1 == tensor.values.size() || (i + 1) % 32 == 0)
    {
      file << '\n';
    }
    else
    {
      file << ' ';
    }
  }
  if (tensor.values.empty())
  {
    file << '\n';
  }
}

TensorSnapshot readTensor(std::istream &file, int expected_idx)
{
  expectToken(file, "tensor");
  const int idx = readInt(file, "tensor index");
  if (idx != expected_idx)
  {
    throw std::runtime_error("invalid anchor file: unexpected tensor index");
  }

  expectToken(file, "dims");
  const int rank = readInt(file, "tensor rank");
  if (rank < 1 || rank > sadl::Dimensions::MaxDim)
  {
    throw std::runtime_error("invalid anchor file: invalid tensor rank");
  }

  TensorSnapshot tensor;
  tensor.dims.resize(rank);
  for (int i = 0; i < rank; ++i)
  {
    tensor.dims[i] = readInt(file, "tensor dimension");
    if (tensor.dims[i] <= 0)
    {
      throw std::runtime_error("invalid anchor file: non-positive tensor dimension");
    }
  }

  expectToken(file, "quantizer");
  tensor.quantizer = readInt(file, "tensor quantizer");

  expectToken(file, "values");
  const int64_t nb_values = readInt64(file, "tensor values count");
  if (nb_values != tensor.dims.nbElements())
  {
    throw std::runtime_error("invalid anchor file: tensor values count does not match dimensions");
  }
  tensor.values.resize(static_cast<size_t>(nb_values));
  for (int64_t i = 0; i < nb_values; ++i)
  {
    const int v = readInt(file, "tensor value");
    if (v < std::numeric_limits<int16_t>::min() || v > std::numeric_limits<int16_t>::max())
    {
      throw std::runtime_error("invalid anchor file: int16 value out of range");
    }
    tensor.values[static_cast<size_t>(i)] = static_cast<int16_t>(v);
  }
  return tensor;
}

void writeAnchor(const std::string &filename, const std::vector<TensorSnapshot> &inputs, const std::vector<TensorSnapshot> &outputs)
{
  std::ofstream file(filename);
  if (!file)
  {
    throw std::runtime_error("unable to open anchor file for writing: " + filename);
  }

  file << kAnchorMagic << ' ' << kAnchorVersion << '\n';
  file << "inputs " << inputs.size() << '\n';
  for (int i = 0; i < static_cast<int>(inputs.size()); ++i)
  {
    writeTensor(file, i, inputs[static_cast<size_t>(i)]);
  }
  file << "outputs " << outputs.size() << '\n';
  for (int i = 0; i < static_cast<int>(outputs.size()); ++i)
  {
    writeTensor(file, i, outputs[static_cast<size_t>(i)]);
  }
}

void readAnchor(const std::string &filename, std::vector<TensorSnapshot> &inputs, std::vector<TensorSnapshot> &outputs)
{
  std::ifstream file(filename);
  if (!file)
  {
    throw std::runtime_error("unable to open anchor file for reading: " + filename);
  }

  std::string magic;
  int         version = 0;
  if (!(file >> magic >> version) || magic != kAnchorMagic || version != kAnchorVersion)
  {
    throw std::runtime_error("invalid anchor file header: " + filename);
  }

  expectToken(file, "inputs");
  const int nb_inputs = readInt(file, "input count");
  if (nb_inputs < 1)
  {
    throw std::runtime_error("invalid anchor file: invalid input count");
  }
  inputs.clear();
  inputs.reserve(static_cast<size_t>(nb_inputs));
  for (int i = 0; i < nb_inputs; ++i)
  {
    inputs.push_back(readTensor(file, i));
  }

  expectToken(file, "outputs");
  const int nb_outputs = readInt(file, "output count");
  if (nb_outputs < 1)
  {
    throw std::runtime_error("invalid anchor file: invalid output count");
  }
  outputs.clear();
  outputs.reserve(static_cast<size_t>(nb_outputs));
  for (int i = 0; i < nb_outputs; ++i)
  {
    outputs.push_back(readTensor(file, i));
  }
}

void loadModel(const std::string &filename, sadl::Model<int16_t> &model)
{
  std::ifstream file(filename, std::ios::binary);
  if (!model.load(file))
  {
    throw std::runtime_error("unable to read int16 model: " + filename);
  }
}

void checkTensorCompatible(const sadl::Tensor<int16_t> &tensor, const TensorSnapshot &snapshot, const std::string &kind, int idx)
{
  if (!(tensor.dims() == snapshot.dims))
  {
    throw std::runtime_error(kind + " " + std::to_string(idx) + " dimensions differ from anchor");
  }
  if (tensor.quantizer != snapshot.quantizer)
  {
    throw std::runtime_error(kind + " " + std::to_string(idx) + " quantizer differs from anchor");
  }
  if (tensor.size() != static_cast<int64_t>(snapshot.values.size()))
  {
    throw std::runtime_error(kind + " " + std::to_string(idx) + " size differs from anchor");
  }
}

void copySnapshotToTensor(const TensorSnapshot &snapshot, sadl::Tensor<int16_t> &tensor, int idx)
{
  checkTensorCompatible(tensor, snapshot, "input", idx);
  for (int64_t i = 0; i < tensor.size(); ++i)
  {
    tensor[static_cast<int>(i)] = snapshot.values[static_cast<size_t>(i)];
  }
}

int compareOutput(const TensorSnapshot &expected, const sadl::Tensor<int16_t> &actual, int idx)
{
  checkTensorCompatible(actual, expected, "output", idx);

  int64_t nb_errors = 0;
  for (int64_t i = 0; i < actual.size(); ++i)
  {
    const int16_t got = actual[static_cast<int>(i)];
    const int16_t ref = expected.values[static_cast<size_t>(i)];
    if (got != ref)
    {
      if (nb_errors < 10)
      {
        std::cerr << "[ERROR] output " << idx << " mismatch at " << i << ": expected " << static_cast<int>(ref) << " got " << static_cast<int>(got)
                  << '\n';
      }
      ++nb_errors;
    }
  }

  if (nb_errors != 0)
  {
    std::cerr << "[ERROR] output " << idx << " has " << nb_errors << " mismatching values out of " << actual.size() << '\n';
    return -1;
  }
  return 0;
}

int generateAnchor(const std::string &filename_model, const std::string &filename_anchor, int range_min, int range_max, int seed)
{
  if (range_min > range_max)
  {
    throw std::runtime_error("invalid input range: min is greater than max");
  }
  if (range_min < std::numeric_limits<int16_t>::min() || range_max > std::numeric_limits<int16_t>::max())
  {
    throw std::runtime_error("input range is outside int16");
  }

  sadl::Model<int16_t> model;
  loadModel(filename_model, model);
  auto inputs = model.getInputsTemplate();
  if (inputs.empty())
  {
    throw std::runtime_error("model has no inputs: " + filename_model);
  }

  std::mt19937                   gen(static_cast<unsigned>(seed));
  std::uniform_int_distribution<> distribution(range_min, range_max);
  for (auto &input: inputs)
  {
    for (int64_t i = 0; i < input.size(); ++i)
    {
      input[static_cast<int>(i)] = static_cast<int16_t>(distribution(gen));
    }
  }
  const auto input_snapshots = captureTensors(inputs);

  if (!model.init(inputs))
  {
    throw std::runtime_error("model initialization failed: " + filename_model);
  }
  if (!model.apply(inputs))
  {
    throw std::runtime_error("model inference failed: " + filename_model);
  }

  writeAnchor(filename_anchor, input_snapshots, captureOutputs(model));
  return 0;
}

int checkAnchor(const std::string &filename_model, const std::string &filename_anchor)
{
  std::vector<TensorSnapshot> input_snapshots;
  std::vector<TensorSnapshot> output_snapshots;
  readAnchor(filename_anchor, input_snapshots, output_snapshots);

  sadl::Model<int16_t> model;
  loadModel(filename_model, model);
  auto inputs = model.getInputsTemplate();
  if (inputs.size() != input_snapshots.size())
  {
    throw std::runtime_error("model input count differs from anchor");
  }
  for (int i = 0; i < static_cast<int>(inputs.size()); ++i)
  {
    copySnapshotToTensor(input_snapshots[static_cast<size_t>(i)], inputs[static_cast<size_t>(i)], i);
  }

  if (!model.init(inputs))
  {
    throw std::runtime_error("model initialization failed: " + filename_model);
  }
  if (!model.apply(inputs))
  {
    throw std::runtime_error("model inference failed: " + filename_model);
  }

  if (model.nbOutputs() != output_snapshots.size())
  {
    throw std::runtime_error("model output count differs from anchor");
  }

  int status = 0;
  for (int i = 0; i < static_cast<int>(output_snapshots.size()); ++i)
  {
    status |= compareOutput(output_snapshots[static_cast<size_t>(i)], model.result(i), i);
  }
  return status;
}

}   // namespace

int main(int argc, char **argv)
{
  try
  {
    if (argc < 4)
    {
      usage(argv[0]);
      return -1;
    }

    const std::string command         = argv[1];
    const std::string filename_model  = argv[2];
    const std::string filename_anchor = argv[3];

    if (command == "generate")
    {
      if (argc != 4 && argc != 7)
      {
        usage(argv[0]);
        return -1;
      }
      const int range_min = argc == 7 ? parseInt(argv[4], "range min") : -512;
      const int range_max = argc == 7 ? parseInt(argv[5], "range max") : 512;
      const int seed      = argc == 7 ? parseInt(argv[6], "seed") : 5;
      return generateAnchor(filename_model, filename_anchor, range_min, range_max, seed);
    }
    if (command == "check")
    {
      if (argc != 4)
      {
        usage(argv[0]);
        return -1;
      }
      return checkAnchor(filename_model, filename_anchor);
    }

    usage(argv[0]);
    return -1;
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] " << e.what() << '\n';
    return -1;
  }
}
