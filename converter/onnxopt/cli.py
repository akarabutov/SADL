# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import argparse
import sys

import onnx

from onnx2sadl.errors.conversion_error import ConversionError

from .optimizer_arguments import OptimizerArguments


class CommandLineInterface:
    """``python converter/optimize_onnx.py --input_onnx model.onnx --output model_opt.onnx``."""

    def __init__(self):
        self.optimizer_args = OptimizerArguments(embedded=False)

    def build_parser(self):
        parser = argparse.ArgumentParser(prog="onnx optimizer")
        parser.add_argument("--input_onnx", help="ONNX model to optimize")
        parser.add_argument("--output", help="optimized ONNX model to write")
        parser.add_argument("--verbose", action="count", default=1)
        parser.add_argument("--quiet", action="store_true")
        return self.optimizer_args.add_to_parser(parser)

    def run(self, argv=None):
        parser = self.build_parser()
        args = parser.parse_args(argv)
        verbose = 0 if args.quiet else args.verbose
        try:
            if self.optimizer_args.wants_list(args):
                print("Available passes:\n" + self.optimizer_args.list_passes(args))
                return 0
            if not args.input_onnx or not args.output:
                parser.error("--input_onnx and --output are required")
            pipeline = self.optimizer_args.build(args, verbose)
            onnx.save(pipeline.run(onnx.load(args.input_onnx)), args.output)
        except ConversionError as e:
            print(e, file=sys.stderr)
            return 1
        if verbose:
            print("[INFO] optimized model written to {}".format(args.output))
        return 0
