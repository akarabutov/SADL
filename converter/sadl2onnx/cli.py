# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import argparse
import sys

from onnx2sadl.errors.conversion_error import ConversionError

from .sadl_to_onnx_converter import SadlToOnnxConverter


class CommandLineInterface:
    """``python converter/sadl_to_onnx.py --input_sadl model.sadl --output model.onnx``."""

    def build_parser(self):
        parser = argparse.ArgumentParser(prog="sadl2onnx conversion")
        parser.add_argument("--input_sadl", required=True, help="SADL model file (SADL0005)")
        parser.add_argument("--output", required=True, help="ONNX file to write")
        parser.add_argument(
            "--keep_transposes",
            action="store_true",
            help="do not remove the transpose pairs around NCHW-only ONNX ops",
        )
        parser.add_argument("--verbose", action="count", default=0)
        return parser

    def run(self, argv=None):
        args = self.build_parser().parse_args(argv)
        converter = SadlToOnnxConverter(verbose=args.verbose, optimize_transposes=not args.keep_transposes)
        try:
            converter.convert_file(args.input_sadl, args.output)
        except ConversionError as e:
            print(e, file=sys.stderr)
            return 1
        print("[INFO] ONNX model written to {} (NHWC layout, as in SADL)".format(args.output))
        return 0
