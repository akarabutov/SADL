# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import argparse
import sys

import onnx

from .converter import Converter
from .converter_config import ConverterConfig
from .errors.conversion_error import ConversionError
from .graph.node_annotation import NodeAnnotation


class CommandLineInterface:
    """``python converter/main.py --input_onnx model.onnx --output model.sadl``."""

    def build_parser(self):
        parser = argparse.ArgumentParser(prog="onnx2sadl conversion", usage="NB: force run on CPU")
        parser.add_argument("--input_onnx", action="store", nargs="?", type=str, help="name of the onnx file")
        parser.add_argument(
            "--input_default_value",
            action="store",
            nargs="?",
            type=int,
            default=128,
            help="default values to replace named inputs",
        )
        parser.add_argument("--output", action="store", nargs="?", type=str, help="name of model binary file")
        parser.add_argument("--nchw", action="store_true")
        parser.add_argument("--nhwc", action="store_true")
        parser.add_argument("--verbose", action="count")
        parser.add_argument(
            "--do_not_add_transpose_before",
            action="store",
            nargs="+",
            default=[],
            help="specify a node where add transpose before will be disable",
        )
        parser.add_argument(
            "--do_not_add_transpose_after",
            action="store",
            nargs="+",
            default=[],
            help="specify a node where add transpose after will be disable",
        )
        parser.add_argument(
            "--force_transpose_constant_tensor_dim4",
            action="store_true",
            help="If set, force transpose on 4D constant tensors (e.g., NCHW to NHWC). Default is False.",
        )
        parser.add_argument(
            "--info_string", action="store", nargs="?", type=str, help="info string to put in the model"
        )
        return self.optimizer_args.add_to_parser(parser)

    @property
    def optimizer_args(self):
        # sibling package (converter/onnxopt), imported only by the command line
        from onnxopt.optimizer_arguments import OptimizerArguments

        return OptimizerArguments(embedded=True)

    def run(self, argv=None):
        """Returns the process exit code."""
        args = self.build_parser().parse_args(argv)
        if self.optimizer_args.wants_list(args):
            print("Available passes:\n" + self.optimizer_args.list_passes(args))
            return 0
        if args.input_onnx is None:
            print("[ERROR] You should specify an onnx file", file=sys.stderr)
            return 1
        if args.output is None:
            print("[ERROR] You should specify an output file", file=sys.stderr)
            return 1

        print("[INFO] ONNX converter")
        info_string = args.info_string
        if info_string is None:
            print("[WARNING] no info string provided, please use --info_string to pass an info on the model")
            info_string = ""

        try:
            config = ConverterConfig(
                data_layout=self._data_layout(args),
                verbose=args.verbose or 0,
                user_annotations=self._user_annotations(args),
                input_default_value=args.input_default_value,
                force_transpose_constant_tensor_dim4=args.force_transpose_constant_tensor_dim4,
                info_string=info_string,
                optimizer=self.optimizer_args.build(args, max(1, args.verbose or 0)),
            )
            Converter(config).convert(onnx.load(args.input_onnx), args.output)
        except ConversionError as e:
            print(e, file=sys.stderr)
            return 1
        return 0

    @staticmethod
    def _data_layout(args):
        if args.nchw:
            return "nchw"
        if args.nhwc:
            return "nhwc"
        return None

    @staticmethod
    def _user_annotations(args):
        annotations = {}
        # NOTE(refactoring): transpose_before_in0 holds a layout ("nchw"/"nhwc") or
        # None. False is not None, so OnnxGraphParser still inserts a transpose
        # (NCHW->NHWC) for these nodes instead of skipping it. Kept as in the
        # original converter; None would probably be the intended value.
        for name in args.do_not_add_transpose_before:
            annotations.setdefault(name, NodeAnnotation.user_override()).transpose_before_in0 = False
        for name in args.do_not_add_transpose_after:
            annotations.setdefault(name, NodeAnnotation.user_override()).to_nhwc_after_out = False
        return annotations
