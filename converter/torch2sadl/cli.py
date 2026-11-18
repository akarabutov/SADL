# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import argparse
import json
import sys

from onnx2sadl.errors.conversion_error import ConversionError

from .api import convert_pytorch_to_sadl
from .input_spec import InputSpec
from .pytorch_model_loader import PytorchModelLoader
from .pytorch_onnx_exporter import PytorchOnnxExporter


class CommandLineInterface:
    """``python converter/pytorch_to_sadl.py --model model.pt --input_shape 1,3,64,64 --output model.sadl``."""

    def build_parser(self):
        parser = argparse.ArgumentParser(prog="pytorch2sadl conversion")
        parser.add_argument("--model", required=True, help="TorchScript file, state_dict file or pickled module")
        parser.add_argument(
            "--model_class",
            help="module:ClassName (or file.py:ClassName) to build the model when --model is a state_dict",
        )
        parser.add_argument("--model_args", default="{}", help="JSON keyword arguments for --model_class")
        parser.add_argument(
            "--allow_pickle",
            action="store_true",
            help="allow loading a pickled module (torch.save(model)); only for trusted files",
        )
        parser.add_argument(
            "--input_shape",
            action="append",
            required=True,
            help="input shape, e.g. 1,3,64,64 (repeat for several inputs; batch must be 1)",
        )
        parser.add_argument("--output", required=True, help="SADL file to write")
        parser.add_argument("--save_onnx", help="also save the intermediate ONNX model")
        parser.add_argument("--opset", type=int, default=PytorchOnnxExporter.DEFAULT_OPSET)
        parser.add_argument("--info_string", default=None, help="info string to put in the model")
        parser.add_argument("--input_default_value", type=int, default=128)
        parser.add_argument("--force_transpose_constant_tensor_dim4", action="store_true")
        parser.add_argument("--do_not_add_transpose_before", nargs="+", default=[])
        parser.add_argument("--do_not_add_transpose_after", nargs="+", default=[])
        parser.add_argument("--verbose", action="count", default=0)
        return parser

    def run(self, argv=None):
        args = self.build_parser().parse_args(argv)
        print("[INFO] PyTorch to SADL converter")
        try:
            specs = [InputSpec.parse(s) for s in args.input_shape]
            model_kwargs = json.loads(args.model_args)
            loader = PytorchModelLoader(allow_pickle=args.allow_pickle, verbose=args.verbose)
            model = loader.load(args.model, args.model_class, model_kwargs)
            if args.info_string is None:
                print("[WARNING] no info string provided, please use --info_string to pass an info on the model")
            input_names = [s.name for s in specs] if all(s.name for s in specs) else None
            convert_pytorch_to_sadl(
                model,
                specs,
                args.output,
                info_string=args.info_string or "",
                onnx_path=args.save_onnx,
                opset=args.opset,
                input_names=input_names,
                input_default_value=args.input_default_value,
                force_transpose_constant_tensor_dim4=args.force_transpose_constant_tensor_dim4,
                do_not_add_transpose_before=args.do_not_add_transpose_before,
                do_not_add_transpose_after=args.do_not_add_transpose_after,
                verbose=args.verbose,
            )
        except (ConversionError, json.JSONDecodeError) as e:
            print(e, file=sys.stderr)
            return 1
        print("[INFO] SADL model written to {}".format(args.output))
        return 0
