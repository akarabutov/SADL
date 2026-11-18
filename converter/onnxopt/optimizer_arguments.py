# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from onnx2sadl.errors.conversion_error import ConversionError

from .optimization_pipeline import OptimizationPipeline
from .pass_registry import PassRegistry
from .plugin_loader import PluginLoader


class OptimizerArguments:
    """Command line options of the ONNX optimizer, shared by all the tools.

    Standalone (optimize_onnx.py): ``--passes --plugin --input_shape --continue_on_error``.
    Embedded (main.py, pytorch_to_sadl.py): same options prefixed with
    ``--optimize_`` plus an ``--optimize`` switch; giving passes or plugins
    also enables the optimization.
    """

    def __init__(self, embedded):
        self.embedded = embedded
        self.prefix = "optimize_" if embedded else ""

    def add_to_parser(self, parser):
        p = self.prefix
        group = parser.add_argument_group("ONNX optimization")
        if self.embedded:
            group.add_argument("--optimize", action="store_true",
                               help="optimize the ONNX model first (default passes: {})".format(
                                   ", ".join(PassRegistry.DEFAULT_PASSES)))
        group.add_argument("--" + p + "passes", nargs="+", default=None, metavar="PASS",
                           help="passes to run, in order (default: {}; see --list_passes)".format(
                               ", ".join(PassRegistry.DEFAULT_PASSES)))
        group.add_argument("--" + p + "plugin", action="append", default=[], metavar="MODULE:NAME",
                           help="user pass (OptimizationPass class or function) from module:Name or "
                                "file.py:Name; appended to the passes unless named in them")
        group.add_argument("--" + p + "input_shape", action="append", default=[], metavar="NAME:DIMS",
                           help="fix an input shape for onnxsim, e.g. input:1,3,64,64")
        group.add_argument("--" + p + "continue_on_error", action="store_true",
                           help="skip a failing pass instead of stopping")
        group.add_argument("--list_passes", action="store_true", help="list the available passes and exit")
        return parser

    def wants_list(self, args):
        return args.list_passes

    def list_passes(self, args):
        registry = PassRegistry()
        for spec in getattr(args, self.prefix + "plugin"):
            PluginLoader(registry).load(spec)
        return registry.describe()

    def build(self, args, verbose=0):
        """The OptimizationPipeline asked for by ``args``, or None when optimization is off."""
        p = self.prefix
        names = getattr(args, p + "passes")
        plugins = getattr(args, p + "plugin")
        if self.embedded and not (args.optimize or names or plugins):
            return None

        registry = PassRegistry()
        plugin_names = [PluginLoader(registry).load(spec) for spec in plugins]
        names = list(PassRegistry.DEFAULT_PASSES if names is None else names)
        names += [n for n in plugin_names if n not in names]
        options = {}
        shapes = self.parse_input_shapes(getattr(args, p + "input_shape"))
        if shapes:
            options["onnxsim"] = {"input_shapes": shapes}
        return OptimizationPipeline.from_names(
            names,
            registry=registry,
            pass_options=options,
            continue_on_error=getattr(args, p + "continue_on_error"),
            verbose=verbose,
        )

    @staticmethod
    def parse_input_shapes(specs):
        shapes = {}
        for spec in specs:
            name, sep, dims = spec.rpartition(":")
            try:
                shapes[name] = [int(d) for d in dims.split(",")]
            except ValueError:
                sep = ""
            if not sep or not name:
                raise ConversionError("[ERROR] input shape must look like name:1,3,64,64, got {!r}".format(spec))
        return shapes
