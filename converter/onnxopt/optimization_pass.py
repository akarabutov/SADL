# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class OptimizationPass:
    """One transformation of an ONNX model.

    Subclasses set ``name`` (used on the command line) and implement ``run``.
    A pass may modify the model in place or return a new one; it must return
    the resulting onnx.ModelProto.

    Example of a user pass::

        class DropDropout(OptimizationPass):
            name = "drop_dropout"

            def run(self, model):
                ...
                return model
    """

    name = None
    description = ""

    def run(self, model):
        raise NotImplementedError

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.name)
