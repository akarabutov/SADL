# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import importlib
import os
import sys

from onnx2sadl.errors.conversion_error import ConversionError


class PluginLoader:
    """Imports user passes given as ``module:Name`` or ``path/to/file.py:Name``.

    ``Name`` may be an OptimizationPass subclass, an instance, or a function
    ``model -> model``; it is then registered in a PassRegistry.
    """

    def __init__(self, registry):
        self.registry = registry

    def load(self, spec):
        """Import ``spec``, register it and return its pass name."""
        return self.registry.register(self.import_object(spec))

    @staticmethod
    def import_object(spec):
        module_name, sep, attr = spec.rpartition(":")
        if not sep or not module_name or not attr:
            raise ConversionError("[ERROR] plugin must look like module:Name or file.py:Name, got {!r}".format(spec))
        if module_name.endswith(".py"):
            directory, filename = os.path.split(os.path.abspath(module_name))
            if directory not in sys.path:
                sys.path.insert(0, directory)
            module_name = filename[:-3]
        try:
            return getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as e:
            raise ConversionError("[ERROR] cannot import plugin {}: {}".format(spec, e))
