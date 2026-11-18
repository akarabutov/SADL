# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import importlib
import os
import sys
import warnings

import torch

from onnx2sadl.errors.conversion_error import ConversionError


class PytorchModelLoader:
    """Loads a torch.nn.Module from disk for the command line tool.

    Supported inputs:
      * a TorchScript file (``torch.jit.save``)
      * a weights file (state_dict) together with ``model_class``
        ("package.module:ClassName", or "path/to/file.py:ClassName")
      * a fully pickled module (``torch.save(model)``); only with
        ``allow_pickle``, since unpickling can execute arbitrary code
    """

    def __init__(self, allow_pickle=False, verbose=0):
        self.allow_pickle = allow_pickle
        self.verbose = verbose

    def load(self, path, model_class=None, model_kwargs=None):
        if not os.path.isfile(path):
            raise ConversionError("[ERROR] no such model file: {}".format(path))
        if model_class:
            return self._load_state_dict(path, model_class, model_kwargs or {})
        model = self._try_torchscript(path)
        if model is not None:
            return model
        if self.allow_pickle:
            return self._load_pickled(path)
        raise ConversionError(
            "[ERROR] {} is not a TorchScript file: pass --model_class for a state_dict, "
            "or --allow_pickle for a pickled module from a trusted source".format(path)
        )

    def _try_torchscript(self, path):
        try:
            with warnings.catch_warnings():  # recent torch flags jit.load as deprecated
                warnings.simplefilter("ignore", FutureWarning)
                model = torch.jit.load(path, map_location="cpu")
        except Exception:
            return None
        if self.verbose:
            print("[INFO] loaded TorchScript model", path)
        return model

    def _load_pickled(self, path):
        try:
            model = torch.load(path, map_location="cpu", weights_only=False)
        except (ImportError, AttributeError) as e:
            raise ConversionError(
                "[ERROR] cannot unpickle {}: {}. The module defining the model class must be "
                "importable (add its directory to PYTHONPATH)".format(path, e)
            )
        if not isinstance(model, torch.nn.Module):
            raise ConversionError("[ERROR] {} does not contain a torch.nn.Module".format(path))
        if self.verbose:
            print("[INFO] loaded pickled model", path)
        return model

    def _load_state_dict(self, path, model_class, model_kwargs):
        cls = self.import_class(model_class)
        model = cls(**model_kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        model.load_state_dict(state)
        if self.verbose:
            print("[INFO] loaded weights {} into {}".format(path, model_class))
        return model

    @staticmethod
    def import_class(spec):
        """``"package.module:ClassName"`` or ``"path/to/file.py:ClassName"`` -> class."""
        module_name, sep, class_name = spec.rpartition(":")
        if not sep or not module_name or not class_name:
            raise ConversionError("[ERROR] --model_class must look like module:ClassName, got {!r}".format(spec))
        if module_name.endswith(".py"):
            directory, filename = os.path.split(os.path.abspath(module_name))
            if directory not in sys.path:
                sys.path.insert(0, directory)
            module_name = filename[:-3]
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ConversionError("[ERROR] cannot import {}: {}".format(spec, e))
