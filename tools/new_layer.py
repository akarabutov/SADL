#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.
"""Scaffold a new SADL layer: the automated version of CONTRIBUTE.md.

    python tools/new_layer.py                       # answer the questions
    python tools/new_layer.py --dry-run             # same, show the diff only
    python tools/new_layer.py --config NEW_LAYER_MyOp.json --yes
    python tools/new_layer.py --check MyOp          # remaining TODO(new_layer MyOp)

Standard library only. The checks after writing use the converter's Python
environment (--python) and, with --build-check, a C++ compiler.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_scaffold.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
