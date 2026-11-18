# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.
"""SADL -> ONNX converter package (reverse of onnx2sadl).

Layout:
    reader/    parses a SADL0005 file into a SadlModel
    builder/   accumulates ONNX nodes/initializers and optimizes transposes
    emitters/  one LayerEmitter per SADL layer type, producing ONNX nodes
"""
