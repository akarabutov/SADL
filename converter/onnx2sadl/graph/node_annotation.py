# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class NodeAnnotation:
    """Layout decisions taken for one ONNX node (or initializer / graph input).

    Attributes:
        to_remove:            the node is dropped, its output aliases its input 0
        transpose_before_in0: insert a transpose before input 0 ("nchw"/"nhwc")
        transpose_before_in1: same for inputs 1 and 2
        to_nhwc_after_out:    insert an NCHW->NHWC transpose after the output
        to_transpose:         constant data must be transposed to NHWC
        layout_onnx:          layout of the node output in the ONNX graph
    """

    def __init__(
        self,
        to_remove=False,
        transpose_before_in0=None,
        transpose_before_in1=None,
        to_nhwc_after_out=False,
        to_transpose=False,
        layout_onnx=None,
    ):
        self.to_remove = to_remove
        self.transpose_before_in0 = transpose_before_in0
        self.transpose_before_in1 = transpose_before_in1
        self.to_nhwc_after_out = to_nhwc_after_out
        self.to_transpose = to_transpose
        self.layout_onnx = layout_onnx

    @classmethod
    def user_override(cls):
        """Annotation given on the command line: None means "keep the computed value"."""
        return cls(
            to_remove=None,
            transpose_before_in0=None,
            to_nhwc_after_out=None,
            to_transpose=None,
        )

    def apply_override(self, other):
        """Copy every non-None field of a user override into this annotation."""
        if other.transpose_before_in0 is not None:
            self.transpose_before_in0 = other.transpose_before_in0
        if other.to_nhwc_after_out is not None:
            self.to_nhwc_after_out = other.to_nhwc_after_out
        if other.to_remove is not None:
            self.to_remove = other.to_remove
        if other.to_transpose is not None:
            self.to_transpose = other.to_transpose

    def __repr__(self):
        return (
            "to_remove={}, to_transpose={}, layout_onnx={}, transpose_before_in0={} "
            "transpose_before_in1={} to_nhwc_after_out={}".format(
                self.to_remove,
                self.to_transpose,
                self.layout_onnx,
                self.transpose_before_in0,
                self.transpose_before_in1,
                self.to_nhwc_after_out,
            )
        )
