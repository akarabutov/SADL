# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import re

from .layer_params_reader import LayerParamsReader


class ListParamsReader(LayerParamsReader):
    """A fixed sequence of fields; each field is an int32 list ("name[]"),
    an int32 ("name"), a bool ("name?"), or N raw int32 ("name[N]", e.g. "pads[4]")."""

    _FIXED = re.compile(r"^(\w+)\[(\d+)\]$")

    def __init__(self, *fields, reads_quantizer=False):
        self.fields = fields
        self.reads_quantizer = reads_quantizer

    def read(self, reader):
        params = {}
        for field in self.fields:
            fixed = self._FIXED.match(field)
            if field.endswith("[]"):
                params[field[:-2]] = reader.i32_list()
            elif fixed:
                params[fixed.group(1)] = [reader.i32() for _ in range(int(fixed.group(2)))]
            elif field.endswith("?"):
                params[field[:-1]] = reader.bool()
            else:
                params[field] = reader.i32()
        return params
