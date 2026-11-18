# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

from .layer_params_reader import LayerParamsReader


class ListParamsReader(LayerParamsReader):
    """A fixed sequence of fields; each field is an int32 list ("name[]"),
    an int32 ("name"), a bool ("name?"), or 4 raw int32 ("name[4]")."""

    def __init__(self, *fields, reads_quantizer=False):
        self.fields = fields
        self.reads_quantizer = reads_quantizer

    def read(self, reader):
        params = {}
        for field in self.fields:
            if field.endswith("[]"):
                params[field[:-2]] = reader.i32_list()
            elif field.endswith("[4]"):
                params[field[:-3]] = [reader.i32() for _ in range(4)]
            elif field.endswith("?"):
                params[field[:-1]] = reader.bool()
            else:
                params[field] = reader.i32()
        return params
