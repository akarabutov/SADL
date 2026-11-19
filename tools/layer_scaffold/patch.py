# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class Patch:
    """One insertion into an existing file.

    present(source_file) -> bool : the change is already there (nothing to do)
    apply(source_file)   -> bool : make the change; False if its anchor was not found
    manual                       : the code to add by hand when the anchor is missing
    """

    APPLIED = "applied"
    PRESENT = "already present"
    MANUAL = "MANUAL (anchor not found)"
    MISSING_FILE = "MANUAL (file not found)"

    def __init__(self, rel, description, present, apply, manual, section):
        self.rel = rel
        self.description = description
        self.present = present
        self.apply = apply
        self.manual = manual
        self.section = section  # CONTRIBUTE.md section
        self.status = None
