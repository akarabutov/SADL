# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import difflib

from .patch import Patch
from .source_file import SourceFile


class ChangeSet:
    """New files and patches of existing files, computed in memory first.

    Nothing is written until ``write()``: a failure while preparing the changes
    leaves the repository untouched.
    """

    def __init__(self, repository):
        self.repo = repository
        self.new_files = {}  # rel -> content
        self.patches = []
        self._sources = {}  # rel -> SourceFile

    def add_file(self, rel, content):
        self.new_files[rel] = content

    def add_patches(self, patches):
        self.patches += patches

    def existing_new_files(self):
        return [rel for rel in self.new_files if self.repo.exists(rel)]

    def resolve(self):
        """Run every patch in memory; sets Patch.status."""
        for patch in self.patches:
            if not self.repo.exists(patch.rel):
                patch.status = Patch.MISSING_FILE
                continue
            sf = self._sources.get(patch.rel)
            if sf is None:
                sf = self._sources[patch.rel] = SourceFile.load(self.repo, patch.rel)
            if patch.present(sf):
                patch.status = Patch.PRESENT
            elif patch.apply(sf):
                patch.status = Patch.APPLIED
            else:
                patch.status = Patch.MANUAL

    @property
    def manual_patches(self):
        return [p for p in self.patches if p.status in (Patch.MANUAL, Patch.MISSING_FILE)]

    @property
    def modified_files(self):
        return [rel for rel, sf in self._sources.items() if sf.changed]

    def diff(self):
        """Unified diff of every change (new files included)."""
        out = []
        for rel, sf in self._sources.items():
            if sf.changed:
                out += difflib.unified_diff(
                    sf.original.replace("\r\n", "\n").splitlines(keepends=True),
                    sf.text().replace("\r\n", "\n").splitlines(keepends=True),
                    "a/" + rel, "b/" + rel)
        for rel, content in self.new_files.items():
            out += difflib.unified_diff([], content.splitlines(keepends=True), "/dev/null", "b/" + rel)
        return "".join(out)

    def write(self):
        existing = self.existing_new_files()
        if existing:
            raise FileExistsError("already exist: " + ", ".join(existing))
        for rel, content in self.new_files.items():
            path = self.repo.path(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
        for rel, sf in self._sources.items():
            if sf.changed:
                with open(self.repo.path(rel), "w", encoding="utf-8", newline="") as f:
                    f.write(sf.text())
