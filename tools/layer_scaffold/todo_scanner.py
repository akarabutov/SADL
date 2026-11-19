# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import os


class TodoScanner:
    """Lists the remaining ``TODO(new_layer <Name>)`` markers of a layer."""

    EXTENSIONS = (".h", ".hpp", ".cpp", ".cu", ".py", ".js", ".sh", ".md", ".txt", ".cmake")
    SKIP_DIRS = {".git", "build", "node_modules", "venv", ".venv", "__pycache__", "layer_scaffold"}

    def __init__(self, repository):
        self.repo = repository

    def scan(self, name):
        marker = f"TODO(new_layer {name})"
        found = []
        for dirpath, dirnames, filenames in os.walk(self.repo.root):
            dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS and not d.startswith("build")]
            for filename in sorted(filenames):
                if not filename.endswith(self.EXTENSIONS) or filename.startswith("NEW_LAYER_"):
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        for number, line in enumerate(f, 1):
                            if marker in line:
                                rel = os.path.relpath(path, self.repo.root).replace(os.sep, "/")
                                found.append((rel, number, line.strip()))
                except OSError:
                    continue
        return sorted(found)
