# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import re


class SourceFile:
    """A text file edited in memory, line by line.

    Each line keeps its own ending, so files with CRLF, LF or mixed endings are
    written back unchanged except for the inserted lines, which use the
    dominant ending of the file.
    """

    def __init__(self, rel, text):
        self.rel = rel
        self.original = text
        self._lines = []  # [content, ending]
        pos = 0
        for m in re.finditer(r"\r\n|\n|\r", text):  # not splitlines(): \f, \x1c... are not line breaks here
            self._lines.append([text[pos:m.start()], m.group()])
            pos = m.end()
        if pos < len(text):
            self._lines.append([text[pos:], ""])
        crlf = text.count("\r\n")
        self.newline = "\r\n" if crlf > text.count("\n") - crlf else "\n"

    @classmethod
    def load(cls, repository, rel):
        with open(repository.path(rel), encoding="utf-8", newline="") as f:
            return cls(rel, f.read())

    def text(self):
        return "".join(content + ending for content, ending in self._lines)

    @property
    def changed(self):
        return self.text() != self.original

    @property
    def lines(self):
        return [content for content, _ in self._lines]

    # --- queries --------------------------------------------------------------
    def contains(self, pattern):
        return any(re.search(pattern, line) for line in self.lines)

    def find_first(self, pattern, start=0, end=None):
        lines = self.lines
        for i in range(start, len(lines) if end is None else end):
            if re.search(pattern, lines[i]):
                return i
        return None

    def find_last(self, pattern, start=0, end=None):
        lines = self.lines
        for i in reversed(range(start, len(lines) if end is None else end)):
            if re.search(pattern, lines[i]):
                return i
        return None

    def line(self, index):
        return self._lines[index][0]

    # --- edits ----------------------------------------------------------------
    def insert(self, index, new_lines):
        """Insert ``new_lines`` before line ``index`` (``len`` = at the end)."""
        if index == len(self._lines) and self._lines and self._lines[-1][1] == "":
            self._lines[-1][1] = self.newline  # file without final newline
        self._lines[index:index] = [[line, self.newline] for line in new_lines]

    def replace(self, index, content):
        self._lines[index][0] = content

    def insert_sorted(self, pattern, new_line, key, start=0, end=None):
        """Insert ``new_line`` among the lines matching ``pattern``, ordered by ``key(line)``.

        Lines are inserted before the first matching line with a greater key,
        else after the last matching line. Returns False if no line matches.
        """
        lines = self.lines
        stop = len(lines) if end is None else end
        matches = [i for i in range(start, stop) if re.search(pattern, lines[i])]
        if not matches:
            return False
        new_key = key(new_line)
        for i in matches:
            if key(lines[i]) > new_key:
                self.insert(i, [new_line])
                return True
        self.insert(matches[-1] + 1, [new_line])
        return True
