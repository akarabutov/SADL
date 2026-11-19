# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.


class AbortGeneration(Exception):
    """The user stopped the questionnaire."""


class Prompter:
    """Console questions with a default value and validation (re-asks on error)."""

    def __init__(self, input_fn=input, output_fn=print):
        self._input = input_fn
        self._print = output_fn

    def say(self, text=""):
        self._print(text)

    def _raw(self, prompt):
        try:
            return self._input(prompt).strip()
        except EOFError:
            raise AbortGeneration("input closed")

    def ask(self, question, default=None, convert=None, help_text=None):
        """Free answer; ``convert`` turns the text into a value or raises ValueError."""
        if help_text:
            self.say("  " + help_text)
        suffix = f" [{default}]" if default not in (None, "") else ""
        while True:
            answer = self._raw(f"{question}{suffix}: ")
            if answer == "" and default is not None:
                answer = str(default)
            if answer == "?" and help_text:
                self.say("  " + help_text)
                continue
            try:
                return convert(answer) if convert else answer
            except ValueError as e:
                self.say(f"  ! {e}")

    def choice(self, question, options, default, help_text=None):
        """One of ``options`` ({value: description})."""
        self.say(question + ":")
        for value, description in options.items():
            mark = "*" if value == default else " "
            self.say(f"  {mark} {value:<10} {description}")
        if help_text:
            self.say("  " + help_text)

        def convert(text):
            if text not in options:
                raise ValueError(f"answer one of: {', '.join(options)}")
            return text

        return self.ask("  choice", default, convert)

    def yes_no(self, question, default=False):
        def convert(text):
            text = text.lower()
            if text in ("y", "yes", "д", "да"):
                return True
            if text in ("n", "no", "н", "нет"):
                return False
            raise ValueError("answer y or n")

        return self.ask(question + " (y/n)", "y" if default else "n", convert)
