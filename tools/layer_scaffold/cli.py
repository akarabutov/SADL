# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.

import argparse
import json
import sys
from pathlib import Path

from .change_set import ChangeSet
from .cpp_templates import CppTemplates
from .layer_spec import LayerSpec
from .netron_templates import NetronTemplates
from .notes_writer import NotesWriter
from .prompter import AbortGeneration, Prompter
from .python_templates import PythonTemplates
from .questionnaire import Questionnaire
from .registration_patches import RegistrationPatches
from .repository import Repository
from .todo_scanner import TodoScanner
from .verifier import Verifier


class CommandLineInterface:
    """python tools/new_layer.py: scaffold a new SADL layer (see CONTRIBUTE.md)."""

    def __init__(self, prompter=None):
        self.prompter = prompter or Prompter()

    def parser(self):
        p = argparse.ArgumentParser(
            prog="new_layer.py",
            description="Create the files of a new SADL layer and register it everywhere "
                        "(C++, sample tools, CUDA, Netron, ONNX <-> SADL converters, tests), "
                        "following CONTRIBUTE.md. What is left to write by hand is marked "
                        "TODO(new_layer <Name>).",
        )
        p.add_argument("--config", help="JSON answers (written next to the notes as NEW_LAYER_<Name>.json); "
                                        "no questions are asked")
        p.add_argument("--dry-run", action="store_true", help="show the diff, write nothing")
        p.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation before writing")
        p.add_argument("--check", metavar="NAME", help="list the remaining TODO markers of layer NAME")
        p.add_argument("--no-verify", action="store_true", help="skip the checks after writing")
        p.add_argument("--build-check", action="store_true",
                       help="also compile the sample tools with the new layer (needs a C++ compiler)")
        p.add_argument("--python", help="interpreter of the converter environment, for the checks")
        p.add_argument("--cxx", help="C++ compiler for --build-check (default: CXX, cl, g++, clang++)")
        p.add_argument("--root", default=str(Path(__file__).resolve().parents[2]),
                       help="SADL repository root (default: %(default)s)")
        return p

    def run(self, argv=None):
        args = self.parser().parse_args(argv)
        try:
            repo = Repository(args.root)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return 2
        if args.check:
            return self._check(repo, args.check)
        try:
            spec = self._spec(repo, args)
        except AbortGeneration:
            print("\nAborted, nothing written.")
            return 1
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"[ERROR] {e}")
            return 2

        errors = repo.conflicts(spec)
        if errors:
            print("[ERROR] cannot generate the layer:\n  - " + "\n  - ".join(errors))
            return 2

        changes, file_roles = self._build_changes(repo, spec)
        existing = changes.existing_new_files()
        if existing:
            print("[ERROR] these files already exist: " + ", ".join(existing))
            return 2
        changes.resolve()
        notes = NotesWriter(spec, changes, file_roles).render()
        changes.add_file(spec.notes_path, notes)
        changes.add_file(spec.config_path, spec.to_json())

        self._summary(spec, changes)
        if args.dry_run:
            print(changes.diff())
            print("Dry run: nothing written.")
            return 0
        if not args.yes and not self._confirm():
            print("Nothing written.")
            return 1
        changes.write()
        print(f"\nWritten. Next steps: {spec.notes_path}")
        if args.no_verify:
            return 0
        return self._verify(repo, spec, changes, args)

    # --- steps ----------------------------------------------------------------
    def _spec(self, repo, args):
        if args.config:
            with open(args.config, encoding="utf-8") as f:
                spec = LayerSpec.from_dict(json.load(f), default_id=repo.operation_type_count())
        else:
            spec = Questionnaire(repo, self.prompter).run()
        spec.validate()
        return spec

    @staticmethod
    def _build_changes(repo, spec):
        cpp = CppTemplates(spec, repo.cpp_license_header())
        py = PythonTemplates(spec)
        changes = ChangeSet(repo)
        roles = {}

        def add(rel, content, role):
            changes.add_file(rel, content)
            roles[rel] = role

        add(spec.cpp_layer_path, cpp.cpu_layer(), "cpp")
        if spec.cuda != "skip":
            add(spec.cuda_layer_path, cpp.cuda_layer(), "cuda")
        add(spec.converter_path, py.converter(), "converter")
        if spec.serializer_kind == "custom":
            add(spec.serializer_path, py.serializer(), "serializer")
        if spec.needs_emitter_file:
            add(spec.emitter_path, py.emitter(), "emitter")
        if spec.test_model:
            add(spec.test_model_script_path, py.test_model(), "test")
        changes.add_patches(RegistrationPatches(spec, cpp, py, NetronTemplates(spec)).build())
        return changes, roles

    def _summary(self, spec, changes):
        print(f"\nLayer {spec.name} (SADL id {spec.sadl_id})")
        print("New files:")
        for rel in changes.new_files:
            print(f"  + {rel}")
        print("Existing files:")
        for patch in changes.patches:
            print(f"  {patch.status:<26} {patch.rel}: {patch.description}")
        if changes.manual_patches:
            print(f"{len(changes.manual_patches)} change(s) must be made by hand: see {spec.notes_path}")

    def _confirm(self):
        try:
            return self.prompter.yes_no("Write these changes", True)
        except AbortGeneration:
            return False

    def _verify(self, repo, spec, changes, args):
        verifier = Verifier(repo, spec, python=args.python or sys.executable, cxx=args.cxx)
        py_files = [rel for rel in list(changes.new_files) + changes.modified_files if rel.endswith(".py")]
        checks = [
            ("registries + parameter round trip", verifier.python_registries),
            ("flake8", lambda: verifier.flake8(py_files)),
        ]
        if args.build_check:
            checks.append(("C++ build", verifier.cpp_build))
        failed = False
        print("\nChecks:")
        for label, check in checks:
            status, message = check()
            failed |= status == "FAIL"
            print(f"  [{status}] {label}: {message}")
        if not args.build_check:
            print("  [SKIP] C++ build: use --build-check")
        return 1 if failed else 0

    @staticmethod
    def _check(repo, name):
        found = TodoScanner(repo).scan(name)
        for rel, number, text in found:
            print(f"{rel}:{number}: {text}")
        print(f"{len(found)} TODO(new_layer {name}) marker(s) left")
        return 1 if found else 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # e.g. a cp1252 Windows console and the ✓ of sadl_cuda/README.md
            stream.reconfigure(errors="replace")
    return CommandLineInterface().run(argv)
