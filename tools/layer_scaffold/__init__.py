# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2010-2024, ITU/ISO/IEC. See LICENSE in the repository root.
"""Scaffolding of a new SADL layer (see CONTRIBUTE.md and tools/new_layer.py).

One class per module:
    param                  Param: one layer parameter and its binary layout
    layer_spec             LayerSpec: every answer about the new layer
    repository             Repository: read-only queries on the SADL tree
    prompter               Prompter: console questions with defaults
    questionnaire          Questionnaire: asks the questions, builds a LayerSpec
    source_file            SourceFile: a text file edited in memory (line endings kept)
    patch                  Patch: one insertion into an existing file
    cpp_templates          CppTemplates: C++ code of the new layer
    python_templates       PythonTemplates: converter, serializer, emitter, test model
    netron_templates       NetronTemplates: Netron OpSpec entry and decoder
    registration_patches   RegistrationPatches: every Patch for a LayerSpec
    change_set             ChangeSet: new files + patches, preview and write
    notes_writer           NotesWriter: NEW_LAYER_<Name>.md checklist
    verifier               Verifier: import, round-trip, flake8 and C++ build checks
    todo_scanner           TodoScanner: lists the remaining TODO markers
    cli                    CommandLineInterface
"""
