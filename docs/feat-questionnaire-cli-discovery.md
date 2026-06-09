# Questionnaire Discovery and CLI Selection

## Problem

The package contains multiple questionnaire definitions, but the command-line interface
always runs BFI-10. Users must write Python to discover other instruments or supply
parameters for questionnaires such as the target-dependent Consumer Involvement Scale.

## Proposed Feature

Expose a small, explicit questionnaire catalog through the CLI. Users should be able to
list available questionnaires, inspect their metadata and required parameters, and
select one for a standard batch or protocol run.

## User and Research Value

- Makes existing questionnaires usable without custom Python.
- Provides a predictable path for adding future instruments.
- Reduces accidental runs against the wrong questionnaire or version.
- Surfaces references, licences, response formats, and required parameters before a run.

## Core Capabilities

- List available questionnaire IDs, names, versions, languages, and parameter needs.
- Select a questionnaire with `--questionnaire`.
- Show detailed questionnaire metadata without invoking a provider.
- Pass validated parameters to target-dependent questionnaire builders.
- Use the selected questionnaire for both normal and protocol runs.
- Return clear errors for unknown IDs, missing parameters, and invalid values.

## In Scope

- The currently coded BFI-10, PDMI, and Consumer Involvement questionnaires.
- A simple explicit mapping maintained alongside questionnaire package exports.
- CLI commands or flags for listing and describing questionnaires.
- Repeatable `KEY=VALUE` questionnaire parameter input.
- Backward-compatible default selection of BFI-10.

## Out of Scope

- Dynamic plugin loading or filesystem module discovery.
- Downloading questionnaire definitions from the internet.
- A graphical questionnaire editor.
- Silently accepting instruments with missing citation or licence metadata.

## Acceptance Criteria

- `--list-questionnaires` prints all supported questionnaire IDs and exits without
  creating experiment files or contacting a provider.
- `--questionnaire bfi10` preserves the current default run behavior.
- Static questionnaires can be selected by stable ID.
- Parameterized questionnaires reject missing required parameters before client setup.
- Description output includes version, language, reference, licence, item count, and
  scale IDs.
- The selected questionnaire ID and version appear in run and response records.
- CLI tests cover listing, selection, description, parameter validation, unknown IDs,
  and the backward-compatible default.

## Dependencies and Risks

- The catalog should stay explicit and small rather than becoming an unnecessary plugin
  framework.
- Stable questionnaire IDs become part of the public CLI interface.
- Builder parameter names and validation need a consistent convention.
- Licence information must remain visible when instruments are exposed more broadly.

