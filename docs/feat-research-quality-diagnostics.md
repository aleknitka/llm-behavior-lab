# Research Quality Diagnostics

## Problem

Successful parsing does not guarantee useful simulated questionnaire data. Models may
repeat one answer, ignore reverse-worded items, produce unstable answers across
iterations, or respond differently because of item order. The current analysis layer
does not identify these patterns.

## Proposed Feature

Add a diagnostic analysis suite that evaluates response completeness, variability,
internal consistency signals, and repeated-measure stability. Diagnostics should flag
potential quality problems without silently excluding records or claiming human
psychometric validity.

## User and Research Value

- Separates technically valid responses from potentially low-information behavior.
- Makes model and protocol comparisons easier to audit.
- Provides evidence for exclusion rules before formal statistical analysis.
- Helps detect prompt, questionnaire, and provider-specific failure patterns.

## Core Capabilities

- Summarize completed, invalid, and failed response rates.
- Detect straight-lining and unusually low within-subject response entropy.
- Compare paired reverse-worded items where questionnaire metadata supports it.
- Measure test-retest agreement across protocol iterations.
- Compare response distributions by model, condition, and item position.
- Flag excessive raw-response formatting violations or fallback behavior.
- Export subject-level flags and experiment-level diagnostic summaries.

## In Scope

- Rule-based diagnostics with configurable, documented thresholds.
- Likert and single-choice questionnaires as the initial supported formats.
- Protocol iteration and factor metadata already stored in response records.
- Diagnostic outputs that preserve the evidence behind every flag.
- Aggregate summaries that avoid modifying source response files.

## Out of Scope

- Automatically deleting or excluding responses.
- Diagnosing psychological traits or disorders.
- Claiming that human quality-control thresholds transfer directly to LLMs.
- Advanced causal inference or automatic publication-ready statistical conclusions.

## Acceptance Criteria

- Diagnostics can run against one response file or a complete run directory.
- Every flag includes a diagnostic ID, measured value, threshold, and affected IDs.
- Thresholds are configurable and persisted in diagnostic output metadata.
- Missing metadata causes a diagnostic to report `not_applicable`, not a false failure.
- Re-running diagnostics with the same inputs and settings is deterministic.
- Protocol iterations can be compared by base subject and condition.
- Tests cover clean data, straight-lined data, invalid records, missing metadata, and
  unstable repeated responses.

## Dependencies and Risks

- Questionnaire definitions need explicit reverse-item or trait relationships for some
  checks.
- Thresholds are study-design choices and should not become undocumented defaults.
- Small samples can make aggregate distribution warnings unreliable.
- Diagnostic terminology must clearly distinguish warning signals from validity claims.

