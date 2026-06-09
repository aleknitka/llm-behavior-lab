# Multi-Model Benchmark Experiments

## Problem

The current workflow configures one model and provider per invocation. Comparing models,
temperatures, capability settings, or prompt variants requires separate commands and
manual coordination, making it easy for persona samples or protocol assignments to
drift between runs.

## Proposed Feature

Introduce a benchmark experiment specification that runs one shared persona and
questionnaire design against multiple named model configurations. Each configuration
produces an independent run while retaining common benchmark and subject identifiers.

## User and Research Value

- Enables controlled model-to-model comparisons.
- Reuses identical personas and protocol assignments across configurations.
- Reduces manual setup and accidental experimental confounds.
- Produces a coherent dataset for robustness and sensitivity analysis.

## Core Capabilities

- Define multiple named model configurations in one validated specification.
- Configure provider URL, model, temperature, timeout, seed, and capability flags.
- Generate personas and protocol assignments once, then reuse them for every run.
- Track a stable benchmark ID and configuration ID in run and response metadata.
- Continue other configurations when one provider or model fails, according to policy.
- Produce a benchmark manifest and combined comparison table.
- Support questionnaire and protocol runs through the same benchmark abstraction.

## In Scope

- OpenAI-compatible hosted and local endpoints.
- Multiple models on one provider or across different providers.
- Shared personas, questionnaire versions, protocols, and base seeds.
- Per-configuration run directories using the existing storage layout.
- Basic comparison summaries for completion, invalid-response, and answer distributions.

## Out of Scope

- Model training or fine-tuning.
- Automatic model discovery from provider APIs.
- Price optimization or provider account management.
- Treating nondeterministic provider outputs as perfectly reproducible.

## Acceptance Criteria

- A benchmark specification with two configurations creates two independent run records
  under one experiment.
- All configurations use the same subject IDs and protocol assignments.
- Every response identifies its benchmark configuration and provider snapshot.
- Configuration validation rejects duplicate IDs and ambiguous model settings.
- A failed configuration does not corrupt successful configuration outputs.
- Combined exports align results by subject, questionnaire item, condition, and
  configuration.
- Tests verify shared inputs, metadata isolation, partial failure, and deterministic
  configuration ordering.

## Dependencies and Risks

- Provider credentials should remain runtime configuration and must not be persisted.
- Different providers expose inconsistent seed, structured-output, and logprob behavior.
- Benchmark scale can multiply request volume and cost quickly.
- Comparisons require clear disclosure when configurations use different capabilities.

