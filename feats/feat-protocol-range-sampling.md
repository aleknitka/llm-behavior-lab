# Deterministic Protocol Range Sampling

## Problem

Protocol factors currently enumerate exact values. Numeric comparisons such as
younger adults aged 20-35 versus older adults aged 60-75 require manually
listing every age or choosing one representative value. Listing every value
changes the factorial cell count, while representative values understate
within-condition variation.

## Proposed Feature

Allow numeric protocol levels to define an inclusive range and deterministic
sampling policy. Each expanded persona receives one sampled value for the range
level using a seed derived from the protocol seed, base subject, factor level,
and iteration.

Example:

```json
{
  "name": "age_group",
  "field": "age",
  "levels": [
    {
      "id": "younger",
      "range": {"minimum": 20, "maximum": 35}
    },
    {
      "id": "older",
      "range": {"minimum": 60, "maximum": 75}
    }
  ]
}
```

## Required Behavior

- Preserve the existing exact `value` form for categorical and fixed numeric
  levels.
- Require exactly one of `value` or `range` on each factor level.
- Support integer ranges first; reject ranges for non-numeric demographic
  fields.
- Validate both bounds through the target demographic field and require
  `minimum <= maximum`.
- Sample inclusively and deterministically without relying on global random
  state.
- Keep the condition ID tied to the range level ID, not the sampled value.
- Persist the sampled value in assignments so reruns and analysis reproduce the
  exact persona.
- Keep the number of factorial conditions unchanged: two age ranges remain two
  conditions.

## Seed Derivation

Derive a local seed from a stable hash of:

```text
protocol seed : base subject ID : factor name : level ID : iteration index
```

This isolates sampling from factor ordering and unrelated random draws. Adding
another factor should not silently change already defined range samples.

## Acceptance Criteria

- Identical protocol inputs and experiment IDs produce identical sampled values
  and subject assignments.
- Different iterations can sample different values within the same range.
- Every sampled value is within its inclusive bounds and validates against the
  demographic model.
- Reordering factors does not change a range factor's sampled values for the
  same base subject, level, and iteration.
- Protocol JSON and assignment JSONL retain the declared range and realized
  value respectively.
- Tests cover invalid bounds, unsupported fields, deterministic replay, factor
  reordering, and inclusive endpoint sampling.
