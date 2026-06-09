# Questionnaire Scoring

The scoring layer converts typed item responses into versioned scale scores without
modifying raw response records.

## Model Structure

A scored questionnaire defines:

1. `Item` objects with bounded scalar response formats or explicit choice options.
2. `Scale` objects containing ordered `ItemMapping` entries.
3. A versioned `ScoringModel` with one `ScaleScoringRule` per output scale.

Stable item, scale, and scoring-model IDs are part of the persisted data contract.

## Scalar Values

- `LikertAnswerValue.value` is used directly.
- `NumericAnswerValue.value` is supported; reversed numeric items require both bounds.
- `SingleChoiceAnswerValue.option_id` requires a numeric `ItemMapping.scoring_key`.
- Text and multiple-choice answers are not scalar in the first scoring version.

For a reverse-scored bounded item:

```text
keyed value = minimum + maximum - raw value
```

Computed scale direction must be documented. Current coded models use
"higher means more of the named construct."

## Transformations

- `sum`: add keyed item values; all weights must be one.
- `mean`: arithmetic mean of keyed values; all weights must be one.
- `weighted_mean`: divide the weighted keyed total by the sum of positive weights.

Output bounds document the intended score range. Interpretation bands are optional,
non-overlapping intervals attached to completed scores.

## Missing Data

The first scoring version uses complete-scale scoring. If any mapped item is absent,
failed, invalid, or unscorable, that subject-scale record is `unscorable` and lists the
affected item IDs. It does not impute, prorate, or silently drop items.

## Reliability

Cronbach's alpha is calculated with `pingouin.cronbach_alpha` over complete keyed item
matrices using listwise deletion. Records also store the confidence interval and item
and subject counts.

Reliability is `not_computable` when:

- the scale has fewer than two items;
- fewer than two complete subjects are available; or
- total-score variance is zero.

Published alpha values belong in source metadata. Experiment-computed alpha describes
the current simulated sample and is persisted separately.

## Adding Scoring to a Questionnaire

### 1. Confirm source authority

Identify the source's item membership, response coding, reversals, aggregation, score
direction, and treatment of missing data. Do not infer an executable rule from factor
membership alone.

### 2. Add scale mappings

```python
Scale(
    id="example_trait",
    name="Example Trait",
    construct="Example construct",
    item_mappings=[
        ItemMapping(item_id="example_01", reverse_scored=True),
        ItemMapping(item_id="example_02"),
    ],
)
```

Use `scoring_key` for choice options and `weight` only for a weighted mean.

### 3. Add a versioned scoring model

```python
ScoringModel(
    id="default",
    name="Source mean score",
    version="1.0",
    provenance="source_defined",
    scale_rules=[
        ScaleScoringRule(
            scale_id="example_trait",
            transformation=Transformation.MEAN,
            output_min=1,
            output_max=5,
        )
    ],
)
```

Use `provenance="project_defined"` only when the rule is an explicit project
convention, and explain that choice in questionnaire metadata and documentation.

### 4. Test known answers

Add tests for minimum, midpoint, and maximum score combinations; every reversal;
choice mappings; missing responses; and model validation. For multi-item scales, test a
known reliability matrix independently from the library call.

### 5. Document interpretation

State the score direction, transformation, output range, source citation, provenance,
and limitations. If no authoritative aggregation rule exists, leave
`scoring_models=[]`. The Consumer Involvement definition demonstrates this deliberate
unscored state.

## Current Inventories

- BFI-10: five two-item mean scores. Positive-worded items are numerically reversed
  because the coded response format uses `1 = Strongly agree` and
  `5 = Strongly disagree`.
- PDMI: eight source subscales and emotional/reasoned composite means.
- Consumer Involvement: factor mappings only; no executable scoring model.

