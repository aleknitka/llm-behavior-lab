from collections.abc import Mapping

import pandas as pd
import pingouin

from llm_behavior_lab.scoring.models import (
    ScaleReliabilityRecord,
    ScaleReliabilityStatus,
)


def calculate_reliability(
    *,
    scale_id: str,
    scoring_model_id: str,
    scoring_model_version: str,
    item_values: Mapping[str, Mapping[str, float]],
    metadata: dict[str, object] | None = None,
) -> ScaleReliabilityRecord:
    """Calculate complete-case Cronbach alpha for one scale and subject group."""
    frame = pd.DataFrame.from_dict(
        {subject: dict(values) for subject, values in item_values.items()},
        orient="index",
    ).dropna(axis=0, how="any")
    subject_count, item_count = frame.shape
    if item_count < 2:
        return ScaleReliabilityRecord(
            scale_id=scale_id,
            scoring_model_id=scoring_model_id,
            scoring_model_version=scoring_model_version,
            status=ScaleReliabilityStatus.NOT_COMPUTABLE,
            subject_count=subject_count,
            item_count=item_count,
            reason="at least two items are required",
            metadata=metadata or {},
        )
    if subject_count < 2:
        return ScaleReliabilityRecord(
            scale_id=scale_id,
            scoring_model_id=scoring_model_id,
            scoring_model_version=scoring_model_version,
            status=ScaleReliabilityStatus.NOT_COMPUTABLE,
            subject_count=subject_count,
            item_count=item_count,
            reason="at least two subjects are required",
            metadata=metadata or {},
        )
    if frame.sum(axis=1).var(ddof=1) == 0:
        return ScaleReliabilityRecord(
            scale_id=scale_id,
            scoring_model_id=scoring_model_id,
            scoring_model_version=scoring_model_version,
            status=ScaleReliabilityStatus.NOT_COMPUTABLE,
            subject_count=subject_count,
            item_count=item_count,
            reason="total score variance must be positive",
            metadata=metadata or {},
        )
    alpha, confidence_interval = pingouin.cronbach_alpha(data=frame, nan_policy="listwise")
    return ScaleReliabilityRecord(
        scale_id=scale_id,
        scoring_model_id=scoring_model_id,
        scoring_model_version=scoring_model_version,
        status=ScaleReliabilityStatus.COMPLETED,
        subject_count=subject_count,
        item_count=item_count,
        alpha=float(alpha),
        confidence_interval=(float(confidence_interval[0]), float(confidence_interval[1])),
        metadata=metadata or {},
    )
