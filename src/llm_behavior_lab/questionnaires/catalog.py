from collections.abc import Mapping

from llm_behavior_lab.questionnaires.base.scale import Questionnaire
from llm_behavior_lab.questionnaires.bfi10 import BFI_10
from llm_behavior_lab.questionnaires.consumer_involvement import (
    build_consumer_involvement_questionnaire,
)
from llm_behavior_lab.questionnaires.pdmi import PDMI


def resolve_questionnaire(
    questionnaire_id: str,
    parameters: Mapping[str, str] | None = None,
) -> Questionnaire:
    """Resolve a coded questionnaire by stable ID and validated parameters."""
    if questionnaire_id == BFI_10.id:
        return BFI_10
    if questionnaire_id == PDMI.id:
        return PDMI
    if questionnaire_id == "consumer_involvement":
        target = (parameters or {}).get("target")
        if target is None:
            raise ValueError("consumer_involvement requires questionnaire parameter 'target'")
        return build_consumer_involvement_questionnaire(target)
    raise ValueError(f"unknown questionnaire_id: {questionnaire_id}")


def questionnaire_ids() -> list[str]:
    """Return stable IDs for all coded questionnaire families."""
    return [BFI_10.id, PDMI.id, "consumer_involvement"]
