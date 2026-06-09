from collections.abc import Mapping

from pydantic import BaseModel

from llm_behavior_lab.questionnaires.base.scale import Questionnaire
from llm_behavior_lab.questionnaires.bfi10 import BFI_10
from llm_behavior_lab.questionnaires.consumer_involvement import (
    CONSUMER_INVOLVEMENT_TEMPLATE,
    build_consumer_involvement_questionnaire,
)
from llm_behavior_lab.questionnaires.pdmi import PDMI


class QuestionnaireParameterSpec(BaseModel):
    """Describe one parameter required to construct a questionnaire."""

    name: str
    required: bool
    description: str
    example: str | None = None


class QuestionnaireDescriptor(BaseModel):
    """Provider-independent metadata exposed by questionnaire discovery."""

    id: str
    name: str
    version: str
    language: str | None
    reference: str
    licence: str
    source_url: str | None
    item_count: int
    scale_ids: list[str]
    scoring_model_ids: list[str]
    response_format_types: list[str]
    parameters: list[QuestionnaireParameterSpec]


def _describe(
    questionnaire: Questionnaire,
    *,
    parameters: list[QuestionnaireParameterSpec] | None = None,
) -> QuestionnaireDescriptor:
    return QuestionnaireDescriptor(
        id=questionnaire.id,
        name=questionnaire.name,
        version=questionnaire.version,
        language=questionnaire.language,
        reference=questionnaire.reference,
        licence=questionnaire.licence,
        source_url=questionnaire.metadata.get("source_url"),
        item_count=len(questionnaire.items),
        scale_ids=[scale.id for scale in questionnaire.scales],
        scoring_model_ids=[model.id for model in questionnaire.scoring_models],
        response_format_types=sorted(
            {str(item.response_format.type) for item in questionnaire.items}
        ),
        parameters=parameters or [],
    )


_DESCRIPTORS = {
    BFI_10.id: _describe(BFI_10),
    CONSUMER_INVOLVEMENT_TEMPLATE.id: _describe(
        CONSUMER_INVOLVEMENT_TEMPLATE,
        parameters=[
            QuestionnaireParameterSpec(
                name="target",
                required=True,
                description="Product or service category inserted into each item.",
                example="meal delivery services",
            )
        ],
    ),
    PDMI.id: _describe(PDMI),
}


def list_questionnaires() -> list[QuestionnaireDescriptor]:
    """Return all coded questionnaires in deterministic stable-ID order."""
    return [_DESCRIPTORS[questionnaire_id] for questionnaire_id in sorted(_DESCRIPTORS)]


def describe_questionnaire(questionnaire_id: str) -> QuestionnaireDescriptor:
    """Return discovery metadata for one exact stable questionnaire ID."""
    try:
        return _DESCRIPTORS[questionnaire_id]
    except KeyError as error:
        raise ValueError(f"unknown questionnaire_id: {questionnaire_id}") from error


def resolve_questionnaire(
    questionnaire_id: str,
    parameters: Mapping[str, str] | None = None,
) -> Questionnaire:
    """Resolve a coded questionnaire by exact stable ID and validated parameters."""
    supplied_parameters = dict(parameters or {})
    describe_questionnaire(questionnaire_id)

    if questionnaire_id == CONSUMER_INVOLVEMENT_TEMPLATE.id:
        unknown = sorted(set(supplied_parameters) - {"target"})
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(
                f"consumer_involvement received unknown questionnaire parameters: {names}"
            )
        target = supplied_parameters.get("target")
        if target is None:
            raise ValueError("consumer_involvement requires questionnaire parameter 'target'")
        return build_consumer_involvement_questionnaire(target)

    if supplied_parameters:
        raise ValueError(f"{questionnaire_id} does not accept questionnaire parameters")
    if questionnaire_id == BFI_10.id:
        return BFI_10
    return PDMI


def questionnaire_ids() -> list[str]:
    """Return exact stable IDs for all coded questionnaire families."""
    return [descriptor.id for descriptor in list_questionnaires()]
