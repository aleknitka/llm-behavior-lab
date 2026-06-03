from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------


class Gender(StrEnum):
    FEMALE = "female"
    MALE = "male"
    NON_BINARY = "non_binary"


class EducationLevel(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    VOCATIONAL = "vocational"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"
    OTHER = "other"


class EmploymentStatus(StrEnum):
    STUDENT = "student"
    EMPLOYED_FULL_TIME = "employed_full_time"
    EMPLOYED_PART_TIME = "employed_part_time"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    RETIRED = "retired"
    CAREGIVER = "caregiver"
    OTHER = "other"


class AffluenceLevel(StrEnum):
    VERY_LOW = "very_low"
    LOW = "low"
    LOWER_MIDDLE = "lower_middle"
    MIDDLE = "middle"
    UPPER_MIDDLE = "upper_middle"
    HIGH = "high"
    VERY_HIGH = "very_high"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"
    UNKNOWN = "unknown"


class FamilyStatus(StrEnum):
    SINGLE = "single"
    IN_RELATIONSHIP = "in_relationship"
    MARRIED = "married"
    CIVIL_PARTNERSHIP = "civil_partnership"
    SEPARATED = "separated"
    DIVORCED = "divorced"
    WIDOWED = "widowed"
    COHABITING = "cohabiting"
    OTHER = "other"


class Urbanicity(StrEnum):
    URBAN_CORE = "urban_core"
    URBAN = "urban"
    SUBURBAN = "suburban"
    TOWN = "town"
    RURAL = "rural"
    REMOTE_RURAL = "remote_rural"
    OTHER = "other"


class EuropeanCountry(StrEnum):
    # Core European sovereign states
    ALBANIA = "AL"
    ANDORRA = "AD"
    AUSTRIA = "AT"
    BELARUS = "BY"
    BELGIUM = "BE"
    BOSNIA_AND_HERZEGOVINA = "BA"
    BULGARIA = "BG"
    CROATIA = "HR"
    CYPRUS = "CY"
    CZECHIA = "CZ"
    DENMARK = "DK"
    ESTONIA = "EE"
    FINLAND = "FI"
    FRANCE = "FR"
    GERMANY = "DE"
    GREECE = "GR"
    HUNGARY = "HU"
    ICELAND = "IS"
    IRELAND = "IE"
    ITALY = "IT"
    KOSOVO = "XK"  # user-assigned; not official ISO 3166-1
    LATVIA = "LV"
    LIECHTENSTEIN = "LI"
    LITHUANIA = "LT"
    LUXEMBOURG = "LU"
    MALTA = "MT"
    MOLDOVA = "MD"
    MONACO = "MC"
    MONTENEGRO = "ME"
    NETHERLANDS = "NL"
    NORTH_MACEDONIA = "MK"
    NORWAY = "NO"
    POLAND = "PL"
    PORTUGAL = "PT"
    ROMANIA = "RO"
    RUSSIA = "RU"
    SAN_MARINO = "SM"
    SERBIA = "RS"
    SLOVAKIA = "SK"
    SLOVENIA = "SI"
    SPAIN = "ES"
    SWEDEN = "SE"
    SWITZERLAND = "CH"
    TURKEY = "TR"
    UKRAINE = "UA"
    UNITED_KINGDOM = "GB"
    VATICAN_CITY = "VA"

    # Commonly needed European territories / special cases
    ALAND_ISLANDS = "AX"
    FAROE_ISLANDS = "FO"
    GIBRALTAR = "GI"
    GUERNSEY = "GG"
    ISLE_OF_MAN = "IM"
    JERSEY = "JE"
    SVALBARD_AND_JAN_MAYEN = "SJ"


class ScaleScore(BaseModel):
    """
    Generic scale score.

    Example:
        risk_tolerance = ScaleScore(
            value=4.2,
            min_value=1,
            max_value=7,
            instrument="DOSPERT-short"
        )
    """

    value: float
    min_value: float
    max_value: float
    instrument: str | None = None

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float, info: Any) -> float:
        return value


class LikertItem(BaseModel):
    item_id: str
    text: str
    value: int = Field(ge=1, le=7)
    reverse_scored: bool = False


class FreeTextResponse(BaseModel):
    prompt: str
    response: str


# ---------------------------------------------------------------------
# Participant profile
# ---------------------------------------------------------------------


class Demographics(BaseModel):
    age: int | None = Field(default=None, ge=0, le=120)
    gender: Gender | None = None
    education_level: EducationLevel | None = None
    field_of_study: str | None = None
    employment_status: EmploymentStatus | None = None
    affluence_level: AffluenceLevel | None = None
    occupation: str | None = None
    country: EuropeanCountry | None = None
    region: str | None = None
    urbanicity: Urbanicity | None = None
    family_status: FamilyStatus | None = None
    household_size: int | None = Field(default=None, ge=1)
    has_children: bool | None = None
    number_of_dependants: int | None = Field(default=None, ge=0)


class SocioEconomicContext(BaseModel):
    income_band: str | None = None
    perceived_financial_security: ScaleScore | None = None
    financial_stress: ScaleScore | None = None
    savings_buffer_months: float | None = Field(default=None, ge=0)
    debt_burden: ScaleScore | None = None
    time_availability: ScaleScore | None = None
    caregiving_burden: ScaleScore | None = None
    digital_access: ScaleScore | None = None
    social_support: ScaleScore | None = None


class Psychographics(BaseModel):
    # Values and motivations
    security_value: ScaleScore | None = None
    achievement_value: ScaleScore | None = None
    autonomy_value: ScaleScore | None = None
    tradition_value: ScaleScore | None = None
    benevolence_value: ScaleScore | None = None
    status_motivation: ScaleScore | None = None
    convenience_motivation: ScaleScore | None = None
    curiosity_motivation: ScaleScore | None = None

    # Personality, e.g. Big Five
    openness: ScaleScore | None = None
    conscientiousness: ScaleScore | None = None
    extraversion: ScaleScore | None = None
    agreeableness: ScaleScore | None = None
    neuroticism: ScaleScore | None = None

    # Attitudes and identities
    trust_in_institutions: ScaleScore | None = None
    trust_in_experts: ScaleScore | None = None
    trust_in_technology: ScaleScore | None = None
    privacy_concern: ScaleScore | None = None
    environmental_concern: ScaleScore | None = None
    identity_labels: list[str] = Field(default_factory=list)


class DecisionStyle(BaseModel):
    risk_tolerance: ScaleScore | None = None
    loss_aversion: ScaleScore | None = None
    time_preference: ScaleScore | None = None
    need_for_cognition: ScaleScore | None = None
    numeracy: ScaleScore | None = None
    cognitive_reflection: ScaleScore | None = None
    maximising_tendency: ScaleScore | None = None
    ambiguity_tolerance: ScaleScore | None = None
    impulsivity: ScaleScore | None = None


class SituationalState(BaseModel):
    mood_valence: ScaleScore | None = None
    stress: ScaleScore | None = None
    fatigue: ScaleScore | None = None
    sleep_quality: ScaleScore | None = None
    current_time_pressure: ScaleScore | None = None
    perceived_stakes: ScaleScore | None = None


class Participant(BaseModel):
    participant_id: str
    demographics: Demographics = Field(default_factory=Demographics)
    socioeconomic: SocioEconomicContext = Field(default_factory=SocioEconomicContext)
    psychographics: Psychographics = Field(default_factory=Psychographics)
    decision_style: DecisionStyle = Field(default_factory=DecisionStyle)
    baseline_state: SituationalState = Field(default_factory=SituationalState)
    custom_attributes: dict[str, Any] = Field(default_factory=dict)
