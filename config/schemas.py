from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


FEATURE_DEFAULTS = {
    "OverallQual": 6,
    "GrLivArea": 1500,
    "TotalBsmtSF": 1000,
    "FullBath": 2,
    "YearBuilt": 2005,
    "GarageCars": 2,
    "KitchenQual": "TA",
    "HouseStyle": "1Story",
    "LotArea": 8000,
    "Neighborhood": "CollgCr",
    "ExterQual": "TA",
    "BedroomAbvGr": 3,
}

FEATURE_FIELDS = tuple(FEATURE_DEFAULTS.keys())


class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    OverallQual: Optional[int] = Field(None, ge=1, le=10)
    GrLivArea: Optional[int] = None
    TotalBsmtSF: Optional[int] = None
    FullBath: Optional[int] = None
    YearBuilt: Optional[int] = None
    GarageCars: Optional[int] = None
    KitchenQual: Optional[str] = None
    HouseStyle: Optional[str] = None
    LotArea: Optional[int] = None
    Neighborhood: Optional[str] = None
    ExterQual: Optional[str] = None
    BedroomAbvGr: Optional[int] = None

    def missing_fields(self) -> List[str]:
        return [field_name for field_name in FEATURE_FIELDS if getattr(self, field_name) is None]

    def with_defaults(self) -> "ReviewedHouseFeatures":
        merged = {
            field_name: getattr(self, field_name)
            if getattr(self, field_name) is not None
            else FEATURE_DEFAULTS[field_name]
            for field_name in FEATURE_FIELDS
        }
        return ReviewedHouseFeatures(**merged)


class ReviewedHouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    OverallQual: int = Field(FEATURE_DEFAULTS["OverallQual"], ge=1, le=10)
    GrLivArea: int = Field(FEATURE_DEFAULTS["GrLivArea"])
    TotalBsmtSF: int = Field(FEATURE_DEFAULTS["TotalBsmtSF"])
    FullBath: int = Field(FEATURE_DEFAULTS["FullBath"])
    YearBuilt: int = Field(FEATURE_DEFAULTS["YearBuilt"])
    GarageCars: int = Field(FEATURE_DEFAULTS["GarageCars"])
    KitchenQual: str = Field(FEATURE_DEFAULTS["KitchenQual"])
    HouseStyle: str = Field(FEATURE_DEFAULTS["HouseStyle"])
    LotArea: int = Field(FEATURE_DEFAULTS["LotArea"])
    Neighborhood: str = Field(FEATURE_DEFAULTS["Neighborhood"])
    ExterQual: str = Field(FEATURE_DEFAULTS["ExterQual"])
    BedroomAbvGr: int = Field(FEATURE_DEFAULTS["BedroomAbvGr"])


class FieldCompleteness(BaseModel):
    value_present: bool
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[str] = None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    features: HouseFeatures
    missing_fields: List[str] = Field(
        default_factory=list,
        validation_alias="missing_features",
        serialization_alias="missing_features",
    )
    field_metadata: Dict[str, FieldCompleteness] = Field(default_factory=dict)
    is_complete: bool = False

    @classmethod
    def from_features(
        cls,
        features: HouseFeatures,
        field_metadata: Optional[Dict[str, FieldCompleteness]] = None,
    ) -> "ExtractionResult":
        missing_fields = features.missing_fields()
        metadata = field_metadata or {
            field_name: FieldCompleteness(value_present=getattr(features, field_name) is not None)
            for field_name in FEATURE_FIELDS
        }
        return cls(
            features=features,
            missing_fields=missing_fields,
            field_metadata=metadata,
            is_complete=not missing_fields,
        )


class ExtractionRequest(BaseModel):
    description: str = Field(..., min_length=1)

    prompt_version: str = Field("v1", min_length=1)


class TypicalSalePriceRange(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None


class SummaryStats(BaseModel):
    median_sale_price: Optional[float] = None
    mean_sale_price: Optional[float] = None
    typical_sale_price_range: TypicalSalePriceRange = Field(default_factory=TypicalSalePriceRange)


class StageFallback(BaseModel):
    used_fallback: bool = False
    error_code: Optional[str] = None
    user_message: Optional[str] = None
    error: Optional[str] = None


class FallbackMetadata(BaseModel):
    extraction: StageFallback = Field(default_factory=StageFallback)
    prediction: StageFallback = Field(default_factory=StageFallback)
    interpretation: StageFallback = Field(default_factory=StageFallback)


class PromptVersions(BaseModel):
    extraction: str = "v1"
    interpretation: str = "v1"


class PromptVariant(BaseModel):
    version: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    template: str = Field(..., min_length=1)


class PromptEvaluationExample(BaseModel):
    case_id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    notes: Optional[str] = None


class PromptEvaluationCriteria(BaseModel):
    requires_valid_json: bool = True
    requires_schema_validation: bool = True
    requires_missing_fields_alignment: bool = True
    completeness_rule: str = (
        "Pass when extracted values are grounded in the description and missing_fields "
        "matches the feature keys left as null."
    )


class ExtractionPromptCatalogResponse(BaseModel):
    default_version: str = "v1"
    winner_selected: bool = False
    selection_notes: str = (
        "Three Stage 1 prompt variants are defined, but no winner is selected in the "
        "backend because evaluation will happen separately."
    )
    available_versions: List[str] = Field(default_factory=list)
    prompts: List[PromptVariant] = Field(default_factory=list)
    sample_inputs: List[PromptEvaluationExample] = Field(default_factory=list)
    evaluation_criteria: PromptEvaluationCriteria = Field(default_factory=PromptEvaluationCriteria)


class ExtractionStageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    extraction: ExtractionResult = Field(
        ...,
        validation_alias="extracted_data",
        serialization_alias="extracted_data",
    )
    fallback: StageFallback = Field(default_factory=StageFallback)
    prompt_version: str = "v1"


class PredictionRequest(BaseModel):
    reviewed_features: ReviewedHouseFeatures


class PredictionStageResponse(BaseModel):
    prediction: float
    reviewed_features: ReviewedHouseFeatures
    summary_stats: SummaryStats = Field(default_factory=SummaryStats)
    fallback: StageFallback = Field(default_factory=StageFallback)


class InterpretationRequest(BaseModel):
    reviewed_features: ReviewedHouseFeatures
    prediction: float
    summary_stats: Optional[SummaryStats] = None
    prompt_version: str = Field("v1", min_length=1)


class InterpretationStageResponse(BaseModel):
    interpretation: str
    summary_stats: SummaryStats = Field(default_factory=SummaryStats)
    fallback: StageFallback = Field(default_factory=StageFallback)
    prompt_version: str = "v1"


class AgentRequest(BaseModel):
    description: str = Field(..., min_length=1)
    reviewed_features: Optional[ReviewedHouseFeatures] = None
    extraction_prompt_version: str = Field("v1", min_length=1)
    interpretation_prompt_version: str = Field("v1", min_length=1)


class AgentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prediction: float
    interpretation: str
    reviewed_features: ReviewedHouseFeatures
    summary_stats: SummaryStats = Field(default_factory=SummaryStats)
    fallback: FallbackMetadata = Field(default_factory=FallbackMetadata)
    prompt_versions: PromptVersions = Field(default_factory=PromptVersions)
    extraction: ExtractionResult = Field(
        ...,
        validation_alias="extracted_data",
        serialization_alias="extracted_data",
    )