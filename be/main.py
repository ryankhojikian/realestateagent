import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.schemas import (
    AgentRequest,
    AgentResponse,
    ExtractionPromptCatalogResponse,
    ExtractionRequest,
    ExtractionResult,
    ExtractionStageResponse,
    FEATURE_FIELDS,
    FallbackMetadata,
    FieldCompleteness,
    HouseFeatures,
    InterpretationRequest,
    InterpretationStageResponse,
    PredictionRequest,
    PredictionStageResponse,
    PromptEvaluationExample,
    PromptVariant,
    PromptVersions,
    ReviewedHouseFeatures,
    StageFallback,
    SummaryStats,
)

MODEL_PATH = ROOT_DIR / "model" / "house_price_model.pkl"
METRICS_PATH = ROOT_DIR / "model" / "training_metrics.json"
ENV_PATH = ROOT_DIR / "config" / ".env"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_PREDICTION_FALLBACK = 195000.0
DEFAULT_EXTRACTION_PROMPT_VERSION = "v1"
DEFAULT_INTERPRETATION_PROMPT_VERSION = "v3"
SUPPORTED_LLM_PROVIDERS = {"ollama", "gemini"}
DEFAULT_LLM_PROVIDER = "gemini"

EXTRACTION_PROMPTS = {
    "v1": """
You extract structured property features from a home description.

USER TEXT:
{user_text}

Return ONLY valid JSON with this structure:
{
  "features": {
    "OverallQual": int | null,
    "GrLivArea": int | null,
    "TotalBsmtSF": int | null,
    "FullBath": int | null,
    "YearBuilt": int | null,
    "GarageCars": int | null,
    "KitchenQual": str | null,
    "HouseStyle": str | null,
    "LotArea": int | null,
    "Neighborhood": str | null,
    "ExterQual": str | null,
    "BedroomAbvGr": int | null
  },
  "missing_fields": ["field names that were not mentioned"],
  "field_metadata": {
    "OverallQual": {"confidence": 0.0, "source": "optional snippet"},
    "GrLivArea": {"confidence": 0.0, "source": "optional snippet"}
  }
}

Rules:
1. Use null when a field is missing from the text.
2. Do not invent values and do not apply defaults.
3. missing_fields must match the fields that are null.
4. confidence is optional, but if included it must be between 0 and 1.
5. Preserve the original feature names exactly as shown in the schema.
""".strip(),
    "v2": """
You are doing evidence-first information extraction for a house-pricing pipeline.

USER TEXT:
{user_text}

Task:
- Extract only facts that are explicitly supported by the text.
- If the text gives a synonym or paraphrase, map it to the closest supported field.
- If a value is uncertain, leave it null instead of guessing.

Return ONLY valid JSON:
{
  "features": {
    "OverallQual": int | null,
    "GrLivArea": int | null,
    "TotalBsmtSF": int | null,
    "FullBath": int | null,
    "YearBuilt": int | null,
    "GarageCars": int | null,
    "KitchenQual": str | null,
    "HouseStyle": str | null,
    "LotArea": int | null,
    "Neighborhood": str | null,
    "ExterQual": str | null,
    "BedroomAbvGr": int | null
  },
  "missing_fields": ["field names with null values"],
  "field_metadata": {
    "OverallQual": {"confidence": 0.0, "source": "direct quote or snippet"},
    "GrLivArea": {"confidence": 0.0, "source": "direct quote or snippet"}
  }
}

Evaluation rules:
1. Every non-null feature must have textual evidence in the user text.
2. missing_fields must include every null field and no populated field.
3. Do not output extra keys.
4. Keep the response machine-parseable JSON only.
""".strip(),
    "v3": """
Convert the property description into a strict extraction object for downstream validation.

Supported fields:
- OverallQual: overall quality score from 1-10
- GrLivArea: above-ground living area in square feet
- TotalBsmtSF: basement area in square feet
- FullBath: number of full bathrooms
- YearBuilt: construction year
- GarageCars: garage capacity in cars
- KitchenQual: kitchen quality label
- HouseStyle: house style label
- LotArea: lot size in square feet
- Neighborhood: neighborhood name
- ExterQual: exterior quality label
- BedroomAbvGr: bedrooms above grade

USER TEXT:
{user_text}

Instructions:
1. Work silently and infer nothing beyond the description.
2. Normalize values only when the meaning is explicit.
3. If the description mentions a feature indirectly but not specifically enough, set it to null.
4. Return ONLY this JSON object:
{
  "features": {
    "OverallQual": int | null,
    "GrLivArea": int | null,
    "TotalBsmtSF": int | null,
    "FullBath": int | null,
    "YearBuilt": int | null,
    "GarageCars": int | null,
    "KitchenQual": str | null,
    "HouseStyle": str | null,
    "LotArea": int | null,
    "Neighborhood": str | null,
    "ExterQual": str | null,
    "BedroomAbvGr": int | null
  },
  "missing_fields": [],
  "field_metadata": {}
}
5. Ensure missing_fields exactly lists the null-valued feature names.
""".strip()
}

STAGE1_EVALUATION_CASES = (
    PromptEvaluationExample(
        case_id="sample_1",
        description=(
            "Updated 2-story home in NridgHt with 2,250 square feet above grade, "
            "built in 2008. It has 4 bedrooms, 3 full baths, a 2-car garage, "
            "excellent exterior quality, a good kitchen, 1,100 sqft basement, "
            "and sits on a 9,200 sqft lot."
        ),
        notes="High-completeness example with mostly explicit numeric signals.",
    ),
    PromptEvaluationExample(
        case_id="sample_2",
        description=(
            "Cozy 1-story starter house in OldTown with 1,180 sqft of living space. "
            "Three bedrooms, one full bath, built in 1954, modest lot, and no basement "
            "details were provided. The seller mentions a detached single-car garage."
        ),
        notes="Partial-information example with some fields missing or described loosely.",
    ),
    PromptEvaluationExample(
        case_id="sample_3",
        description=(
            "Spacious family property near StoneBrk featuring a large open layout, "
            "high-end finishes, and room for multiple cars in the garage. "
            "The listing highlights strong curb appeal and a modern kitchen but does not "
            "state square footage, year built, bathroom count, or lot size."
        ),
        notes="Ambiguous example meant to test restraint and correct null handling.",
    ),
)

INTERPRETATION_PROMPTS = {
    "v1": """
You are a real-estate pricing analyst explaining a machine-learning valuation to a non-technical audience.

--- PROPERTY FEATURES ---
{reviewed_features}

--- VALUATION CONTEXT ---
Predicted sale price : ${predicted_price:,.0f}
Median sale price    : {median_sale_price}
Mean sale price      : {mean_sale_price}
Typical price range  : {typical_range_low} – {typical_range_high}

--- QUALITY CODE REFERENCE ---
ExterQual / KitchenQual values: Ex = Excellent | Gd = Good | TA = Typical/Average | Fa = Fair | Po = Poor

--- INSTRUCTIONS ---
Write a clear, 3–5 sentence explanation structured as follows:
1. Open with a one-sentence market-positioning statement: is the predicted price below, within, or above the typical range, and by roughly how much?
2. Identify the two or three features with the strongest positive influence on the price (overall quality, living area, year built, and garage capacity are the highest-weight signals in the model).
3. Note any features that limit or cap the valuation (e.g. below-average quality codes, small lot, older build year, missing basement area).
4. Close with a brief confidence note referencing the model's typical error margin (~$17–27k RMSE on held-out data).

Rules:
- Write in plain English suitable for a home buyer or seller.
- Do not fabricate features or invent numbers not present in the data above.
- Do not reproduce the raw JSON; refer to field values in natural language.
- Keep the response under 150 words.
""".strip(),

    "v2": """
You are a quantitative real-estate analyst producing a structured valuation brief.

--- PROPERTY FEATURES ---
{reviewed_features}

--- MARKET BENCHMARKS ---
Predicted sale price : ${predicted_price:,.0f}
Median sale price    : {median_sale_price}
Mean sale price      : {mean_sale_price}
Typical price range  : {typical_range_low} – {typical_range_high}

--- QUALITY CODE REFERENCE ---
ExterQual / KitchenQual: Ex = Excellent | Gd = Good | TA = Typical/Average | Fa = Fair | Po = Poor

--- OUTPUT FORMAT ---
Produce exactly four labeled sections with no extra headings:

MARKET POSITION
One sentence placing the predicted price relative to the typical range (compute the percentage above or below the midpoint of the range if possible).

VALUE DRIVERS
Bullet list of up to three features that push the price up. For each, state the feature name, its value, and the direction of influence (e.g. "OverallQual 9/10 — significantly above average, strong upward driver").

VALUE CONSTRAINTS
Bullet list of up to two features that limit the price. If none are limiting, write "None identified."

MODEL CONFIDENCE
One sentence on reliability: the model achieved ~$17k MAE and ~$27k RMSE on test data (R² ≈ 0.91), so the estimate carries roughly ±$20k uncertainty at 68% confidence.

Rules:
- Use only data provided above; do not invent values.
- Keep the full response under 200 words.
- Do not reproduce the raw JSON.
""".strip(),

    "v3": """
You are a friendly real-estate advisor writing a short valuation story for a home buyer or seller.

--- PROPERTY DETAILS ---
{reviewed_features}

--- PRICE ESTIMATE ---
Estimated value      : ${predicted_price:,.0f}
Typical market range : {typical_range_low} – {typical_range_high}
Market median        : {median_sale_price}

--- QUALITY CODE GUIDE ---
ExterQual / KitchenQual: Ex = Excellent | Gd = Good | TA = Typical/Average | Fa = Fair | Po = Poor

--- YOUR TASK ---
Write a warm, conversational paragraph (4–6 sentences) that tells the story of why this home is priced where it is.

Cover these points naturally in the narrative:
• Where does the price sit relative to the typical market range — is it a great deal, right in line, or a premium property?
• What two or three things make this home stand out (or hold it back) in terms of value?
• How does the age and condition of the home factor in?
• End with a reassuring note that the model's estimate has a typical accuracy of ±$20,000, so the final sale price may vary based on negotiation and local conditions.

Rules:
- Write in second person ("your home", "you're looking at…") to make it personal.
- Avoid technical jargon; translate quality codes into plain words (e.g. "above-average kitchen finish").
- Do not reproduce the raw JSON or feature field names verbatim.
- Keep the response under 160 words.
""".strip(),
}

STAGE2_EVALUATION_CASES = (
    PromptEvaluationExample(
        case_id="interp_1",
        description=(
            "High-quality 2-story home: OverallQual 9, GrLivArea 2250, YearBuilt 2008, "
            "GarageCars 2, KitchenQual Ex, ExterQual Ex, TotalBsmtSF 1100, FullBath 3, "
            "BedroomAbvGr 4, LotArea 9200, Neighborhood NridgHt, HouseStyle 2Story. "
            "Predicted price $320,000 vs. typical range $130k–$215k."
        ),
        notes="Above-range premium property — tests correct market-position framing and quality driver identification.",
    ),
    PromptEvaluationExample(
        case_id="interp_2",
        description=(
            "Modest starter home: OverallQual 5, GrLivArea 1180, YearBuilt 1954, "
            "GarageCars 1, KitchenQual TA, ExterQual TA, TotalBsmtSF 0, FullBath 1, "
            "BedroomAbvGr 3, LotArea 6000, Neighborhood OldTown, HouseStyle 1Story. "
            "Predicted price $118,000 vs. typical range $130k–$215k."
        ),
        notes="Below-range property — tests constraint identification and below-median framing.",
    ),
    PromptEvaluationExample(
        case_id="interp_3",
        description=(
            "Mid-market property: OverallQual 6, GrLivArea 1500, YearBuilt 2005, "
            "GarageCars 2, KitchenQual Gd, ExterQual TA, TotalBsmtSF 800, FullBath 2, "
            "BedroomAbvGr 3, LotArea 8500, Neighborhood CollgCr, HouseStyle 1Story. "
            "Predicted price $172,000 vs. typical range $130k–$215k."
        ),
        notes="Within-range average property — tests balanced narrative with no extreme drivers.",
    ),
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StageProcessingError(Exception):
    def __init__(self, error_code: str, user_message: str, detail: str):
        super().__init__(detail)
        self.error_code = error_code
        self.user_message = user_message
        self.detail = detail

    def to_fallback(self) -> StageFallback:
        return StageFallback(
            used_fallback=True,
            error_code=self.error_code,
            user_message=self.user_message,
            error=self.detail,
        )


def load_model():
    try:
        loaded_model = joblib.load(MODEL_PATH)
        print("ML model loaded successfully.")
        return loaded_model
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return None


def load_summary_stats() -> SummaryStats:
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        return SummaryStats.model_validate(metrics.get("summary_stats", {}))
    except Exception as exc:
        print(f"Summary stats unavailable: {exc}")
        return SummaryStats()


def load_env_file() -> Dict[str, str]:
    if not ENV_PATH.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


ENV_SETTINGS = load_env_file()
model = load_model()
SUMMARY_STATS = load_summary_stats()


def get_setting(name: str, default: str) -> str:
    return os.getenv(name) or ENV_SETTINGS.get(name) or default


def get_llm_provider() -> str:
    provider = get_setting("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    return provider if provider in SUPPORTED_LLM_PROVIDERS else DEFAULT_LLM_PROVIDER


def get_request_timeout_seconds() -> int:
    raw_timeout = get_setting(
        "LLM_TIMEOUT_SECONDS",
        get_setting("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)),
    )
    try:
        return max(1, int(raw_timeout))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def get_ollama_url() -> str:
    host = get_setting("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    return f"{host}/api/generate"


def get_ollama_model() -> str:
    return get_setting("MODEL_NAME", DEFAULT_OLLAMA_MODEL)


def get_gemini_model() -> str:
    return get_setting("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def get_gemini_api_key() -> str:
    return get_setting("GEMINI_API_KEY", "")


def get_gemini_url() -> str:
    return f"{GEMINI_API_BASE_URL}/{get_gemini_model()}:generateContent"


def resolve_prompt_version(
    requested_version: str,
    prompt_registry: Dict[str, str],
    default_version: str,
) -> str:
    return requested_version if requested_version in prompt_registry else default_version


def build_field_metadata(
    features: HouseFeatures,
    raw_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, FieldCompleteness]:
    raw_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    metadata: Dict[str, FieldCompleteness] = {}

    for field_name in FEATURE_FIELDS:
        field_details = raw_metadata.get(field_name, {})
        value_present = getattr(features, field_name) is not None

        try:
            metadata[field_name] = FieldCompleteness.model_validate(
                {
                    "value_present": value_present,
                    "confidence": field_details.get("confidence"),
                    "source": field_details.get("source"),
                }
            )
        except Exception:
            metadata[field_name] = FieldCompleteness(value_present=value_present)

    return metadata


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"${value:,.0f}"


def truncate_detail(detail: Any, limit: int = 300) -> str:
    text = str(detail).strip()
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def extract_http_error_detail(response: Optional[requests.Response], fallback: Any) -> str:
    if response is None:
        return truncate_detail(fallback)

    response_text = response.text
    try:
        payload = response.json()
    except ValueError:
        return truncate_detail(response_text or fallback)

    if not isinstance(payload, dict):
        return truncate_detail(response_text or fallback)

    provider_error = payload.get("error")
    if isinstance(provider_error, dict):
        message = provider_error.get("message")
        status = provider_error.get("status")
        code = provider_error.get("code", response.status_code)

        if isinstance(message, str) and message.strip():
            parts = [str(code)] if code else []
            if isinstance(status, str) and status.strip():
                parts.append(status.strip())
            prefix = " ".join(parts)
            return truncate_detail(f"{prefix}: {message.strip()}" if prefix else message.strip())

    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return truncate_detail(message.strip())

    return truncate_detail(response_text or fallback)


def build_stage_fallback(
    error_code: str,
    user_message: str,
    detail: Any,
) -> StageFallback:
    return StageFallback(
        used_fallback=True,
        error_code=error_code,
        user_message=user_message,
        error=truncate_detail(detail),
    )


def build_extraction_fallback(error_code: str, detail: Any) -> StageFallback:
    messages = {
        "llm_timeout": "Extraction timed out. Review and complete the feature form before predicting.",
        "llm_unavailable": "The extraction service is unavailable. You can still complete the feature form manually.",
        "llm_http_error": "The extraction service returned an error. Review the extracted fields before predicting.",
        "llm_malformed_response": "The extraction service returned an unreadable response. Review the extracted fields before predicting.",
        "llm_malformed_json": "The extraction service returned malformed JSON. Review and complete the feature form manually.",
        "extraction_validation_error": "The extracted fields did not pass validation. Review and complete the feature form before predicting.",
        "extraction_stage_error": "Extraction could not be completed cleanly. Review the feature form before predicting.",
    }
    return build_stage_fallback(
        error_code,
        messages.get(error_code, messages["extraction_stage_error"]),
        detail,
    )


def build_interpretation_fallback_metadata(error_code: str, detail: Any) -> StageFallback:
    messages = {
        "llm_timeout": "The explanation service timed out, so a rules-based explanation is shown instead.",
        "llm_unavailable": "The explanation service is unavailable, so a rules-based explanation is shown instead.",
        "llm_http_error": "The explanation service returned an error, so a rules-based explanation is shown instead.",
        "llm_malformed_response": "The explanation service returned an unreadable response, so a rules-based explanation is shown instead.",
        "interpretation_stage_error": "A rules-based explanation is shown because the interpretation stage failed.",
    }
    return build_stage_fallback(
        error_code,
        messages.get(error_code, messages["interpretation_stage_error"]),
        detail,
    )


def extraction_requires_review(extraction_stage: ExtractionStageResponse) -> bool:
    return (
        extraction_stage.fallback.used_fallback
        or not extraction_stage.extraction.is_complete
    )


def build_review_required_detail(
    extraction_stage: ExtractionStageResponse,
) -> Dict[str, Any]:
    fallback = extraction_stage.fallback
    missing_fields = extraction_stage.extraction.missing_fields

    if fallback.used_fallback:
        message = fallback.user_message or (
            "Extraction could not be completed cleanly. Review and complete the feature form before predicting."
        )
        error_code = fallback.error_code or "review_required"
        error = fallback.error
    else:
        message = (
            "Extraction is incomplete. Review and complete the feature form before predicting."
        )
        error_code = "review_required"
        error = None

    return {
        "message": message,
        "error_code": error_code,
        "missing_fields": missing_fields,
        "fallback": fallback.model_dump(),
        "extracted_data": extraction_stage.extraction.model_dump(by_alias=True),
        "prompt_version": extraction_stage.prompt_version,
        "error": error,
    }


def validate_extraction_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise StageProcessingError(
            "llm_malformed_json",
            "The extraction service returned malformed JSON. Review and complete the feature form manually.",
            "Extraction payload was not a JSON object.",
        )

    features_payload = payload.get("features")
    if not isinstance(features_payload, dict):
        raise StageProcessingError(
            "llm_malformed_json",
            "The extraction service returned malformed JSON. Review and complete the feature form manually.",
            "Extraction payload is missing a valid 'features' object.",
        )

    return payload


def validate_reported_missing_fields(payload: Dict[str, Any], features: HouseFeatures) -> None:
    reported_missing = payload.get("missing_fields")
    if reported_missing is None:
        return

    if (
        not isinstance(reported_missing, list)
        or any(field_name not in FEATURE_FIELDS for field_name in reported_missing)
    ):
        raise StageProcessingError(
            "extraction_validation_error",
            "The extracted fields did not pass validation. Review and complete the feature form before predicting.",
            "missing_fields must be a list of supported feature names.",
        )

    expected_missing = sorted(features.missing_fields())
    actual_missing = sorted(reported_missing)
    if actual_missing != expected_missing:
        raise StageProcessingError(
            "extraction_validation_error",
            "The extracted fields did not pass validation. Review and complete the feature form before predicting.",
            f"missing_fields mismatch. expected={expected_missing}, actual={actual_missing}",
        )


def build_extraction_prompt(text: str, prompt_version: str) -> str:
    version = resolve_prompt_version(
        prompt_version,
        EXTRACTION_PROMPTS,
        DEFAULT_EXTRACTION_PROMPT_VERSION,
    )
    return EXTRACTION_PROMPTS[version].replace("{user_text}", text)


def build_extraction_prompt_catalog() -> ExtractionPromptCatalogResponse:
    prompt_versions = sorted(EXTRACTION_PROMPTS.keys())
    return ExtractionPromptCatalogResponse(
        default_version=DEFAULT_EXTRACTION_PROMPT_VERSION,
        available_versions=prompt_versions,
        prompts=[
            PromptVariant(
                version=version,
                description=description,
                template=EXTRACTION_PROMPTS[version],
            )
            for version, description in (
                ("v1", "Baseline schema-first extractor."),
                ("v2", "Evidence-first extractor with stricter grounding instructions."),
                ("v3", "Field-guided extractor optimized for conservative null handling."),
            )
        ],
        sample_inputs=list(STAGE1_EVALUATION_CASES),
    )


def build_interpretation_prompt_catalog() -> ExtractionPromptCatalogResponse:
    prompt_versions = sorted(INTERPRETATION_PROMPTS.keys())
    return ExtractionPromptCatalogResponse(
        default_version=DEFAULT_INTERPRETATION_PROMPT_VERSION,
        available_versions=prompt_versions,
        prompts=[
            PromptVariant(
                version=version,
                description=description,
                template=INTERPRETATION_PROMPTS[version],
            )
            for version, description in (
                (
                    "v1",
                    "Structured analyst prompt: market position, top drivers, constraints, "
                    "and a confidence note. Audience-neutral, under 150 words.",
                ),
                (
                    "v2",
                    "Four-section quantitative brief: MARKET POSITION, VALUE DRIVERS, "
                    "VALUE CONSTRAINTS, MODEL CONFIDENCE. Best for technical reviewers.",
                ),
                (
                    "v3",
                    "Conversational client narrative written in second person. "
                    "Translates quality codes into plain language, under 160 words.",
                ),
            )
        ],
        sample_inputs=list(STAGE2_EVALUATION_CASES),
    )


def build_interpretation_prompt(
    reviewed_features: ReviewedHouseFeatures,
    predicted_price: float,
    summary_stats: SummaryStats,
    prompt_version: str,
) -> str:
    version = resolve_prompt_version(
        prompt_version,
        INTERPRETATION_PROMPTS,
        DEFAULT_INTERPRETATION_PROMPT_VERSION,
    )
    return INTERPRETATION_PROMPTS[version].format(
        reviewed_features=json.dumps(reviewed_features.model_dump(), indent=2),
        predicted_price=predicted_price,
        median_sale_price=format_currency(summary_stats.median_sale_price),
        mean_sale_price=format_currency(summary_stats.mean_sale_price),
        typical_range_low=format_currency(summary_stats.typical_sale_price_range.low),
        typical_range_high=format_currency(summary_stats.typical_sale_price_range.high),
    )


def log_llm_prompt(prompt: str, *, json_mode: bool = False) -> None:
    prompt_type = "json" if json_mode else "text"
    provider = get_llm_provider()
    print(f"[{provider}] sending {prompt_type} prompt:")
    print(f"----- {provider.upper()} PROMPT START -----")
    print(prompt)
    print(f"----- {provider.upper()} PROMPT END -----")


def parse_llm_json_result(raw_result: Any) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except json.JSONDecodeError as exc:
        raise StageProcessingError(
            "llm_malformed_json",
            "The language model returned malformed JSON.",
            truncate_detail(exc),
        ) from exc
    if not isinstance(parsed, dict):
        raise StageProcessingError(
            "llm_malformed_json",
            "The language model returned malformed JSON.",
            "LLM JSON response must be an object.",
        )
    return parsed


def parse_llm_text_result(raw_result: Any) -> str:
    if not isinstance(raw_result, str):
        raise StageProcessingError(
            "llm_malformed_response",
            "The language model returned an invalid text response.",
            "LLM text response must be a string.",
        )

    text_result = raw_result.strip()
    if not text_result:
        raise StageProcessingError(
            "llm_malformed_response",
            "The language model returned an empty response.",
            "LLM returned an empty response.",
        )
    return text_result


def call_ollama(prompt: str, *, json_mode: bool = False) -> Any:
    payload = {
        "model": get_ollama_model(),
        "prompt": prompt,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        get_ollama_url(),
        json=payload,
        timeout=get_request_timeout_seconds(),
    )
    response.raise_for_status()

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise StageProcessingError(
            "llm_malformed_response",
            "The language model response envelope was not valid JSON.",
            truncate_detail(exc),
        ) from exc
    if not isinstance(response_payload, dict):
        raise StageProcessingError(
            "llm_malformed_response",
            "The language model response envelope was not valid JSON.",
            "LLM response envelope must be a JSON object.",
        )

    raw_result = response_payload.get("response", "")
    return parse_llm_json_result(raw_result) if json_mode else parse_llm_text_result(raw_result)


def extract_gemini_text(response_payload: Dict[str, Any]) -> str:
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise StageProcessingError(
            "llm_malformed_response",
            "The language model response envelope was not valid JSON.",
            "Gemini response did not include candidates.",
        )

    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise StageProcessingError(
            "llm_malformed_response",
            "The language model response envelope was not valid JSON.",
            "Gemini candidate must be a JSON object.",
        )

    content = first_candidate.get("content", {})
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        raise StageProcessingError(
            "llm_malformed_response",
            "The language model response envelope was not valid JSON.",
            "Gemini candidate did not include text parts.",
        )

    text_parts = [
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return parse_llm_text_result("".join(text_parts))


def call_gemini(prompt: str, *, json_mode: bool = False) -> Any:
    api_key = get_gemini_api_key().strip()
    if not api_key:
        raise StageProcessingError(
            "llm_unavailable",
            "The Gemini API key is missing.",
            "Set GEMINI_API_KEY before using the Gemini provider.",
        )

    payload: Dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                ]
            }
        ]
    }
    if json_mode:
        payload["generationConfig"] = {"responseMimeType": "application/json"}

    response = requests.post(
        get_gemini_url(),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        json=payload,
        timeout=get_request_timeout_seconds(),
    )
    response.raise_for_status()

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise StageProcessingError(
            "llm_malformed_response",
            "The language model response envelope was not valid JSON.",
            truncate_detail(exc),
        ) from exc
    if not isinstance(response_payload, dict):
        raise StageProcessingError(
            "llm_malformed_response",
            "The language model response envelope was not valid JSON.",
            "LLM response envelope must be a JSON object.",
        )

    raw_result = extract_gemini_text(response_payload)
    return parse_llm_json_result(raw_result) if json_mode else raw_result


def call_llm(prompt: str, *, json_mode: bool = False) -> Any:
    log_llm_prompt(prompt, json_mode=json_mode)

    try:
        if get_llm_provider() == "gemini":
            return call_gemini(prompt, json_mode=json_mode)
        return call_ollama(prompt, json_mode=json_mode)
    except requests.exceptions.Timeout as exc:
        raise StageProcessingError(
            "llm_timeout",
            "The language model timed out.",
            truncate_detail(exc),
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise StageProcessingError(
            "llm_unavailable",
            "The language model service is unavailable.",
            truncate_detail(exc),
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise StageProcessingError(
            "llm_http_error",
            "The language model service returned an error.",
            extract_http_error_detail(exc.response, exc),
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise StageProcessingError(
            "llm_unavailable",
            "The language model request failed.",
            truncate_detail(exc),
        ) from exc


def run_extraction_stage(
    description: str,
    prompt_version: str = DEFAULT_EXTRACTION_PROMPT_VERSION,
) -> ExtractionStageResponse:
    resolved_prompt_version = resolve_prompt_version(
        prompt_version,
        EXTRACTION_PROMPTS,
        DEFAULT_EXTRACTION_PROMPT_VERSION,
    )

    try:
        payload = call_llm(
            build_extraction_prompt(description, resolved_prompt_version),
            json_mode=True,
        )
        payload = validate_extraction_payload(payload)
        try:
            features = HouseFeatures.model_validate(payload["features"])
        except ValidationError as exc:
            raise StageProcessingError(
                "extraction_validation_error",
                "The extracted fields did not pass validation. Review and complete the feature form before predicting.",
                truncate_detail(exc),
            ) from exc
        validate_reported_missing_fields(payload, features)
        field_metadata = build_field_metadata(features, payload.get("field_metadata"))
        extraction = ExtractionResult.from_features(features, field_metadata)
        return ExtractionStageResponse(
            extraction=extraction,
            fallback=StageFallback(),
            prompt_version=resolved_prompt_version,
        )
    except StageProcessingError as exc:
        print(f"Extraction failed: {exc.detail}")
        return ExtractionStageResponse(
            extraction=ExtractionResult.from_features(HouseFeatures()),
            fallback=build_extraction_fallback(exc.error_code, exc.detail),
            prompt_version=resolved_prompt_version,
        )
    except Exception as exc:
        print(f"Extraction failed: {exc}")
        return ExtractionStageResponse(
            extraction=ExtractionResult.from_features(HouseFeatures()),
            fallback=build_extraction_fallback("extraction_stage_error", exc),
            prompt_version=resolved_prompt_version,
        )


def run_prediction_stage(reviewed_features: ReviewedHouseFeatures) -> PredictionStageResponse:
    try:
        if model is None:
            raise StageProcessingError(
                "model_unavailable",
                "The pricing model is unavailable, so a fallback estimate was used.",
                "ML model is not loaded.",
            )

        input_df = pd.DataFrame([reviewed_features.model_dump()])
        predicted_price = float(model.predict(input_df)[0])
        if pd.isna(predicted_price):
            raise StageProcessingError(
                "model_prediction_failure",
                "The pricing model returned an invalid value, so a fallback estimate was used.",
                "Model returned NaN.",
            )
        fallback = StageFallback()
    except StageProcessingError as exc:
        predicted_price = SUMMARY_STATS.median_sale_price or DEFAULT_PREDICTION_FALLBACK
        fallback = exc.to_fallback()
    except Exception as exc:
        predicted_price = SUMMARY_STATS.median_sale_price or DEFAULT_PREDICTION_FALLBACK
        fallback = build_stage_fallback(
            "model_prediction_failure",
            "The pricing model failed, so a fallback estimate was used.",
            exc,
        )

    return PredictionStageResponse(
        prediction=predicted_price,
        reviewed_features=reviewed_features,
        summary_stats=SUMMARY_STATS,
        fallback=fallback,
    )


def build_interpretation_fallback(
    reviewed_features: ReviewedHouseFeatures,
    predicted_price: float,
    summary_stats: SummaryStats,
) -> str:
    range_low = summary_stats.typical_sale_price_range.low
    range_high = summary_stats.typical_sale_price_range.high

    if range_low is not None and range_high is not None:
        range_text = f" Typical prices in the training data were around {format_currency(range_low)} to {format_currency(range_high)}."
    else:
        range_text = ""

    return (
        f"The estimate of {format_currency(predicted_price)} is based on the reviewed feature mix, "
        f"including quality {reviewed_features.OverallQual}/10, {reviewed_features.GrLivArea} sqft of living area, "
        f"and {reviewed_features.BedroomAbvGr} bedrooms.{range_text}"
    )


def run_interpretation_stage(
    reviewed_features: ReviewedHouseFeatures,
    predicted_price: float,
    summary_stats: Optional[SummaryStats] = None,
    prompt_version: str = DEFAULT_INTERPRETATION_PROMPT_VERSION,
) -> InterpretationStageResponse:
    effective_summary_stats = summary_stats or SUMMARY_STATS
    resolved_prompt_version = resolve_prompt_version(
        prompt_version,
        INTERPRETATION_PROMPTS,
        DEFAULT_INTERPRETATION_PROMPT_VERSION,
    )

    try:
        interpretation = call_llm(
            build_interpretation_prompt(
                reviewed_features,
                predicted_price,
                effective_summary_stats,
                resolved_prompt_version,
            )
        )
        fallback = StageFallback()
    except StageProcessingError as exc:
        print(f"Interpretation failed: {exc.detail}")
        interpretation = build_interpretation_fallback(
            reviewed_features,
            predicted_price,
            effective_summary_stats,
        )
        fallback = build_interpretation_fallback_metadata(exc.error_code, exc.detail)
    except Exception as exc:
        print(f"Interpretation failed: {exc}")
        interpretation = build_interpretation_fallback(
            reviewed_features,
            predicted_price,
            effective_summary_stats,
        )
        fallback = build_interpretation_fallback_metadata("interpretation_stage_error", exc)

    return InterpretationStageResponse(
        interpretation=interpretation,
        summary_stats=effective_summary_stats,
        fallback=fallback,
        prompt_version=resolved_prompt_version,
    )


def build_prediction_fallback_note(
    extraction: ExtractionResult,
    existing_fallback: StageFallback,
) -> StageFallback:
    if not extraction.missing_fields:
        return existing_fallback

    missing_fields = ", ".join(extraction.missing_fields)
    note = (
        "Prediction used default values for missing extracted fields because "
        f"reviewed features were not provided: {missing_fields}."
    )

    if existing_fallback.used_fallback and existing_fallback.error:
        error = f"{existing_fallback.error} {note}"
    else:
        error = note

    user_message = existing_fallback.user_message or (
        "Prediction ran with suggested defaults for fields that were still missing after extraction."
    )
    error_code = existing_fallback.error_code or "review_required_defaults"
    return StageFallback(
        used_fallback=True,
        error_code=error_code,
        user_message=user_message,
        error=error,
    )


@app.post("/extract", response_model=ExtractionStageResponse)
async def extract_stage(request: ExtractionRequest) -> ExtractionStageResponse:
    return run_extraction_stage(request.description, request.prompt_version)


@app.get("/extract/prompts", response_model=ExtractionPromptCatalogResponse)
async def get_extraction_prompts() -> ExtractionPromptCatalogResponse:
    return build_extraction_prompt_catalog()


@app.get("/interpret/prompts", response_model=ExtractionPromptCatalogResponse)
async def get_interpretation_prompts() -> ExtractionPromptCatalogResponse:
    return build_interpretation_prompt_catalog()


@app.post("/predict", response_model=PredictionStageResponse)
async def predict_stage(request: PredictionRequest) -> PredictionStageResponse:
    return run_prediction_stage(request.reviewed_features)


@app.post("/interpret", response_model=InterpretationStageResponse)
async def interpret_stage(request: InterpretationRequest) -> InterpretationStageResponse:
    return run_interpretation_stage(
        reviewed_features=request.reviewed_features,
        predicted_price=request.prediction,
        summary_stats=request.summary_stats,
        prompt_version=request.prompt_version,
    )


@app.post("/agent", response_model=AgentResponse)
async def ai_agent(request: AgentRequest) -> AgentResponse:
    extraction_stage = run_extraction_stage(
        request.description,
        request.extraction_prompt_version,
    )
    if request.reviewed_features is None and extraction_requires_review(extraction_stage):
        raise HTTPException(
            status_code=409,
            detail=build_review_required_detail(extraction_stage),
        )

    reviewed_features = (
        request.reviewed_features or extraction_stage.extraction.features.with_defaults()
    )
    prediction_stage = run_prediction_stage(reviewed_features)
    interpretation_stage = run_interpretation_stage(
        reviewed_features=reviewed_features,
        predicted_price=prediction_stage.prediction,
        summary_stats=prediction_stage.summary_stats,
        prompt_version=request.interpretation_prompt_version,
    )

    prediction_fallback = prediction_stage.fallback
    if request.reviewed_features is None:
        prediction_fallback = build_prediction_fallback_note(
            extraction_stage.extraction,
            prediction_fallback,
        )

    return AgentResponse(
        prediction=prediction_stage.prediction,
        interpretation=interpretation_stage.interpretation,
        reviewed_features=reviewed_features,
        summary_stats=prediction_stage.summary_stats,
        fallback=FallbackMetadata(
            extraction=extraction_stage.fallback,
            prediction=prediction_fallback,
            interpretation=interpretation_stage.fallback,
        ),
        prompt_versions=PromptVersions(
            extraction=extraction_stage.prompt_version,
            interpretation=interpretation_stage.prompt_version,
        ),
        extraction=extraction_stage.extraction,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)