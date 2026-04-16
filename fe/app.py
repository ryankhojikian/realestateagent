import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="HomeVal AI — Instant Property Valuations",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Configuration ──────────────────────────────────────────────────────────────

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")
REQUEST_TIMEOUT_SECONDS = 20

FIELD_LABELS: Dict[str, str] = {
    "OverallQual": "Overall Quality",
    "GrLivArea": "Living Area (sqft)",
    "TotalBsmtSF": "Basement Area (sqft)",
    "FullBath": "Full Bathrooms",
    "YearBuilt": "Year Built",
    "GarageCars": "Garage Capacity (cars)",
    "KitchenQual": "Kitchen Quality",
    "HouseStyle": "House Style",
    "LotArea": "Lot Size (sqft)",
    "Neighborhood": "Neighborhood",
    "ExterQual": "Exterior Quality",
    "BedroomAbvGr": "Bedrooms",
}

FIELD_HELP: Dict[str, str] = {
    "OverallQual": "Rate the overall material and finish quality: 1 = Very Poor, 10 = Excellent",
    "GrLivArea": "Total above-ground living area in square feet",
    "TotalBsmtSF": "Total basement area in square feet (0 if no basement)",
    "FullBath": "Number of full bathrooms above basement level",
    "YearBuilt": "Original construction year of the property",
    "GarageCars": "Garage car capacity (0 = no garage)",
    "KitchenQual": "Ex = Excellent · Gd = Good · TA = Average · Fa = Fair · Po = Poor",
    "HouseStyle": "Common values: 1Story, 2Story, 1.5Fin, SLvl, SFoyer",
    "LotArea": "Total lot size in square feet",
    "Neighborhood": "Location within Ames city limits (e.g. CollgCr, NAmes, OldTown, NridgHt)",
    "ExterQual": "Exterior material quality: Ex = Excellent · Gd = Good · TA = Average · Fa = Fair",
    "BedroomAbvGr": "Number of bedrooms above basement level",
}

FIELD_GROUPS: List[Tuple[str, List[str]]] = [
    ("Location", ["Neighborhood", "HouseStyle", "LotArea"]),
    ("Size & Layout", ["GrLivArea", "BedroomAbvGr", "FullBath", "TotalBsmtSF", "GarageCars"]),
    ("Quality", ["OverallQual", "ExterQual", "KitchenQual"]),
    ("History", ["YearBuilt"]),
]

FEATURE_DEFAULTS: Dict[str, Any] = {
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

NUMERIC_FIELDS = {
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "GarageCars",
    "LotArea",
    "BedroomAbvGr",
}

NUMERIC_CONSTRAINTS: Dict[str, Dict[str, int]] = {
    "OverallQual": {"min_value": 1, "max_value": 10},
    "GrLivArea": {"min_value": 0},
    "TotalBsmtSF": {"min_value": 0},
    "FullBath": {"min_value": 0},
    "YearBuilt": {"min_value": 1800, "max_value": 2100},
    "GarageCars": {"min_value": 0},
    "LotArea": {"min_value": 0},
    "BedroomAbvGr": {"min_value": 0},
}

EXAMPLE_DESCRIPTIONS: List[Tuple[str, str]] = [
    (
        "NridgHt — 4 bed, 3 bath, 2,250 sqft",
        "Two-story home in NridgHt with 2,250 sqft, 4 bedrooms, 3 full baths, "
        "2-car garage, excellent kitchen, and 1,100 sqft finished basement. Built in 2003.",
    ),
    (
        "NAmes — Ranch, 3 bed, 1,200 sqft",
        "Cozy 1-story ranch in NAmes. 1,200 sqft living area, 3 bedrooms, 2 full baths, "
        "attached 1-car garage, average kitchen quality, no basement. Built 1978.",
    ),
    (
        "CollgCr — Modern, 3 bed, 1,850 sqft",
        "Modern 2-story in CollgCr. 1,850 sqft, 3 beds, 2 full baths, "
        "2-car garage, good kitchen, 900 sqft basement. Built 2008.",
    ),
    (
        "OldTown — Fixer-upper, 2 bed, 980 sqft",
        "Older 1-story bungalow in OldTown. 980 sqft living area, 2 bedrooms, "
        "1 full bath, no garage, fair kitchen, small 400 sqft basement. Built 1940.",
    ),
]

# ── Custom exceptions ──────────────────────────────────────────────────────────


class BackendUnavailableError(Exception):
    pass


class ApiValidationError(Exception):
    pass


class ApiRequestError(Exception):
    pass


# ── Session state ──────────────────────────────────────────────────────────────


def init_state() -> None:
    st.session_state.setdefault("description", "")
    st.session_state.setdefault("extraction_response", None)
    st.session_state.setdefault("reviewed_features", None)
    st.session_state.setdefault("prediction_response", None)
    st.session_state.setdefault("interpretation_response", None)


def reset_pipeline_state() -> None:
    st.session_state.extraction_response = None
    st.session_state.reviewed_features = None
    st.session_state.prediction_response = None
    st.session_state.interpretation_response = None


def reset_prediction_state() -> None:
    st.session_state.prediction_response = None
    st.session_state.interpretation_response = None


# ── API helpers ────────────────────────────────────────────────────────────────


def parse_validation_error(payload: Dict[str, Any]) -> str:
    details = payload.get("detail", [])
    if isinstance(details, list) and details:
        formatted = []
        for item in details:
            location = " -> ".join(str(part) for part in item.get("loc", []))
            formatted.append(f"{location}: {item.get('msg', 'Invalid value')}")
        return "Validation failed for the submitted fields:\n- " + "\n- ".join(formatted)
    return "Validation failed for the submitted fields."


def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{BACKEND_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendUnavailableError(
            "Backend unavailable. Start the FastAPI service and try again."
        ) from exc

    if response.status_code == 422:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {}
        raise ApiValidationError(parse_validation_error(error_payload))

    if not response.ok:
        try:
            error_payload = response.json()
            detail = error_payload.get("detail") or error_payload
        except ValueError:
            detail = response.text
        raise ApiRequestError(f"Backend request failed: {detail}")

    try:
        return response.json()
    except ValueError as exc:
        raise ApiRequestError("Backend returned invalid JSON.") from exc


def merge_with_defaults(features: Dict[str, Any]) -> Dict[str, Any]:
    merged = {}
    for field_name, default_value in FEATURE_DEFAULTS.items():
        value = features.get(field_name)
        merged[field_name] = default_value if value is None else value
    return merged


def extract_llm_error_detail(fallback: Dict[str, Any]) -> str | None:
    """Return a human-readable LLM error message when the fallback originated from the LLM layer."""
    error_code = fallback.get("error_code", "")
    if not isinstance(error_code, str) or not error_code.startswith("llm_"):
        return None
    raw_error = fallback.get("error")
    if not raw_error:
        return None
    try:
        parsed = json.loads(raw_error)
        nested = parsed.get("error", {})
        llm_message = nested.get("message") or parsed.get("message")
        llm_status = nested.get("status") or parsed.get("status")
        llm_code = nested.get("code") or parsed.get("code")
        parts = []
        if llm_message:
            parts.append(llm_message)
        if llm_status:
            parts.append(f"Status: {llm_status}")
        if llm_code:
            parts.append(f"Code: {llm_code}")
        return " | ".join(parts) if parts else str(raw_error)
    except (ValueError, AttributeError):
        return str(raw_error)


def format_backend_fallback(fallback: Dict[str, Any], default_message: str) -> str:
    message = fallback.get("user_message") or default_message
    error_code = fallback.get("error_code")
    if error_code:
        return f"{message} (`{error_code}`)"
    return message


def format_currency(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.0f}"


# ── UI components ──────────────────────────────────────────────────────────────


def confidence_badge_html(confidence: float | None) -> str:
    if confidence is None:
        return ""
    if confidence >= 0.8:
        return f'<span class="conf-high">{confidence:.0%} confidence</span>'
    if confidence >= 0.5:
        return f'<span class="conf-medium">{confidence:.0%} confidence</span>'
    return f'<span class="conf-low">{confidence:.0%} confidence</span>'


def render_step_tracker(current_step: int) -> None:
    steps = [
        ("1", "Describe", "Enter a property description"),
        ("2", "Review", "Verify extracted features"),
        ("3", "Valuation", "See the predicted price"),
    ]
    parts = ['<div class="steps-row">']
    for idx, (num, title, desc) in enumerate(steps, start=1):
        if idx < current_step:
            css = "step-item done"
            badge = "&#10003;"
        elif idx == current_step:
            css = "step-item active"
            badge = num
        else:
            css = "step-item"
            badge = num
        parts.append(
            f'<div class="{css}">'
            f'<span class="step-num">{badge}</span>'
            f'<span><strong style="display:block;font-size:0.88rem">{title}</strong>'
            f'<span style="font-size:0.75rem;opacity:0.65">{desc}</span></span>'
            f"</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_feature_inputs_grouped(
    reviewed_features: Dict[str, Any],
    missing_fields: List[str],
    field_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    edited_features: Dict[str, Any] = {}

    for group_name, group_fields in FIELD_GROUPS:
        st.markdown(
            f'<div class="group-header">{group_name}</div>',
            unsafe_allow_html=True,
        )
        n_cols = min(2, len(group_fields))
        cols = st.columns(n_cols)
        for idx, field_name in enumerate(group_fields):
            col = cols[idx % n_cols]
            label = FIELD_LABELS.get(field_name, field_name)
            help_text = FIELD_HELP.get(field_name)
            is_missing = field_name in missing_fields
            metadata = field_metadata.get(field_name, {})
            confidence = metadata.get("confidence")
            source = metadata.get("source")
            display_label = f"{label} (missing)" if is_missing else label

            with col:
                if field_name in NUMERIC_FIELDS:
                    constraints = NUMERIC_CONSTRAINTS.get(field_name, {})
                    edited_features[field_name] = int(
                        st.number_input(
                            display_label,
                            value=int(
                                reviewed_features.get(field_name, FEATURE_DEFAULTS[field_name])
                            ),
                            step=1,
                            help=help_text,
                            **constraints,
                        )
                    )
                else:
                    edited_features[field_name] = st.text_input(
                        display_label,
                        value=str(
                            reviewed_features.get(field_name, FEATURE_DEFAULTS[field_name])
                        ),
                        help=help_text,
                    ).strip()

                if is_missing:
                    st.caption("Not found in the description — please verify before predicting.")
                elif source and confidence is not None:
                    st.markdown(
                        f'{confidence_badge_html(confidence)}'
                        f'<span class="source-label"> from {source}</span>',
                        unsafe_allow_html=True,
                    )
                elif source:
                    st.caption(f"Extracted from: {source}")

    return edited_features


def render_market_context(prediction: float, summary_stats: Dict[str, Any]) -> None:
    median = summary_stats.get("median_sale_price")
    range_info = summary_stats.get("typical_sale_price_range", {})
    low = range_info.get("low")
    high = range_info.get("high")

    col1, col2, col3 = st.columns(3)
    with col1:
        if median:
            delta_pct = (prediction - median) / median * 100
            delta_label = f"{delta_pct:+.1f}% vs. market median"
            st.metric("Predicted Price", format_currency(prediction), delta=delta_label)
        else:
            st.metric("Predicted Price", format_currency(prediction))
    with col2:
        st.metric("Market Median", format_currency(median))
    with col3:
        range_str = (
            f"{format_currency(low)} – {format_currency(high)}" if low and high else "N/A"
        )
        st.metric("Typical Range", range_str)


# ── Styling ────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Global ──────────────────────────────────────────────────── */
.stApp { background: #f0f4f8; }
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #0f172a;
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.65;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    width: 100%;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #273549 !important;
    border-color: #475569 !important;
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #1e293b;
}

/* ── App header ──────────────────────────────────────────────── */
.app-header {
    background: #1e3a5f;
    color: white;
    padding: 2rem 2.5rem 1.75rem;
    border-radius: 14px;
    margin-bottom: 1.25rem;
}
.app-header h1 {
    margin: 0 0 0.3rem 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.app-header p {
    margin: 0;
    font-size: 0.97rem;
    opacity: 0.72;
    line-height: 1.5;
}

/* ── Step tracker ────────────────────────────────────────────── */
.steps-row {
    display: flex;
    margin-bottom: 1.5rem;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
.step-item {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 13px 16px;
    background: white;
    border-top: 3px solid transparent;
    color: #94a3b8;
}
.step-item + .step-item { border-left: 1px solid #f1f5f9; }
.step-item.active {
    border-top-color: #2563eb;
    color: #1e3a5f;
}
.step-item.done {
    border-top-color: #16a34a;
    color: #15803d;
}
.step-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #e2e8f0;
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-item.active .step-num { background: #2563eb; color: white; }
.step-item.done .step-num { background: #16a34a; color: white; }

/* ── Field group header ──────────────────────────────────────── */
.group-header {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #94a3b8;
    margin: 1.25rem 0 0.4rem 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #e2e8f0;
}

/* ── Confidence badges ───────────────────────────────────────── */
.conf-high {
    display: inline-block;
    background: #dcfce7; color: #15803d;
    font-size: 0.69rem; font-weight: 700;
    padding: 2px 8px; border-radius: 99px;
    text-transform: uppercase; letter-spacing: 0.03em;
}
.conf-medium {
    display: inline-block;
    background: #fef9c3; color: #92400e;
    font-size: 0.69rem; font-weight: 700;
    padding: 2px 8px; border-radius: 99px;
    text-transform: uppercase; letter-spacing: 0.03em;
}
.conf-low {
    display: inline-block;
    background: #fee2e2; color: #991b1b;
    font-size: 0.69rem; font-weight: 700;
    padding: 2px 8px; border-radius: 99px;
    text-transform: uppercase; letter-spacing: 0.03em;
}
.source-label {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-left: 4px;
}

/* ── Result price block ──────────────────────────────────────── */
.price-block {
    background: #1e3a5f;
    color: white;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.price-block .price-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.6;
    margin-bottom: 0.5rem;
}
.price-block .price-amount {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}

/* ── Interpretation prose ────────────────────────────────────── */
.interp-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-size: 0.93rem;
    line-height: 1.7;
    color: #334155;
}

/* ── General polish ──────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Bootstrap ──────────────────────────────────────────────────────────────────

init_state()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## HomeVal AI")
    st.markdown(
        "An AI-powered property valuation tool. Describe a home in plain English "
        "and get an instant price estimate backed by machine learning."
    )

    st.divider()
    st.markdown("### Try an example")
    st.markdown(
        "<p style='font-size:0.78rem;color:#64748b;margin-top:-8px'>"
        "Click a property to load it into the description field."
        "</p>",
        unsafe_allow_html=True,
    )
    for i, (short_label, full_text) in enumerate(EXAMPLE_DESCRIPTIONS):
        if st.button(short_label, key=f"example_{i}"):
            st.session_state.description = full_text
            reset_pipeline_state()
            st.rerun()

    st.divider()
    st.markdown("### Tips for better results")
    st.markdown(
        """
- Mention the **neighborhood** by name (e.g. *NridgHt*, *CollgCr*)
- Include **square footage** for living area and basement
- State the **number of bedrooms and bathrooms**
- Note the **garage size** (cars) or if there is no garage
- Include the **year built**
- Describe kitchen quality (*excellent*, *good*, *average*)
"""
    )

    st.divider()
    st.markdown("### How it works")
    st.markdown(
        """
1. Your description is parsed by an LLM to extract property features
2. You review and correct any extracted values
3. A trained regression model predicts the sale price
4. An AI explanation tells you what drove the estimate
"""
    )

# ── Main content ───────────────────────────────────────────────────────────────

has_extraction = st.session_state.extraction_response is not None
has_prediction = st.session_state.prediction_response is not None

current_step = 1
if has_extraction:
    current_step = 2
if has_prediction:
    current_step = 3

st.markdown(
    """
<div class="app-header">
    <h1>HomeVal AI</h1>
    <p>
        Describe any residential property in plain English. Our AI extracts the key features,
        you verify them, and we return an instant market valuation.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

render_step_tracker(current_step)

# ── Step 1: Property description ───────────────────────────────────────────────

st.subheader("Property description")

char_count = len(st.session_state.description)
st.session_state.description = st.text_area(
    "Describe the property",
    value=st.session_state.description,
    placeholder=(
        "Example: Two-story home in NridgHt with 2,250 sqft, 4 bedrooms, 3 full baths, "
        "2-car garage, excellent kitchen, and 1,100 sqft finished basement. Built in 2003."
    ),
    height=160,
    label_visibility="collapsed",
)

hint_col, btn_col = st.columns([3, 1])
with hint_col:
    if char_count == 0:
        st.caption("Try one of the sidebar examples to get started quickly.")
    else:
        st.caption(f"{char_count} characters — the more detail, the more accurate the extraction.")
with btn_col:
    run_extraction = st.button("Extract features", use_container_width=True, type="primary")

if run_extraction:
    reset_prediction_state()
    if not st.session_state.description.strip():
        st.error("Enter a property description before running extraction.")
    else:
        with st.spinner("Analysing property description..."):
            try:
                extraction_response = post_json(
                    "/extract",
                    {"description": st.session_state.description},
                )
                extraction_payload = extraction_response["extracted_data"]
                st.session_state.extraction_response = extraction_response
                st.session_state.reviewed_features = merge_with_defaults(
                    extraction_payload["features"]
                )
                if extraction_response["fallback"]["used_fallback"]:
                    st.warning(
                        format_backend_fallback(
                            extraction_response["fallback"],
                            "Extraction used a fallback. Review every field before predicting.",
                        )
                    )
                    llm_detail = extract_llm_error_detail(extraction_response["fallback"])
                    if llm_detail:
                        st.error(f"**LLM error:** {llm_detail}")
                else:
                    st.success("Extraction complete. Review the fields below, then run the valuation.")
            except BackendUnavailableError as exc:
                st.error(str(exc))
            except ApiValidationError as exc:
                st.error(str(exc))
            except ApiRequestError as exc:
                st.error(str(exc))

# ── Step 2: Review extracted features ─────────────────────────────────────────

if st.session_state.extraction_response:
    extraction_response = st.session_state.extraction_response
    extraction_payload = extraction_response["extracted_data"]
    missing_fields = extraction_payload["missing_features"]
    field_metadata = extraction_payload.get("field_metadata", {})

    st.divider()
    st.subheader("Review extracted features")

    if missing_fields:
        st.warning(
            f"{len(missing_fields)} field(s) were not found in your description: "
            + ", ".join(f"`{f}`" for f in missing_fields)
            + ". Edit these before running the valuation."
        )
    else:
        st.success(
            "All features were extracted successfully. "
            "Edit any value if needed, then run the valuation."
        )

    if extraction_response["fallback"]["used_fallback"]:
        st.error(
            format_backend_fallback(
                extraction_response["fallback"],
                "Extraction fallback triggered. Please review all fields carefully.",
            )
        )
        llm_detail = extract_llm_error_detail(extraction_response["fallback"])
        if llm_detail:
            st.error(f"**LLM error:** {llm_detail}")

    with st.form("review_form"):
        reviewed_features = render_feature_inputs_grouped(
            st.session_state.reviewed_features,
            missing_fields,
            field_metadata,
        )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Run valuation", use_container_width=True, type="primary"
        )

    if submitted:
        empty_text_fields = [
            f
            for f in FEATURE_DEFAULTS
            if f not in NUMERIC_FIELDS and not reviewed_features[f]
        ]
        if empty_text_fields:
            st.error(
                "Fill in all text fields before predicting: "
                + ", ".join(f"`{f}`" for f in empty_text_fields)
            )
        else:
            st.session_state.reviewed_features = reviewed_features
            with st.spinner("Running prediction and explanation..."):
                try:
                    prediction_response = post_json(
                        "/predict",
                        {"reviewed_features": reviewed_features},
                    )
                    interpretation_response = post_json(
                        "/interpret",
                        {
                            "reviewed_features": reviewed_features,
                            "prediction": prediction_response["prediction"],
                            "summary_stats": prediction_response["summary_stats"],
                        },
                    )
                    st.session_state.prediction_response = prediction_response
                    st.session_state.interpretation_response = interpretation_response
                    st.rerun()
                except BackendUnavailableError as exc:
                    st.error(str(exc))
                except ApiValidationError as exc:
                    st.error(str(exc))
                except ApiRequestError as exc:
                    st.error(str(exc))

# ── Step 3: Valuation result ───────────────────────────────────────────────────

if st.session_state.prediction_response and st.session_state.interpretation_response:
    prediction_response = st.session_state.prediction_response
    interpretation_response = st.session_state.interpretation_response
    predicted_price = prediction_response["prediction"]

    st.divider()
    st.subheader("Valuation result")

    if prediction_response["fallback"]["used_fallback"]:
        st.warning(
            format_backend_fallback(
                prediction_response["fallback"],
                "Prediction used a fallback estimate.",
            )
        )
    if interpretation_response["fallback"]["used_fallback"]:
        st.warning(
            format_backend_fallback(
                interpretation_response["fallback"],
                "Interpretation used a fallback explanation.",
            )
        )
        llm_detail = extract_llm_error_detail(interpretation_response["fallback"])
        if llm_detail:
            st.error(f"**LLM error:** {llm_detail}")

    st.markdown(
        f"""
<div class="price-block">
    <div class="price-label">Estimated Sale Price</div>
    <div class="price-amount">{format_currency(predicted_price)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    render_market_context(predicted_price, prediction_response["summary_stats"])

    st.divider()

    interp_col, feat_col = st.columns([3, 2], gap="large")

    with interp_col:
        st.subheader("AI explanation")
        st.markdown(
            f'<div class="interp-box">{interpretation_response["interpretation"]}</div>',
            unsafe_allow_html=True,
        )

    with feat_col:
        st.subheader("Submitted features")
        features_df = pd.DataFrame(
            [
                {"Feature": FIELD_LABELS.get(k, k), "Value": v}
                for k, v in prediction_response["reviewed_features"].items()
            ]
        )
        st.dataframe(
            features_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    if st.button("Start a new valuation", use_container_width=False):
        reset_pipeline_state()
        st.session_state.description = ""
        st.rerun()

elif not has_extraction:
    st.info(
        "Enter a property description above and click **Extract features** to begin. "
        "Need inspiration? Use one of the example properties in the sidebar."
    )
