from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import streamlit as st


def _clean_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _contains_indic_script(text: str) -> bool:
    # Covers common Indic Unicode blocks.
    indic_script_pattern = r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]"
    return bool(re.search(indic_script_pattern, text))


COMMON_INDIAN_NAME_PARTS = {
    "aarav",
    "aarya",
    "abhishek",
    "aditya",
    "ajay",
    "akshay",
    "alok",
    "amit",
    "ananya",
    "ankit",
    "arjun",
    "ashok",
    "bhavna",
    "deepak",
    "divya",
    "farhan",
    "faizan",
    "gautam",
    "harsha",
    "harish",
    "imran",
    "isha",
    "karthik",
    "madhav",
    "jatin",
    "kavya",
    "kiran",
    "krishna",
    "lakshmi",
    "mohammed",
    "manish",
    "meera",
    "mohit",
    "neha",
    "naveen",
    "pooja",
    "prajwal",
    "nikhil",
    "nisha",
    "pradeep",
    "pranav",
    "priya",
    "rahul",
    "raj",
    "rajesh",
    "rakesh",
    "ravi",
    "rohan",
    "sachin",
    "salman",
    "sanjay",
    "sharath",
    "shivam",
    "shivani",
    "sharma",
    "shreya",
    "singh",
    "sneha",
    "surya",
    "suresh",
    "tanvi",
    "varun",
    "verma",
    "vijay",
    "vikas",
    "vivek",
    "yasmin",
    "zoya",
}

COMMON_INDIAN_SURNAMES = {
    "agarwal",
    "agrawal",
    "banerjee",
    "bhattacharya",
    "bose",
    "chatterjee",
    "chauhan",
    "chopra",
    "das",
    "desai",
    "dixit",
    "gandhi",
    "goyal",
    "gupta",
    "iyer",
    "iyengar",
    "jain",
    "joshi",
    "kapoor",
    "khanna",
    "kumar",
    "mehta",
    "menon",
    "mishra",
    "mukherjee",
    "nair",
    "pandey",
    "patel",
    "pillai",
    "rao",
    "reddy",
    "saxena",
    "shah",
    "sharma",
    "singh",
    "srinivasan",
    "subramanian",
    "trivedi",
    "varma",
    "verma",
}

COMMON_NON_INDIAN_SURNAME_HINTS = {
    "anderson",
    "brown",
    "chen",
    "davis",
    "garcia",
    "hernandez",
    "johnson",
    "jones",
    "kim",
    "lee",
    "lopez",
    "martin",
    "martinez",
    "miller",
    "nguyen",
    "rodriguez",
    "smith",
    "thomas",
    "williams",
    "wilson",
}

INDIAN_SUFFIXES = {
    "appa",
    "bhai",
    "dev",
    "esh",
    "gowda",
    "jee",
    "kar",
    "kumar",
    "murthy",
    "nath",
    "raj",
    "reddy",
    "swamy",
    "wala",
    "wal",
}


@dataclass
class DetectorResult:
    is_indian: bool
    confidence: float
    detector_used: str


def _naampy_detector(name: str) -> DetectorResult | None:
    """
    Best-effort wrapper around Naampy-like APIs.
    Returns None if package/API is unavailable.
    """
    try:
        import naampy  # type: ignore
    except Exception:
        return None

    cleaned = _clean_name(name)

    # Try common API shapes defensively.
    try:
        if hasattr(naampy, "is_indian_name"):
            decision = bool(naampy.is_indian_name(cleaned))
            return DetectorResult(decision, 0.90, "naampy.is_indian_name")
    except Exception:
        pass

    try:
        if hasattr(naampy, "predict_origin"):
            result = naampy.predict_origin(cleaned)
            result_text = str(result).lower()
            decision = "india" in result_text or "indian" in result_text
            return DetectorResult(decision, 0.85, "naampy.predict_origin")
    except Exception:
        pass

    try:
        if hasattr(naampy, "classify"):
            result = naampy.classify(cleaned)
            result_text = str(result).lower()
            decision = "india" in result_text or "indian" in result_text
            return DetectorResult(decision, 0.85, "naampy.classify")
    except Exception:
        pass

    return None


@st.cache_resource(show_spinner=False)
def _load_ethnicolr():
    try:
        # Imported lazily so app still runs when package is absent.
        import ethnicolr  # type: ignore

        return ethnicolr
    except Exception:
        return None


def _ethnicolr_detector(name: str) -> DetectorResult | None:
    """
    Uses ethnicolr wiki-name models when available.
    This is generally stronger for Romanized names than Indic script checks.
    """
    ethnicolr = _load_ethnicolr()
    if ethnicolr is None:
        return None

    cleaned = _clean_name(name)
    if not cleaned:
        return DetectorResult(False, 0.0, "ethnicolr")

    df = pd.DataFrame({"name": [cleaned]})
    prediction_df = None

    # Different ethnicolr versions expose different functions.
    for fn_name in ("pred_wiki_name", "pred_fl_reg_name", "pred_wiki_ln"):
        fn = getattr(ethnicolr, fn_name, None)
        if fn is None:
            continue
        try:
            prediction_df = fn(df.copy(), "name")
            break
        except Exception:
            continue

    if prediction_df is None or prediction_df.empty:
        return None

    row = prediction_df.iloc[0]
    normalized = {str(k).strip().lower(): row[k] for k in row.index}

    # Use probability columns when present. ethnicolr usually exposes one-vs-group scores.
    asian_prob = 0.0
    white_prob = 0.0
    hispanic_prob = 0.0
    black_prob = 0.0
    for col, value in normalized.items():
        try:
            v = float(value)
        except Exception:
            continue
        if "asian" in col:
            asian_prob = max(asian_prob, v)
        elif "white" in col:
            white_prob = max(white_prob, v)
        elif "hispanic" in col or "latino" in col:
            hispanic_prob = max(hispanic_prob, v)
        elif "black" in col or "african" in col:
            black_prob = max(black_prob, v)

    # Conservative proxy: high Asian confidence with low alternative-group confidence.
    non_asian_competitor = max(white_prob, hispanic_prob, black_prob)
    is_indian = asian_prob >= 0.60 and non_asian_competitor <= 0.35
    confidence = asian_prob if asian_prob > 0 else (0.65 if is_indian else 0.40)
    return DetectorResult(is_indian, confidence, "ethnicolr-prob")


@st.cache_resource(show_spinner=False)
def _load_indic_ner_pipeline():
    try:
        from transformers import pipeline

        return pipeline(
            "token-classification",
            model="ai4bharat/IndicNER",
            aggregation_strategy="simple",
        )
    except Exception:
        return None


def _indic_ner_detector(name: str) -> DetectorResult | None:
    """
    IndicNER alone does not classify nationality.
    We treat 'PERSON + Indic script' as likely Indian for lightweight filtering.
    """
    nlp = _load_indic_ner_pipeline()
    if nlp is None:
        return None

    cleaned = _clean_name(name)
    if not cleaned:
        return DetectorResult(False, 0.0, "indicner")

    try:
        entities = nlp(cleaned)
    except Exception:
        return None

    has_person = any("PER" in str(e.get("entity_group", "")).upper() for e in entities)
    has_indic = _contains_indic_script(cleaned)
    decision = bool(has_person and has_indic)
    confidence = 0.80 if decision else 0.35
    return DetectorResult(decision, confidence, "indicner+script")


def _heuristic_detector(name: str) -> DetectorResult:
    cleaned = _clean_name(name)
    parts = re.split(r"[\s\.\-']+", cleaned.lower())
    parts = [p for p in parts if p]

    if not parts:
        return DetectorResult(False, 0.0, "heuristic")

    if _contains_indic_script(cleaned):
        return DetectorResult(True, 0.92, "heuristic-script")

    first_name_hit = parts[0] in COMMON_INDIAN_NAME_PARTS
    surname_hit = parts[-1] in COMMON_INDIAN_SURNAMES
    indian_token_hits = sum(1 for p in parts if p in COMMON_INDIAN_NAME_PARTS)
    non_indian_surname_hit = parts[-1] in COMMON_NON_INDIAN_SURNAME_HINTS
    suffix_hit = any(parts[-1].endswith(sfx) for sfx in INDIAN_SUFFIXES)
    indian_initial_pattern = bool(re.match(r"^[a-z]\.\s*[a-z]+", cleaned.lower()))
    multipart_hint = len(parts) >= 3

    score = 0.20
    if first_name_hit:
        score += 0.30
    if surname_hit:
        score += 0.35
    if indian_token_hits >= 2:
        score += 0.15
    if suffix_hit:
        score += 0.20
    if indian_initial_pattern:
        score += 0.15
    if multipart_hint:
        score += 0.10
    if non_indian_surname_hit:
        score -= 0.35

    score = max(0.0, min(0.95, score))
    is_indian = score >= 0.50 or (first_name_hit and surname_hit)
    return DetectorResult(is_indian, score, "heuristic-lexicon-v3")


def _deterministic_name_rule(name: str) -> DetectorResult | None:
    cleaned = _clean_name(name)
    parts = re.split(r"[\s\.\-']+", cleaned.lower())
    parts = [p for p in parts if p]
    if not parts:
        return None

    first = parts[0]
    last = parts[-1]
    first_hit = first in COMMON_INDIAN_NAME_PARTS
    surname_hit = last in COMMON_INDIAN_SURNAMES
    non_indian_surname_hit = last in COMMON_NON_INDIAN_SURNAME_HINTS

    # Explicit positive override for obvious Indian-name patterns.
    if (first_hit and not non_indian_surname_hit) or (first_hit and surname_hit):
        return DetectorResult(True, 0.86, "deterministic-firstname")

    return None


def _aggregate_detector_results(results: list[DetectorResult]) -> DetectorResult:
    if not results:
        return DetectorResult(False, 0.0, "none")

    used = []
    signed_score = 0.0
    for result in results:
        used.append(result.detector_used)
        if result.is_indian:
            signed_score += result.confidence
        else:
            signed_score -= result.confidence * 0.5

    # Simple bounded projection to [0, 1]
    confidence = max(0.0, min(1.0, (signed_score + 1.0) / 2.0))
    confidence = max(0.0, min(1.0, confidence))
    is_indian = confidence >= 0.50
    return DetectorResult(is_indian, confidence, "+".join(used))


def detect_indian_name(name: str, detector_preference: str) -> DetectorResult:
    cleaned = _clean_name(name)
    if not cleaned:
        return DetectorResult(False, 0.0, "empty")

    deterministic = _deterministic_name_rule(cleaned)
    if deterministic is not None:
        return deterministic

    detector_chain: list[Callable[[str], DetectorResult | None]]
    if detector_preference == "Ethnicolr first (recommended)":
        detector_chain = [_ethnicolr_detector, _naampy_detector]
    else:
        detector_chain = [_naampy_detector, _ethnicolr_detector]

    collected_results: list[DetectorResult] = []
    for detector in detector_chain:
        result = detector(cleaned)
        if result is not None:
            collected_results.append(result)

    # Keep Indic script as a weak bonus signal, not a primary model.
    indic_bonus = _indic_ner_detector(cleaned)
    if indic_bonus is not None and indic_bonus.is_indian:
        collected_results.append(indic_bonus)

    # Always include heuristic as stable fallback signal.
    collected_results.append(_heuristic_detector(cleaned))
    return _aggregate_detector_results(collected_results)


def _to_excel_bytes(indian_df: pd.DataFrame, non_indian_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        indian_df.to_excel(writer, index=False, sheet_name="indian_names")
        non_indian_df.to_excel(writer, index=False, sheet_name="non_indian_names")
    output.seek(0)
    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title="Indian Name Filter", layout="wide")
    st.title("Candidate Name Filter (Indian Names)")
    st.write(
        "Upload an Excel/CSV file, select candidate-name column, and filter out likely Indian names."
    )

    uploaded = st.file_uploader("Upload file", type=["xlsx", "xls", "csv"])
    if uploaded is None:
        return

    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    if df.empty:
        st.warning("Uploaded file is empty.")
        return

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    columns = list(df.columns)
    default_idx = 0
    for idx, col in enumerate(columns):
        if "name" in str(col).lower():
            default_idx = idx
            break

    name_col = st.selectbox("Candidate name column", columns, index=default_idx)
    detector_preference = st.radio(
        "Model preference",
        options=[
            "Ethnicolr first (recommended)",
            "Naampy first",
        ],
        horizontal=True,
    )
    confidence_threshold = st.slider(
        "Indian-name confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )

    if st.button("Filter Indian Names", type="primary"):
        working_df = df.copy()
        working_df[name_col] = working_df[name_col].fillna("").astype(str)

        results = working_df[name_col].apply(
            lambda x: detect_indian_name(x, detector_preference)
        )

        working_df["is_indian_name"] = results.apply(lambda r: r.is_indian)
        working_df["indian_name_confidence"] = results.apply(lambda r: r.confidence)
        working_df["detector_used"] = results.apply(lambda r: r.detector_used)

        indian_df = working_df[
            (working_df["is_indian_name"])
            & (working_df["indian_name_confidence"] >= confidence_threshold)
        ].copy()
        non_indian_df = working_df.drop(indian_df.index).copy()

        c1, c2 = st.columns(2)
        c1.metric("Likely Indian names", len(indian_df))
        c2.metric("Remaining candidates", len(non_indian_df))

        st.subheader("Likely Indian Names")
        st.dataframe(indian_df, use_container_width=True)

        st.subheader("Remaining Candidates")
        st.dataframe(non_indian_df, use_container_width=True)

        excel_data = _to_excel_bytes(indian_df, non_indian_df)
        st.download_button(
            "Download filtered workbook (2 sheets)",
            data=excel_data,
            file_name="filtered_candidates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
