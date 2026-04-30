from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import pandas as pd


def clean_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def contains_indic_script(text: str) -> bool:
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
    "jatin",
    "karthik",
    "kavya",
    "kiran",
    "krishna",
    "lakshmi",
    "madhav",
    "manish",
    "meera",
    "mohammed",
    "mohit",
    "naveen",
    "neha",
    "nikhil",
    "nisha",
    "pooja",
    "pradeep",
    "prajwal",
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
    "sharma",
    "shivam",
    "shivani",
    "shreya",
    "singh",
    "sneha",
    "suresh",
    "surya",
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
    try:
        import naampy  # type: ignore
    except Exception:
        return None

    cleaned = clean_name(name)
    try:
        if hasattr(naampy, "is_indian_name"):
            return DetectorResult(bool(naampy.is_indian_name(cleaned)), 0.90, "naampy.is_indian_name")
    except Exception:
        pass
    try:
        if hasattr(naampy, "predict_origin"):
            result_text = str(naampy.predict_origin(cleaned)).lower()
            return DetectorResult("india" in result_text or "indian" in result_text, 0.85, "naampy.predict_origin")
    except Exception:
        pass
    try:
        if hasattr(naampy, "classify"):
            result_text = str(naampy.classify(cleaned)).lower()
            return DetectorResult("india" in result_text or "indian" in result_text, 0.85, "naampy.classify")
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _load_ethnicolr():
    try:
        import ethnicolr  # type: ignore

        return ethnicolr
    except Exception:
        return None


def _ethnicolr_detector(name: str) -> DetectorResult | None:
    ethnicolr = _load_ethnicolr()
    if ethnicolr is None:
        return None

    cleaned = clean_name(name)
    if not cleaned:
        return DetectorResult(False, 0.0, "ethnicolr-prob")

    df = pd.DataFrame({"name": [cleaned]})
    prediction_df = None
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

    asian_prob, white_prob, hispanic_prob, black_prob = 0.0, 0.0, 0.0, 0.0
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

    non_asian_competitor = max(white_prob, hispanic_prob, black_prob)
    is_indian = asian_prob >= 0.60 and non_asian_competitor <= 0.35
    confidence = asian_prob if asian_prob > 0 else (0.65 if is_indian else 0.40)
    return DetectorResult(is_indian, confidence, "ethnicolr-prob")


def _heuristic_detector(name: str) -> DetectorResult:
    cleaned = clean_name(name)
    parts = [p for p in re.split(r"[\s\.\-']+", cleaned.lower()) if p]
    if not parts:
        return DetectorResult(False, 0.0, "heuristic")
    if contains_indic_script(cleaned):
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
    cleaned = clean_name(name)
    parts = [p for p in re.split(r"[\s\.\-']+", cleaned.lower()) if p]
    if not parts:
        return None
    first, last = parts[0], parts[-1]
    first_hit = first in COMMON_INDIAN_NAME_PARTS
    surname_hit = last in COMMON_INDIAN_SURNAMES
    non_indian_surname_hit = last in COMMON_NON_INDIAN_SURNAME_HINTS
    if (first_hit and not non_indian_surname_hit) or (first_hit and surname_hit):
        return DetectorResult(True, 0.86, "deterministic-firstname")
    return None


def _aggregate_detector_results(results: list[DetectorResult]) -> DetectorResult:
    if not results:
        return DetectorResult(False, 0.0, "none")
    signed_score = 0.0
    used = []
    for result in results:
        used.append(result.detector_used)
        if result.is_indian:
            signed_score += result.confidence
        else:
            signed_score -= result.confidence * 0.5
    confidence = max(0.0, min(1.0, (signed_score + 1.0) / 2.0))
    return DetectorResult(confidence >= 0.50, confidence, "+".join(used))


def detect_indian_name(name: str, detector_preference: str) -> DetectorResult:
    cleaned = clean_name(name)
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
    collected_results.append(_heuristic_detector(cleaned))
    return _aggregate_detector_results(collected_results)
