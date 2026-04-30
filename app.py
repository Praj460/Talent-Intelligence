import io

import pandas as pd
import streamlit as st

from backend import detect_indian_name


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
