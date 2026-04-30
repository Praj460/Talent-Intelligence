# Indian Candidate Name Filter

This project lets you upload an Excel/CSV file, detect likely Indian names from a selected candidate name column, and download a filtered workbook.

## What it uses

- `ethnicolr` as the primary detector (recommended for most Romanized names)
- `naampy` as optional secondary signal
- deterministic + heuristic rules as stable fallback

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## How filtering works

1. Choose your candidate name column.
2. Pick detector preference (`Ethnicolr first (recommended)` or `Naampy first`).
3. Set a confidence threshold.
4. Download output Excel with:
   - `indian_names` sheet
   - `non_indian_names` sheet

## Note

For most resume datasets with Romanized names, use `Ethnicolr first (recommended)`.
