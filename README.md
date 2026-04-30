# Indian Candidate Name Filter

This project lets you upload an Excel/CSV file, detect likely Indian names from a selected candidate name column, and download a filtered workbook.

## What it uses

- `naampy` (preferred when available)
- `IndicNER` as optional secondary detector
- lightweight heuristic fallback (for reliability if model packages are unavailable)

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
2. Pick detector preference (`Naampy first` or `IndicNER first`).
3. Set a confidence threshold.
4. Download output Excel with:
   - `indian_names` sheet
   - `non_indian_names` sheet

## Note

`IndicNER` is a named entity model, not a nationality classifier. In this app, it is used with script checks as a lightweight approximation. For best accuracy on Romanized names, keep `Naampy first`.
