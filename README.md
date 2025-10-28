# Sexual Assault Analytics (student project)

This repository contains datasets and an analysis scaffold for a compact coursework project (7th semester) on rape/sexual-assault analytics. The project implements several novel analyses (change-point detection, under-reporting heuristics, composite vulnerability index, network analysis, cohort shifts) and a minimal responsive UI to run and present outputs.

Quick start
1. Create and activate a Python virtualenv (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask UI:

```bash
python webapp/app.py
```

Open http://localhost:5000 in your browser. Click "Run Analyses" to start the analysis in the background. Outputs will appear in the `outputs/` directory and the UI gallery.

What I added
- `scripts/analyses.py`: analysis module with a working change-point analysis (1999-2013) and stubs for remaining analyses.
- `webapp/`: minimal Flask + Bootstrap UI to run analyses and view generated PNG/CSV outputs.
- `requirements.txt` and `README.md`.

Next steps I will take once you confirm: implement each analysis one-by-one, add per-analysis pages and concise interpretation text for submission-ready visuals.

If you prefer Streamlit or a different UI, tell me and I will adapt.
# analytics_by_copilot