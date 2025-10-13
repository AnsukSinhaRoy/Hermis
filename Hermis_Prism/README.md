# Portfolio_Viz

Streamlit-based visualization app for portfolio backtesting experiments.  
This repo provides a modular Streamlit app (`streamlit_app.py`) and a package `portfolio_viz/` with `loaders`, `utils`, `viz`, and `ui` modules.

---

## Quick start

### Prereqs
- Python 3.9+ (3.10 or 3.11 recommended)
- `pip` or a virtualenv manager
- (Optional) GitHub account for CI

### Install
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.\.venv\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt
