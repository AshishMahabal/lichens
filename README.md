# Lichen Explorer — USA / CA / LA / Pasadena

Interactive maps of lichen species richness and occurrence density from GBIF.
Created by Ashish Mahabal using ChatGPT 5

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py -- --data-dir ./data_dir

