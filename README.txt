# Cognitive Screening Research Prototype

## Included files
- `app.py` — Streamlit web application
- `alzheimers_screening_model.joblib` — your trained model bundle
- `requirements.txt` — Python dependencies

## Run on Windows
1. Install Python 3.10 or 3.11.
2. Extract this ZIP file.
3. Open Command Prompt in the extracted folder.
4. Run:

```bash
python -m venv venv
venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

5. Streamlit will show a local address, usually `http://localhost:8501`. Open it in your browser.

## Important research limitations
The supplied model was trained on a synthetic dataset. This application is an academic software demonstration only, not a clinical screening or diagnostic tool. It must not be used with real patients or for healthcare decisions.

The form uses the exact feature columns saved with the trained model. Consult the original dataset codebook for the units and numeric coding of every field. Do not guess values or category codes. The app deliberately does not reinterpret target class 0/1 as a medical diagnosis.

The saved model was serialized with scikit-learn 1.6.1, so the requirements file pins that version for compatibility.
