# Credit Card Approval Prediction

This repository contains a credit card approval prediction app built with Python and Streamlit.

## What it does
- Loads `credit.csv`.
- Trains a Random Forest classifier with categorical encoding and numerical imputation.
- Predicts whether a new applicant will be approved.
- Displays model metrics and input controls in a Streamlit web app.

## Setup

```powershell
cd "c:\AI Lab\Credit Card"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The app trains the model on startup and caches it for faster repeated predictions.
- If Streamlit installation fails on Python 3.13, consider using Python 3.11 or 3.12 for compatibility.
