# Prepaid to Postpaid Migration Predictor

A machine learning web app that predicts whether a prepaid customer is likely to migrate to a postpaid plan, based on their demographic and behavioral data.

## Features Used
- **Demographics** — Age, Region, Tenure
- **Top-up History** — Frequency, Average Amount, Regularity Score
- **Voice Usage** — Outgoing & Incoming Call Minutes
- **Data Usage** — Monthly MB Consumed
- **SMS Activity** — Messages Sent per Month
- **USSD Queries** — Short code usage frequency
- **Roaming Activity** — Whether customer has roamed

## Algorithm
XGBoost Classifier — 80% accuracy on test set

---

## How to Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
1. Create a new GitHub repository (e.g. `migration-predictor`)
2. Upload these files:
   - `app.py`
   - `model.pkl`
   - `requirements.txt`

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select your repository and set **Main file path** to `app.py`
5. Click **Deploy**

Your app will be live at:
`https://your-username-migration-predictor.streamlit.app`

### Step 3 — Share the link
Copy the URL and bring it to your interview!

---

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
