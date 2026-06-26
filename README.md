# Insurance Claim-Settlement Bias & Prediction Dashboard

An interactive Streamlit app that audits an insurance death-claim dataset for **differential
repudiation rates** (age / income / team / state / payment mode) and trains four
classification models to test how predictable the claim outcome is.

> **Read this first — what the dashboard does and does not claim.**
> The app surfaces *disparities* in repudiation rates, computed directly from the data.
> A disparity is **not** proof of bias. The dataset does **not** record whether each
> repudiation was *justified* (fraud, non-disclosure, exclusion clause), so differential
> rates can reflect genuinely different risk books **or** inconsistent adjudication — the
> two cannot be separated from this data alone. Treat every gap as a **lead to audit**, to
> be confirmed against documented repudiation grounds.

---

## What's inside

| Tab | Objective |
|-----|-----------|
| 1️⃣ Descriptive | Cross-tabulations of every categorical field against `POLICY_STATUS`, with counts, row-%, and per-group repudiation charts. |
| 2️⃣ Diagnostic / Bias | χ² tests + Cramér's V to rank which factors drive the outcome; deep dives by team/zone, age band, income quartile. |
| 3️⃣ Feature Engineering | The full cleaning + encoding pipeline, documented and reproducible. |
| 4️⃣ Models | KNN, Decision Tree, Random Forest, Gradient Boosting + metric table. |
| 5️⃣ Evaluation | ROC curves, metric comparison, confusion matrices, feature importance. |
| 📋 Findings | Written summary, caveats, and recommended next steps. |

## Data

`Insurance.csv` (bundled sample, ~1,790 death-claim records). Target = `POLICY_STATUS`
(`Approved Death Claim` vs `Repudiate Death`). You can also upload your own CSV with the
same schema from the sidebar.

Expected columns: `POLICY_NO, PI_NAME, PI_GENDER, SUM_ASSURED, ZONE, PAYMENT_MODE,
EARLY_NON, PI_OCCUPATION, MEDICAL_NONMED, PI_STATE, REASON_FOR_CLAIM, PI_AGE,
PI_ANNUAL_INCOME, POLICY_STATUS`.

---


## Sample outputs

The `sample_outputs/` folder contains the five pre-rendered analysis charts (descriptive,
diagnostic/bias, confusion matrices, ROC + metrics, feature importance). They are *optional* —
the app regenerates them live — but are handy for a quick look or to embed in the README.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints (default http://localhost:8501).

## Deploy free on Streamlit Community Cloud

1. **Create a GitHub repo** and push these files (keep them in the repo root):
   ```
   app.py
   requirements.txt
   Insurance.csv
   README.md
   ```
   ```bash
   git init
   git add app.py requirements.txt Insurance.csv README.md
   git commit -m "Insurance claim bias dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io**, sign in with GitHub.
3. **New app** → pick your repo, branch `main`, main file `app.py` → **Deploy**.
4. First build installs `requirements.txt` (a couple of minutes), then your dashboard is
   live at a shareable `*.streamlit.app` URL.

> If you don't want to commit the data, delete `Insurance.csv` from the repo and rely on
> the sidebar uploader instead — but then the app needs a file uploaded before it renders
> charts. Keeping the sample in the repo makes the deployed app work out of the box.

---

## Method notes (for auditability)

- **Income = 0 is treated as missing**, not as a real zero (≈62% of rows). A binary
  `INCOME_MISSING` flag is added; the value is median-imputed for distance-based models.
- **ZONE case variants are merged** (`South`/`SOUTH` → `SOUTH`).
- **High-cardinality fields** (`ZONE`, `PI_STATE`, `PI_OCCUPATION`, `REASON_FOR_CLAIM`)
  keep their top-N categories; the long tail is bucketed as `Other` (sidebar-controlled).
- **Encoding:** numerics standardised; categoricals one-hot encoded with
  `handle_unknown='ignore'`.
- **Class imbalance** (~68/32) handled with `class_weight='balanced'` on the tree models.
- **Positive class = repudiated claim**, so *recall* = the share of true repudiations the
  model catches.
- **Leakage caution:** `REASON_FOR_CLAIM` and `EARLY_NON` are strongly outcome-related. For
  a fairness audit (vs accuracy), re-run with `ZONE`/`PI_STATE` removed to measure how much
  predictive power rides on processing channel rather than case facts.

All numbers recompute live from the loaded data; nothing in the app is hard-coded.
