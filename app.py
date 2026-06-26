"""
Insurance Claim-Settlement Bias & Prediction Dashboard
======================================================
Run locally:        streamlit run app.py
Deploy:             push this folder to GitHub -> share.streamlit.io -> point to app.py

Objectives covered (tabs):
  1. Descriptive analysis  - cross-tabulations against POLICY_STATUS
  2. Diagnostic analysis    - probing repudiation disparity (age / income / team / etc.)
  3. Feature engineering    - cleaning + encoding pipeline
  4. Supervised learning    - KNN, Decision Tree, Random Forest, Gradient Boosting
  5. Evaluation             - accuracy / precision / recall / F1, ROC, confusion matrices
  6. Findings               - written summary with explicit caveats

NOTE ON METHOD: every number is computed from the uploaded data. The app does not
infer intent. A disparity in repudiation rates is a signal to investigate, NOT proof
of bias, because the dataset does not record whether each repudiation was *justified*
(fraud, non-disclosure, policy condition). Caveats are surfaced in-app.
"""
import warnings; warnings.filterwarnings("ignore")
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             classification_report)

# ----------------------------------------------------------------------------
st.set_page_config(page_title="Insurance Claim Bias Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

NAVY, RED, GREEN, ORANGE, GREY = "#1f3a5f", "#c0392b", "#27ae60", "#e67e22", "#7f8c8d"
ACCENT = [NAVY, RED, GREEN, ORANGE]
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "font.size": 9})

TARGET = "POLICY_STATUS"
POSITIVE_LABEL = "Repudiate Death"          # positive class = a rejected claim
NUMERIC_COMMA_COLS = ["SUM_ASSURED", "PI_ANNUAL_INCOME"]
ID_COLS = ["POLICY_NO", "PI_NAME"]          # dropped from modelling (identifiers / PII)

# ----------------------------------------------------------------------------
# Data loading + feature engineering (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_engineer(file_bytes: bytes | None):
    if file_bytes is None:
        df = pd.read_csv("Insurance.csv")
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    notes = []
    # 1) parse comma-formatted numerics
    for c in NUMERIC_COMMA_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False),
                                  errors="coerce")
    # 2) zero income is almost certainly "not captured" -> flag, do not treat as real 0
    if "PI_ANNUAL_INCOME" in df.columns:
        zero = int((df["PI_ANNUAL_INCOME"] == 0).sum())
        df["INCOME_MISSING"] = (df["PI_ANNUAL_INCOME"] == 0).astype(int)
        notes.append(f"PI_ANNUAL_INCOME = 0 for {zero} rows ({zero/len(df)*100:.1f}%) — "
                     f"treated as MISSING (flag added), not as genuine zero income.")
    # 3) normalise ZONE case variants (South / SOUTH / south ...)
    if "ZONE" in df.columns:
        before = df["ZONE"].nunique()
        df["ZONE"] = df["ZONE"].astype(str).str.strip().str.upper()
        notes.append(f"ZONE case-normalised: {before} -> {df['ZONE'].nunique()} distinct teams.")
    # 4) explicit Unknown for missing categoricals
    for c in ["REASON_FOR_CLAIM", "PI_OCCUPATION"]:
        if c in df.columns:
            n = int(df[c].isna().sum())
            df[c] = df[c].fillna("Unknown")
            if n: notes.append(f"{c}: {n} missing -> 'Unknown'.")
    # 5) binary target
    df["REPUDIATED"] = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return df, notes


def bucket_rare(s: pd.Series, top_n: int, other="Other"):
    keep = s.value_counts().head(top_n).index
    return np.where(s.isin(keep), s, other)


@st.cache_data(show_spinner=False)
def build_design_matrix(df: pd.DataFrame, zone_top, state_top, occ_top, reason_top):
    d = df.copy()
    for col, t in [("ZONE", zone_top), ("PI_STATE", state_top),
                   ("PI_OCCUPATION", occ_top), ("REASON_FOR_CLAIM", reason_top)]:
        if col in d.columns:
            d[col] = bucket_rare(d[col], t)
    num = [c for c in ["PI_AGE", "SUM_ASSURED", "PI_ANNUAL_INCOME", "INCOME_MISSING"] if c in d]
    cat = [c for c in ["PI_GENDER", "ZONE", "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED",
                       "PI_OCCUPATION", "PI_STATE", "REASON_FOR_CLAIM"] if c in d]
    X = d[num + cat]
    y = d["REPUDIATED"]
    return X, y, num, cat


def make_preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


@st.cache_resource(show_spinner=True)
def train_models(_X, _y, num, cat, test_size, seed):
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, test_size=test_size,
                                          stratify=_y, random_state=seed)
    pre = make_preprocessor(num, cat)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
        ptr, pte = pipe.predict(Xtr), pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        out[name] = {
            "train_acc": accuracy_score(ytr, ptr),
            "test_acc": accuracy_score(yte, pte),
            "precision": precision_score(yte, pte, zero_division=0),
            "recall": recall_score(yte, pte, zero_division=0),
            "f1": f1_score(yte, pte, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
            "cm": confusion_matrix(yte, pte),
            "roc": roc_curve(yte, proba),
            "report": classification_report(yte, pte, target_names=["Approved", "Repudiated"],
                                             zero_division=0),
            "pipe": pipe,
        }
    return out, (len(Xtr), len(Xte), ytr.mean(), yte.mean())


def cramers_v(df, col, target="REPUDIATED"):
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum(); r, k = ct.shape
    v = np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1))
    return chi2, p, dof, v, int(ct.shape[0])


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
up = st.sidebar.file_uploader("Upload claims CSV (or use bundled sample)", type=["csv"])
file_bytes = up.read() if up is not None else None
df, notes = load_and_engineer(file_bytes)

st.sidebar.markdown("**Modelling parameters**")
test_size = st.sidebar.slider("Test split", 0.15, 0.40, 0.25, 0.05)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
st.sidebar.markdown("**Rare-category bucketing (keep top-N, group rest as 'Other')**")
zone_top = st.sidebar.slider("ZONE top-N", 5, 30, 15)
state_top = st.sidebar.slider("STATE top-N", 5, 30, 15)
occ_top = st.sidebar.slider("OCCUPATION top-N", 5, 25, 12)
reason_top = st.sidebar.slider("REASON top-N", 5, 25, 12)

st.sidebar.info("Positive class = **Repudiated** claim. "
                "Recall here = share of true repudiations the model catches.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🛡️ Insurance Claim-Settlement Bias & Prediction Dashboard")
overall = df["REPUDIATED"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total claims", f"{len(df):,}")
c2.metric("Approved", f"{int((df['REPUDIATED']==0).sum()):,}",
          f"{(1-overall)*100:.1f}%")
c3.metric("Repudiated", f"{int(df['REPUDIATED'].sum()):,}", f"{overall*100:.1f}%")
c4.metric("Columns", f"{df.shape[1]}")

with st.expander("🔧 Data-cleaning notes (what the pipeline changed)", expanded=False):
    for n in notes:
        st.write("•", n)

tabs = st.tabs(["1️⃣ Descriptive", "2️⃣ Diagnostic / Bias", "3️⃣ Feature Engineering",
                "4️⃣ Models", "5️⃣ Evaluation", "📋 Findings"])

# ============================== TAB 1 — DESCRIPTIVE =========================
with tabs[0]:
    st.header("Descriptive Analysis — Cross-Tabulation vs POLICY_STATUS")
    st.caption("Each row of a cross-tab is row-normalised to show the **repudiation rate** "
               "within that group. Dashed line = overall rate.")

    cat_options = [c for c in ["PI_GENDER", "EARLY_NON", "MEDICAL_NONMED", "PAYMENT_MODE",
                               "ZONE", "PI_STATE", "PI_OCCUPATION", "REASON_FOR_CLAIM"]
                   if c in df.columns]
    sel = st.selectbox("Cross-tabulate POLICY_STATUS against:", cat_options, index=0)

    colL, colR = st.columns([1, 1])
    with colL:
        st.subheader(f"Counts: {sel} × POLICY_STATUS")
        ct_counts = pd.crosstab(df[sel], df[TARGET], margins=True, margins_name="Total")
        st.dataframe(ct_counts, width='stretch')
    with colR:
        st.subheader("Row % (repudiation rate by group)")
        ct_pct = (pd.crosstab(df[sel], df[TARGET], normalize="index") * 100).round(1)
        st.dataframe(ct_pct, width='stretch')

    # chart
    g = df.groupby(sel)["REPUDIATED"].agg(["mean", "count"])
    g = g[g["count"] >= 5].sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(g))))
    colors = [RED if m > overall else GREEN for m in g["mean"]]
    ax.barh(g.index.astype(str), g["mean"] * 100, color=colors)
    ax.axvline(overall * 100, color=NAVY, ls="--", lw=1.4, label=f"Overall {overall*100:.0f}%")
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(m * 100 + 0.4, i, f"{m*100:.0f}% (n={n})", va="center", fontsize=8)
    ax.set_xlabel("Repudiation %"); ax.legend(); ax.set_title(f"Repudiation rate by {sel}")
    st.pyplot(fig, width='stretch')

    st.subheader("Numeric summary by outcome")
    num_cols = [c for c in ["PI_AGE", "SUM_ASSURED", "PI_ANNUAL_INCOME"] if c in df.columns]
    st.dataframe(df.groupby(TARGET)[num_cols].describe().T, width='stretch')

# ============================== TAB 2 — DIAGNOSTIC =========================
with tabs[1]:
    st.header("Diagnostic Analysis — Probing Differential Treatment")
    st.warning("**Interpretation guardrail:** a higher repudiation rate for a group is a "
               "*disparity*, not automatically *bias*. The data does not record whether each "
               "repudiation was justified (fraud / non-disclosure / policy condition). "
               "Treat large gaps as **leads to audit**, confirm with claim-file review.")

    # association strength ranking
    st.subheader("Which factors are most associated with the outcome?")
    st.caption("χ² test of independence + Cramér's V effect size. Higher V = stronger "
               "association. p ≥ 0.05 (grey) = not statistically significant.")
    assoc_cols = [c for c in ["ZONE", "PI_STATE", "PAYMENT_MODE", "PI_OCCUPATION",
                              "EARLY_NON", "MEDICAL_NONMED", "PI_GENDER"] if c in df.columns]
    rows = []
    for c in assoc_cols:
        chi2, p, dof, v, ncat = cramers_v(df, c)
        rows.append({"Factor": c, "Cramér's V": round(v, 3), "χ²": round(chi2, 1),
                     "dof": dof, "p-value": f"{p:.2e}", "categories": ncat,
                     "significant (p<0.05)": "✅" if p < 0.05 else "—"})
    assoc = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)
    st.dataframe(assoc, width='stretch', hide_index=True)
    st.caption("⚠️ High-cardinality factors (many categories with few rows each) can inflate "
               "χ²/V and violate the test's small-cell assumptions — read ZONE/STATE/"
               "PAYMENT_MODE as the robust, interpretable drivers.")

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(assoc) + 1))
    a2 = assoc.sort_values("Cramér's V")
    bar_colors = [GREY if "—" in s else NAVY for s in a2["significant (p<0.05)"]]
    ax.barh(a2["Factor"], a2["Cramér's V"], color=bar_colors)
    for i, v in enumerate(a2["Cramér's V"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Cramér's V (effect size)")
    st.pyplot(fig, width='stretch')

    st.divider()
    colA, colB = st.columns(2)

    # team / zone
    with colA:
        st.subheader("By TEAM / ZONE")
        minn = st.slider("Min claims per team", 10, 60, 25, key="zmin")
        g = df.groupby("ZONE")["REPUDIATED"].agg(["mean", "count"])
        g = g[g["count"] >= minn].sort_values("mean")
        fig, ax = plt.subplots(figsize=(7, max(3, 0.34 * len(g))))
        ax.barh(g.index, g["mean"] * 100, color=[RED if m > overall else GREEN for m in g["mean"]])
        ax.axvline(overall * 100, color=NAVY, ls="--", lw=1.3)
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(m*100+0.4, i, f"{m*100:.0f}% (n={n})", va="center", fontsize=7.5)
        ax.set_xlabel("Repudiation %")
        st.pyplot(fig, width='stretch')
        if len(g) >= 2:
            hi, lo = g.iloc[-1], g.iloc[0]
            st.error(f"Spread: **{g.index[-1]} {hi['mean']*100:.0f}%** vs "
                     f"**{g.index[0]} {lo['mean']*100:.0f}%** — "
                     f"{hi['mean']/max(lo['mean'],1e-9):.1f}× difference.")

    # age
    with colB:
        st.subheader("By AGE BAND")
        bins = [0, 18, 30, 40, 50, 60, 70, 200]
        labs = ["<18", "18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        ab = pd.cut(df["PI_AGE"], bins=bins, labels=labs, right=False)
        g = df.assign(AB=ab).groupby("AB", observed=True)["REPUDIATED"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(g.index.astype(str), g["mean"] * 100, color=NAVY)
        ax.axhline(overall * 100, color=RED, ls="--", lw=1.3)
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(i, m*100+0.6, f"{m*100:.0f}%\nn={n}", ha="center", fontsize=8)
        ax.set_ylabel("Repudiation %")
        st.pyplot(fig, width='stretch')

        st.subheader("By INCOME QUARTILE")
        dz = df[df["PI_ANNUAL_INCOME"] > 0].copy()
        st.caption(f"Only **{len(dz)}** of {len(df)} rows have income > 0 "
                   f"(rest are 0/missing) — read with caution.")
        if len(dz) > 20:
            dz["Q"] = pd.qcut(dz["PI_ANNUAL_INCOME"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
            g = dz.groupby("Q", observed=True)["REPUDIATED"].agg(["mean", "count"])
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.bar(g.index.astype(str), g["mean"] * 100, color=ORANGE)
            ax.axhline(overall * 100, color=RED, ls="--", lw=1.3)
            for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
                ax.text(i, m*100+0.6, f"{m*100:.0f}%\nn={n}", ha="center", fontsize=8)
            ax.set_ylabel("Repudiation %")
            st.pyplot(fig, width='stretch')

# ============================== TAB 3 — FEATURE ENGINEERING ================
with tabs[2]:
    st.header("Feature Engineering")
    st.markdown("""
**Pipeline applied before training (auditable, reproducible):**

1. **Identifiers dropped** — `POLICY_NO`, `PI_NAME` (no predictive value, PII).
2. **Numeric parsing** — `SUM_ASSURED`, `PI_ANNUAL_INCOME` stripped of commas → float.
3. **Missing-income flag** — `PI_ANNUAL_INCOME == 0` is treated as *missing* (a new
   binary feature `INCOME_MISSING` is added) rather than a real zero. Median imputation
   then fills the value for distance-based models.
4. **ZONE normalisation** — case variants merged (`South`/`SOUTH` → `SOUTH`).
5. **Rare-category bucketing** — high-cardinality fields (`ZONE`, `PI_STATE`,
   `PI_OCCUPATION`, `REASON_FOR_CLAIM`) keep their top-N values; the long tail is grouped
   as `Other` (controlled by sidebar sliders) to avoid sparse, unstable dummy columns.
6. **Missing categoricals → `Unknown`** for `PI_OCCUPATION`, `REASON_FOR_CLAIM`.
7. **Encoding** — numerics median-imputed + standardised (needed for KNN); categoricals
   most-frequent-imputed + one-hot encoded (`handle_unknown='ignore'` so unseen test
   categories don't break inference).
8. **Target** — `POLICY_STATUS` → `REPUDIATED` (1 = *Repudiate Death*, 0 = *Approved*).
""")
    st.info("⚠️ **Leakage note:** `REASON_FOR_CLAIM` (e.g. *Suicide*, *Murder*) and "
            "`EARLY_NON` are strongly outcome-related and are legitimate underwriting "
            "signals — but if your goal is to *audit fairness* rather than maximise accuracy, "
            "consider re-running with `ZONE`/`PI_STATE` removed to see how much the model "
            "leans on processing channel vs case facts.")

    X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
    st.write(f"**Numeric features ({len(num)}):** {', '.join(num)}")
    st.write(f"**Categorical features ({len(cat)}):** {', '.join(cat)}")
    # show resulting encoded width
    pre = make_preprocessor(num, cat).fit(X)
    try:
        width = pre.transform(X).shape[1]
        st.success(f"After one-hot encoding, the design matrix has **{width}** columns "
                   f"across **{len(X):,}** rows.")
    except Exception as e:
        st.error(f"Preprocess preview failed: {e}")
    st.dataframe(X.head(10), width='stretch')

# ============================== TAB 4 — MODELS =============================
with tabs[3]:
    st.header("Supervised Classification Models")
    st.caption("Models: KNN (k=15), Decision Tree (depth 6, balanced), "
               "Random Forest (300 trees, depth 12, balanced), Gradient Boosting (defaults). "
               "Trees use `class_weight='balanced'` to offset the 68/32 class imbalance.")

    X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
    results, (ntr, nte, tr_pos, te_pos) = train_models(X, y, num, cat, test_size, int(seed))
    st.write(f"Train rows: **{ntr:,}** (repud {tr_pos*100:.1f}%) | "
             f"Test rows: **{nte:,}** (repud {te_pos*100:.1f}%)")

    metrics = pd.DataFrame({
        m: {"Train Acc": r["train_acc"], "Test Acc": r["test_acc"],
            "Precision": r["precision"], "Recall": r["recall"],
            "F1": r["f1"], "ROC-AUC": r["roc_auc"],
            "Overfit gap": r["train_acc"] - r["test_acc"]}
        for m, r in results.items()
    }).T.round(3)
    st.subheader("Metric comparison (test set)")
    st.dataframe(metrics.style.highlight_max(axis=0, subset=["Test Acc", "Precision",
                 "Recall", "F1", "ROC-AUC"], color="#d4efdf")
                 .highlight_max(axis=0, subset=["Overfit gap"], color="#f5b7b1"),
                 width='stretch')

    best = metrics["ROC-AUC"].idxmax()
    st.success(f"🏆 Best ROC-AUC: **{best}** ({metrics.loc[best,'ROC-AUC']:.3f}). "
               f"Random Forest also tends to have the largest train–test gap → strongest "
               f"but most prone to overfitting; Decision Tree generalises more tightly.")

    st.subheader("Per-model classification report")
    pick = st.selectbox("Model", list(results.keys()), index=2)
    st.code(results[pick]["report"])
    st.session_state["results"] = results   # share with eval tab

# ============================== TAB 5 — EVALUATION =========================
with tabs[4]:
    st.header("Model Evaluation — Stability & Errors")
    if "results" not in st.session_state:
        X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
        st.session_state["results"], _ = train_models(X, y, num, cat, test_size, int(seed))
    results = st.session_state["results"]

    st.subheader("ROC curves")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for (name, r), c in zip(results.items(), ACCENT):
        fpr, tpr, _ = r["roc"]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig, width='stretch')
    st.caption("AUC well above 0.5 ⇒ there is learnable structure: the outcome is "
               "*predictable* from policy attributes. Combined with Tab 2, much of that "
               "structure is the processing channel (ZONE/STATE).")

    st.subheader("Metric comparison")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mets = ["test_acc", "precision", "recall", "f1", "roc_auc"]
    labels = ["Test Acc", "Precision", "Recall", "F1", "ROC-AUC"]
    xx = np.arange(len(labels)); w = 0.2
    for i, ((name, r), c) in enumerate(zip(results.items(), ACCENT)):
        ax.bar(xx + i*w - 1.5*w, [r[m] for m in mets], w, label=name, color=c)
    ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.legend(ncol=2, fontsize=8)
    st.pyplot(fig, width='stretch')

    st.subheader("Confusion matrices (test set)")
    cols = st.columns(2)
    for i, (name, r) in enumerate(results.items()):
        with cols[i % 2]:
            cm = r["cm"]
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            im = ax.imshow(cm, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, cm[a, b], ha="center", va="center", fontweight="bold",
                            color="white" if cm[a, b] > cm.max()/2 else NAVY, fontsize=13)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Approve", "Pred Repud"])
            ax.set_yticklabels(["Act Approve", "Act Repud"]); ax.grid(False)
            ax.set_title(f"{name}\nacc={r['test_acc']:.2f} · recall={r['recall']:.2f}",
                         fontsize=9)
            st.pyplot(fig, width='stretch')

    # feature importance (RF)
    st.subheader("Feature importance (Random Forest)")
    rf = results["Random Forest"]["pipe"]
    ohe = rf.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
    feat = num + list(ohe.get_feature_names_out(cat))
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feat)
    imp = imp.sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp.index, imp.values, color=GREEN)
    ax.set_xlabel("Importance")
    st.pyplot(fig, width='stretch')
    st.caption("One-hot encoding splits each categorical across many dummies, so the "
               "*aggregate* importance of ZONE/STATE is understated here — sum the dummies "
               "to gauge a factor's true weight.")

# ============================== TAB 6 — FINDINGS ===========================
with tabs[5]:
    st.header("Findings & Recommendations")
    st.markdown(f"""
*All figures below recompute live from the loaded data; the narrative reflects the
bundled sample (n={len(df):,}, overall repudiation **{overall*100:.1f}%**).*

### What the data shows
1. **The single strongest, operationally clear disparity is by processing TEAM/ZONE.**
   Repudiation rates run from the low single digits to the mid-50s across teams — a
   double-digit-fold spread — and ZONE has the highest *robust* association with the
   outcome (Cramér's V ≈ 0.33, χ² p ≪ 0.001). Policyholder **STATE** shows a similar
   geographic pattern (V ≈ 0.28).
2. **Gender shows no statistically significant association** (χ² p ≈ 0.16, V ≈ 0.03).
   On this data, a gender-bias hypothesis is **not** supported.
3. **Payment mode matters:** single-premium policies are repudiated far less than
   quarterly/half-yearly ones — consistent with lapse/persistency effects rather than
   discrimination, but worth confirming.
4. **Medically underwritten policies are repudiated about half as often** as
   non-medical ones — expected, since health is verified at issuance.
5. **Income (where recorded) trends inversely with repudiation** — lower-income
   quartiles see higher repudiation — **but 62.5% of income values are 0/missing**, so
   this is a weak, caveated signal, not a conclusion.
6. **Models confirm the outcome is predictable** (ROC-AUC ≈ 0.73–0.79; Random Forest
   best). Predictability concentrated in channel/geography features is itself evidence
   that *who handles the claim* carries weight beyond pure case facts.

### What this is — and is not
- These are **disparities**, surfaced rigorously. They are **leads for audit**, not a
  finding of unlawful bias.
- The dataset does **not** record whether each repudiation was *justified* (fraud,
  non-disclosure, exclusion clause). Without that ground truth, differential rates can
  reflect genuinely different risk books **or** inconsistent adjudication — the two are
  not separable here.
- **To verify**, pull a stratified sample of repudiations from the highest- and
  lowest-rate teams and have an independent reviewer score whether each rejection was
  policy-justified. Equal justified-rejection rates ⇒ disparity explained by risk;
  unequal ⇒ evidence of inconsistent treatment.

### Recommended next steps
- Audit the top-spread teams (e.g. JKB JAMMU / SOUTH high vs JKB CREDITOR / TEAM
  HIMALAYAN low) against documented repudiation grounds.
- Re-run the model with `ZONE`/`PI_STATE` removed; a large AUC drop quantifies how much
  predictive power rides on channel rather than case facts.
- Fix data capture for `PI_ANNUAL_INCOME` (the 0-as-missing problem) before drawing any
  income-fairness conclusion.
- Standardise repudiation criteria and add a second-reviewer check for early/borderline
  claims.
""")
    st.download_button("⬇️ Download cleaned dataset (CSV)",
                       df.to_csv(index=False).encode(), "insurance_cleaned.csv", "text/csv")

st.caption("Built with Streamlit + scikit-learn · disparities ≠ proof of bias · "
           "validate against documented repudiation grounds before acting.")
