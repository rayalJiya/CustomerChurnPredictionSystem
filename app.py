import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.common import MODEL_PATH, METADATA_PATH, load_config

st.set_page_config(
    page_title="Churn Intelligence | Customer Retention",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual layer only: prediction/model/risk logic below is preserved.
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#07111f; --panel:#0d1b2d; --panel2:#10233a; --line:#203954; --blue:#38bdf8; --electric:#2563eb; --text:#e8f1ff; --muted:#8fa7c4; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #102b4b 0%, var(--bg) 44%); color:var(--text); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(7,17,31,.82); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b192b,#07111f); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--text); }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#f4f8ff !important; letter-spacing:-.025em; }
p, label, .stMarkdown { color:#c5d5e8; }
.hero { padding:1.6rem 1.8rem; border:1px solid #21456b; border-radius:22px; background:linear-gradient(120deg,rgba(20,53,87,.95),rgba(10,25,43,.94)); box-shadow:0 18px 55px rgba(0,0,0,.18); margin-bottom:1.35rem; }
.eyebrow { color:#65d3ff; text-transform:uppercase; font-size:.75rem; font-weight:700; letter-spacing:.18em; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:2.25rem; line-height:1.15; font-weight:700; color:#f4f8ff; margin:.45rem 0; }
.hero-copy { color:#a9c0db; font-size:1rem; max-width:780px; }
.section-label { color:#6ed7ff; font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.14em; margin:.5rem 0 .65rem; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#10243a,#0b192a); border:1px solid #203d5b; padding:1rem 1.1rem; border-radius:16px; }
div[data-testid="stMetricLabel"] { color:#9fb6d2 !important; }
div[data-testid="stMetricValue"] { color:#f2f7ff !important; font-family:'Space Grotesk',sans-serif; }
.stButton>button, .stDownloadButton>button, button[kind="primary"] { border-radius:10px; border:1px solid #38bdf8; background:linear-gradient(100deg,#2563eb,#0284c7); color:white; font-weight:700; padding:.62rem 1rem; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#7dd3fc; background:linear-gradient(100deg,#1d4ed8,#0369a1); color:white; }
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"],.stFileUploader section { background:#0a192a !important; color:#e8f1ff !important; border-color:#294766 !important; border-radius:10px !important; }
.stTabs [data-baseweb="tab-list"] { gap:.45rem; border-bottom:1px solid #203954; }
.stTabs [data-baseweb="tab"] { background:#0d1b2d; border:1px solid #203954; border-radius:10px 10px 0 0; padding:.7rem 1rem; color:#a9bfd8; }
.stTabs [aria-selected="true"] { background:#153454 !important; color:#72d7ff !important; border-color:#2874ad !important; }
div[data-testid="stForm"] { background:rgba(13,27,45,.72); border:1px solid #203954; padding:1.2rem; border-radius:18px; }
div[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#203954; }
.small-muted { color:#8fa7c4; font-size:.85rem; }
.result-card { padding:1.1rem 1.25rem; border-radius:16px; background:linear-gradient(145deg,#10243a,#0a192a); border:1px solid #234361; height:100%; }
.result-label { color:#8fa7c4; font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; }
.result-value { font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; color:#f4f8ff; margin-top:.35rem; }
</style>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning("Model artifacts are not available yet. Run `python train_model.py` from the project terminal first.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
cfg = load_config()
features = metadata["feature_columns"]
categorical_options = metadata.get("categorical_options", {})
low_max = float(cfg.get("low_risk_max", 0.30))
medium_max = float(cfg.get("medium_risk_max", 0.60))

# Existing risk thresholds and recommendation logic retained.
def get_risk(prob):
    if prob < low_max:
        return "Low"
    if prob < medium_max:
        return "Medium"
    return "High"

def suggestions(row, risk):
    actions = []
    if risk == "High":
        actions.append("Contact the customer proactively and understand their main concern.")
        actions.append("Review support-call history and resolve any unresolved service issues.")
    elif risk == "Medium":
        actions.append("Monitor usage and customer interactions over the next few weeks.")
        actions.append("Offer relevant help or product guidance based on the customer's needs.")
    else:
        actions.append("Maintain the current customer experience and routine engagement.")
    if "Support Calls" in row and row["Support Calls"] >= 7:
        actions.append("High support-call count: review repeat contacts and resolution quality.")
    if "Payment Delay" in row and row["Payment Delay"] >= 20:
        actions.append("Payment delays are elevated: check billing clarity and payment support options.")
    if "Usage Frequency" in row and row["Usage Frequency"] <= 7:
        actions.append("Usage frequency is low: offer onboarding or feature-discovery guidance.")
    if "Contract Length" in row and str(row["Contract Length"]).lower() == "monthly":
        actions.append("Consider whether a suitable longer-term plan provides genuine customer value.")
    return list(dict.fromkeys(actions))

def predict_one(row_df):
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row_df[features])[:, 1][0])
        probability_note = ""
    else:
        score = float(model.decision_function(row_df[features])[0])
        probability = 1 / (1 + math.exp(-max(-30, min(30, score))))
        probability_note = " (approximate score mapping; not calibrated)"
    pred = int(probability >= 0.5)
    risk = get_risk(probability)
    return pred, probability, risk, probability_note

# Sidebar branding / navigation information
with st.sidebar:
    st.markdown("<div class='eyebrow'>◈ CHURN INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("## Retention workspace")
    st.caption("Machine-learning decision support")
    st.divider()
    st.markdown("**Workspace**")
    st.markdown("🔹 Individual prediction  \n🔹 Batch scoring  \n🔹 Model information")
    st.divider()
    st.markdown("**Model selected**")
    st.write(metadata.get("best_model_name", "Saved model"))
    st.markdown("**Risk bands**")
    st.markdown(f"🟢 Low · below {low_max:.0%}  \n🟠 Medium · {low_max:.0%}–{medium_max:.0%}  \n🔴 High · {medium_max:.0%} and above")
    st.divider()
    st.markdown("<div class='small-muted'>Scores are estimates, not guarantees. Review customer context before acting.</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">CUSTOMER ANALYTICS · PREDICT · RETAIN</div>
  <div class="hero-title">Customer Churn Intelligence</div>
  <div class="hero-copy">Identify customers who may be at risk, understand the signals, and support timely retention decisions.</div>
</div>
""", unsafe_allow_html=True)

# Top summary uses actual metadata only; no synthetic KPIs are introduced.
metrics = metadata.get("metrics", {})
with st.container():
    st.markdown("<div class='section-label'>Workspace overview</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected model", str(metadata.get("best_model_name", "Not available")))
    m2.metric("Input features", len(features))
    m3.metric("Risk categories", "3", help="Configured low, medium, and high risk bands")

st.markdown(" ")
tab1, tab2, tab3 = st.tabs(["◈  Individual prediction", "⇧  Batch CSV scoring", "▦  Model information"])

with tab1:
    st.markdown("<div class='section-label'>01 / Customer profile</div>", unsafe_allow_html=True)
    st.subheader("Score an individual customer")
    st.caption("Enter customer details below. The trained preprocessing pipeline is applied automatically.")
    input_values = {}
    with st.form("customer_form"):
        cols = st.columns(3)
        for i, feature in enumerate(features):
            with cols[i % 3]:
                if feature in categorical_options:
                    opts = categorical_options[feature]
                    input_values[feature] = st.selectbox(feature, opts, key=f"individual_{feature}")
                else:
                    defaults = {
                        "Age": (18, 65, 40), "Tenure": (1, 60, 24),
                        "Usage Frequency": (1, 30, 15), "Support Calls": (0, 10, 3),
                        "Payment Delay": (0, 30, 5), "Total Spend": (100, 1000, 500),
                        "Last Interaction": (1, 30, 10)
                    }
                    mn, mx, default = defaults.get(feature, (0, 100000, 0))
                    input_values[feature] = st.number_input(feature, min_value=mn, max_value=mx, value=default, step=1)
        submitted = st.form_submit_button("Run churn assessment  →", type="primary", use_container_width=True)
    if submitted:
        row_df = pd.DataFrame([input_values], columns=features)
        pred, prob, risk, note = predict_one(row_df)
        color = {"Low": "#34d399", "Medium": "#fbbf24", "High": "#fb7185"}[risk]
        prediction_label = "Likely to churn" if pred else "Likely to stay"
        st.markdown("<div class='section-label'>02 / Assessment result</div>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a:
            st.markdown(f"<div class='result-card'><div class='result-label'>Prediction</div><div class='result-value'>{prediction_label}</div></div>", unsafe_allow_html=True)
        with b:
            st.markdown(f"<div class='result-card'><div class='result-label'>Churn score</div><div class='result-value' style='color:{color}'>{prob:.1%}</div></div>", unsafe_allow_html=True)
        with c:
            st.markdown(f"<div class='result-card'><div class='result-label'>Risk category</div><div class='result-value' style='color:{color}'>{risk}</div></div>", unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, prob)))
        if note:
            st.caption("Important: " + note.strip())
        st.markdown("<div class='section-label'>03 / Recommended next actions</div>", unsafe_allow_html=True)
        for item in suggestions(input_values, risk):
            st.markdown(f"- {item}")
        st.caption("Recommendations are rule-based suggestions, not causal explanations or guaranteed outcomes.")

with tab2:
    st.markdown("<div class='section-label'>BATCH SCORING</div>", unsafe_allow_html=True)
    st.subheader("Score multiple customers")
    st.write("Upload a CSV containing all model input columns. `CustomerID` is optional and will be preserved if supplied.")
    upload = st.file_uploader("Choose a customer CSV", type=["csv"], help="Required columns: " + ", ".join(features))
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            missing = [c for c in features if c not in batch.columns]
            if missing:
                st.error(f"Required feature columns missing: {missing}")
            elif batch.empty:
                st.warning("The uploaded CSV has no rows.")
            else:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(batch[features])[:, 1]
                    probability_note = ""
                else:
                    scores = model.decision_function(batch[features])
                    probs = 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
                    probability_note = " (approximate, not calibrated)"
                out = batch.copy()
                out["churn_probability"] = probs
                out["predicted_churn"] = (probs >= 0.5).astype(int)
                out["risk_category"] = [get_risk(float(p)) for p in probs]
                high_count = int((out["risk_category"] == "High").sum())
                k1, k2, k3 = st.columns(3)
                k1.metric("Rows scored", f"{len(out):,}")
                k2.metric("High-risk customers", f"{high_count:,}")
                k3.metric("High-risk share", f"{high_count / len(out):.1%}")
                st.info(f"Churn scores{probability_note}. SVM score mappings are approximate and not calibrated probabilities.")
                st.dataframe(out.head(100), use_container_width=True, hide_index=True)
                st.download_button("Download scored CSV  ↓", out.to_csv(index=False).encode("utf-8"), "churn_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not read or score this CSV: {exc}")

with tab3:
    st.markdown("<div class='section-label'>MODEL TRANSPARENCY</div>", unsafe_allow_html=True)
    st.subheader("Model information")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Selected model**")
        st.markdown(f"<div class='result-card'><div class='result-label'>Model</div><div class='result-value'>{metadata.get('best_model_name', 'Not available')}</div></div>", unsafe_allow_html=True)
        st.markdown("**Input features**")
        st.write(features)
        st.markdown("**Excluded identifier columns**")
        st.write(metadata.get("id_columns_excluded", []))
    with right:
        st.markdown("**Evaluation metrics from saved metadata**")
        st.json(metrics)
        st.markdown("**Risk thresholds**")
        st.write(f"Low: < {low_max:.0%} · Medium: {low_max:.0%} to < {medium_max:.0%} · High: ≥ {medium_max:.0%}")
    st.warning("Treat predictions as decision support, not certainty. Review class-specific recall, precision, calibration, fairness, and business costs before deployment. Retention suggestions are rule-based examples.")

st.markdown("---")
st.markdown("<div class='small-muted'>CHURN INTELLIGENCE · Built with Streamlit and scikit-learn · Model outputs depend on the saved training artifacts.</div>", unsafe_allow_html=True)
