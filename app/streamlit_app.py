import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = Path("models/xgb_model.joblib")
METRICS_PATH = Path("models/metrics.json")
SAMPLE_PATH = Path("models/app_sample.parquet")

st.set_page_config(page_title="Fraud Detection Platform", layout="wide")
st.title("💳 Financial Fraud Detection & Analytics Platform")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_sample():
    return pd.read_parquet(SAMPLE_PATH)


# Validate required files
missing = [p for p in [MODEL_PATH, METRICS_PATH, SAMPLE_PATH] if not p.exists()]
if missing:
    st.error("Missing required files:\n" + "\n".join([f"- {m}" for m in missing]))
    st.stop()

model = load_model()
metrics = load_metrics()
df = load_sample()

st.sidebar.header("Controls")
default_threshold = float(metrics.get("chosen_threshold", 0.5))
threshold = st.sidebar.slider("Fraud Threshold", 0.01, 0.99, default_threshold, 0.01)

# Ensure required columns exist
required_cols = {"Class", "Amount", "Time", "p_fraud"}
if not required_cols.issubset(df.columns):
    st.error(f"Sample file missing required columns: {required_cols}")
    st.stop()

df = df.copy()
df["pred_fraud"] = (df["p_fraud"] >= threshold).astype(int)

# Confusion components
tp = int(((df["pred_fraud"] == 1) & (df["Class"] == 1)).sum())
fp = int(((df["pred_fraud"] == 1) & (df["Class"] == 0)).sum())
fn = int(((df["pred_fraud"] == 0) & (df["Class"] == 1)).sum())
tn = int(((df["pred_fraud"] == 0) & (df["Class"] == 0)).sum())

precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
fraud_rate = float(df["Class"].mean())
flagged = int(df["pred_fraud"].sum())

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows (App Sample)", f"{len(df):,}")
c2.metric("Fraud Rate", f"{fraud_rate*100:.2f}%")
c3.metric("Flagged for Review", f"{flagged:,}")
c4.metric("Precision", f"{precision:.3f}")
c5.metric("Recall", f"{recall:.3f}")

st.caption(
    f"Baseline model PR-AUC: {metrics['test_pr_auc']:.4f} | ROC-AUC: {metrics['test_roc_auc']:.4f} | "
    f"Default threshold: {metrics['chosen_threshold']:.3f}"
)

# Charts
left, right = st.columns(2)

with left:
    st.subheader("Predicted Fraud Probability Distribution")
    fig = px.histogram(
        df,
        x="p_fraud",
        color=df["Class"].map({0: "Legit", 1: "Fraud"}),
        nbins=50,
        title="p_fraud by True Class",
    )
    fig.add_vline(x=threshold)
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Confusion Matrix (Selected Threshold)")
    cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Legit", "Actual Fraud"],
        columns=["Pred Legit", "Pred Fraud"],
    )
    st.dataframe(cm, use_container_width=True)

st.divider()

a1, a2 = st.columns(2)

with a1:
    st.subheader("Transaction Amount Distribution")
    fig_amt = px.histogram(
        df,
        x="Amount",
        color=df["Class"].map({0: "Legit", 1: "Fraud"}),
        nbins=60,
        title="Amount Distribution (log scale on y)",
        log_y=True,
    )
    st.plotly_chart(fig_amt, width="stretch")

with a2:
    st.subheader("Top Flagged Transactions")
    top = df.sort_values("p_fraud", ascending=False).head(15)[
        ["Time", "Amount", "p_fraud", "Class", "pred_fraud"]
    ]
    st.dataframe(top, use_container_width=True)

st.divider()

st.subheader("Threshold Tradeoff: Review Volume vs Fraud Capture Rate")
true_fraud = int((df["Class"] == 1).sum())
thresholds = [i / 100 for i in range(1, 100)]

review_volume = []
fraud_capture = []

for t in thresholds:
    pred = (df["p_fraud"] >= t).astype(int)
    _tp = int(((pred == 1) & (df["Class"] == 1)).sum())
    review_volume.append(int(pred.sum()))
    fraud_capture.append(_tp / max(true_fraud, 1))

trade = pd.DataFrame(
    {"threshold": thresholds, "review_volume": review_volume, "fraud_capture_rate": fraud_capture}
)

fig_trade = go.Figure()
fig_trade.add_trace(go.Scatter(x=trade["threshold"], y=trade["review_volume"], name="Review Volume"))
fig_trade.add_trace(go.Scatter(x=trade["threshold"], y=trade["fraud_capture_rate"], name="Fraud Capture Rate"))
fig_trade.add_vline(x=threshold)
fig_trade.update_layout(xaxis_title="Threshold")
st.plotly_chart(fig_trade, width="stretch")

