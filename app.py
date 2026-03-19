from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.modeling import evaluate_with_ground_truth, train_isolation_forest
from src.pipeline import REQUIRED_COLUMNS, preprocess_attendance

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Attendance Anomaly Detection", layout="wide")

st.title("Employee Attendance Behavior Analysis")
st.caption("Hybrid detection: Isolation Forest + employee-level rolling baseline")

st.header("Dataset Upload")
st.write("Upload a CSV with the required columns or use the generated sample dataset.")

schema_text = ", ".join(REQUIRED_COLUMNS)
with st.expander("Required CSV Schema"):
    st.code(schema_text, language="text")

uploaded_file = st.file_uploader("Upload attendance CSV", type=["csv"])
use_sample_data = st.checkbox("Use local sample dataset (data/sample_attendance.csv)")

raw_df = None
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
elif use_sample_data:
    sample_path = Path("data/sample_attendance.csv")
    if sample_path.exists():
        raw_df = pd.read_csv(sample_path)
    else:
        st.warning("Sample dataset not found. Generate it with: python src/data_generator.py")

if raw_df is None:
    st.info("Upload a CSV or enable sample dataset to continue.")
    st.stop()

st.header("Data Preview")
st.subheader("Raw Dataset Sample")
st.dataframe(raw_df.head(20), use_container_width=True)

try:
    cleaned_df, dropped_df, report = preprocess_attendance(raw_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Rows", report.total_rows)
col2.metric("Valid Rows", report.valid_rows)
col3.metric("Dropped Rows", report.dropped_rows)
col4.metric("Invalid Rows", report.invalid_rows)
col5.metric("Duplicates Removed", report.duplicate_rows)

if not dropped_df.empty:
    st.subheader("Dropped Records")
    st.dataframe(dropped_df.head(50), use_container_width=True)

st.header("Feature Engineering Output")
st.dataframe(cleaned_df.head(25), use_container_width=True)
st.write("Feature summary:")
st.dataframe(cleaned_df.describe(include="all").transpose(), use_container_width=True)

st.header("Model Training")
contamination = st.slider(
    "Anomaly contamination rate",
    min_value=0.01,
    max_value=0.25,
    value=0.08,
    step=0.01,
)

train_clicked = st.button("Train Hybrid Detection Model", type="primary")
if train_clicked:
    try:
        _, scored_df = train_isolation_forest(
            cleaned_df=cleaned_df,
            contamination=float(contamination),
            random_state=42,
            window_days=7,
            min_history=5,
            employee_threshold=2.0,
        )
        eval_summary, confusion_df = evaluate_with_ground_truth(scored_df)
        st.session_state["scored_df"] = scored_df
        st.session_state["eval_summary"] = eval_summary
        st.session_state["confusion_df"] = confusion_df
        st.success("Model training complete.")
    except ValueError as exc:
        st.error(str(exc))

if "scored_df" not in st.session_state:
    st.info("Train the model to view anomaly results.")
    st.stop()

scored_df = st.session_state["scored_df"].copy()
scored_df["date"] = pd.to_datetime(scored_df["date"], errors="coerce")

st.header("Results Visualization")

global_count = int(scored_df["is_anomaly"].sum())
employee_count = int(scored_df["is_anomaly_employee"].sum())
hybrid_count = int(scored_df["is_anomaly_hybrid"].sum())
total_rows = len(scored_df)

m1, m2, m3 = st.columns(3)
m1.metric("Global Anomalies", f"{global_count} ({(global_count / total_rows) * 100:.2f}%)")
m2.metric(
    "Employee Pattern Anomalies",
    f"{employee_count} ({(employee_count / total_rows) * 100:.2f}%)",
)
m3.metric("Hybrid Anomalies", f"{hybrid_count} ({(hybrid_count / total_rows) * 100:.2f}%)")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Work Duration Distribution (Hybrid)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=scored_df,
        x="total_work_minutes",
        hue="is_anomaly_hybrid",
        bins=30,
        kde=True,
        ax=ax,
    )
    ax.set_xlabel("Total Work Minutes")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with chart_col2:
    st.subheader("Hybrid Anomaly Counts by Employee")
    anomaly_by_employee = (
        scored_df.loc[scored_df["is_anomaly_hybrid"]]
        .groupby("employee_id")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    if anomaly_by_employee.empty:
        ax.text(0.5, 0.5, "No anomalies detected", ha="center", va="center")
        ax.axis("off")
    else:
        anomaly_by_employee.plot(kind="bar", ax=ax)
        ax.set_ylabel("Anomaly Count")
        ax.set_xlabel("Employee ID")
    st.pyplot(fig)

st.subheader("Global Anomaly Counts by Employee")
global_by_employee = (
    scored_df.loc[scored_df["is_anomaly"]]
    .groupby("employee_id")
    .size()
    .sort_values(ascending=False)
    .head(15)
)
fig, ax = plt.subplots(figsize=(12, 4))
if global_by_employee.empty:
    ax.text(0.5, 0.5, "No global anomalies detected", ha="center", va="center")
    ax.axis("off")
else:
    global_by_employee.plot(kind="bar", ax=ax, color="#4c72b0")
    ax.set_ylabel("Global Anomaly Count")
    ax.set_xlabel("Employee ID")
st.pyplot(fig)

st.subheader("Top Employees by Login/Logout Pattern Anomalies")
pattern_by_employee = (
    scored_df.loc[scored_df["is_anomaly_employee"]]
    .groupby("employee_id")
    .size()
    .sort_values(ascending=False)
    .head(15)
)
fig, ax = plt.subplots(figsize=(12, 4))
if pattern_by_employee.empty:
    ax.text(0.5, 0.5, "No employee-pattern anomalies", ha="center", va="center")
    ax.axis("off")
else:
    pattern_by_employee.plot(kind="bar", ax=ax, color="#f08a5d")
    ax.set_ylabel("Pattern Anomaly Count")
    ax.set_xlabel("Employee ID")
st.pyplot(fig)

st.subheader("Anomaly Trend (Global vs Employee vs Hybrid)")
trend_df = (
    scored_df.assign(
        global_flag=scored_df["is_anomaly"].astype(int),
        employee_flag=scored_df["is_anomaly_employee"].astype(int),
        hybrid_flag=scored_df["is_anomaly_hybrid"].astype(int),
    )
    .groupby("date", as_index=False)[["global_flag", "employee_flag", "hybrid_flag"]]
    .sum()
)
trend_long = trend_df.melt(
    id_vars="date",
    value_vars=["global_flag", "employee_flag", "hybrid_flag"],
    var_name="signal",
    value_name="count",
)
fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(data=trend_long, x="date", y="count", hue="signal", marker="o", ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel("Detected Anomalies")
st.pyplot(fig)

st.subheader("Detected Hybrid Anomalies (Row-Level)")
hybrid_table = scored_df.loc[scored_df["is_anomaly_hybrid"]].copy()
hybrid_table = hybrid_table.sort_values(by="severity_hybrid", ascending=False)
show_columns = [
    "employee_id",
    "date",
    "login_time",
    "logout_time",
    "department",
    "is_anomaly",
    "is_anomaly_employee",
    "is_anomaly_hybrid",
    "reason_global",
    "reason_employee",
    "reason",
    "severity_employee",
    "severity_hybrid",
    "anomaly_score",
]
optional_columns = [column for column in show_columns if column in hybrid_table.columns]
st.dataframe(hybrid_table[optional_columns].head(200), use_container_width=True)

st.header("Ground Truth Evaluation")
eval_summary = st.session_state.get("eval_summary", pd.DataFrame())
confusion_df = st.session_state.get("confusion_df", pd.DataFrame())
if eval_summary.empty:
    st.info("Ground truth columns not found in dataset. Evaluation is unavailable.")
else:
    st.subheader("Precision / Recall / F1")
    st.dataframe(eval_summary, use_container_width=True)

    st.subheader("Confusion Matrix Counts")
    st.dataframe(confusion_df, use_container_width=True)

st.header("Export")
if st.button("Export Scored Results to output/"):
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M-%S")
    scored_path = output_dir / f"{timestamp}_scored_results.csv"
    summary_path = output_dir / f"{timestamp}_evaluation_summary.csv"
    confusion_path = output_dir / f"{timestamp}_confusion_matrix.csv"

    scored_df.to_csv(scored_path, index=False)
    if not eval_summary.empty:
        eval_summary.to_csv(summary_path, index=False)
        confusion_df.to_csv(confusion_path, index=False)

    st.success(f"Exported scored results: {scored_path}")
    if not eval_summary.empty:
        st.success(f"Exported evaluation summary: {summary_path}")
        st.success(f"Exported confusion matrix: {confusion_path}")
