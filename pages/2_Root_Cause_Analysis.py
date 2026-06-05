
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Root Cause Analysis",
    layout="wide"
)

st.title("🔍 Root Cause Analysis")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully")

    # -------------------------
    # Correlation Heatmap
    # -------------------------

    st.subheader("📌 Correlation Heatmap")

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    )

    corr_matrix = numeric_cols.corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Failure Distribution
    # -------------------------

    st.subheader("📌 Failure Distribution")

    failure_counts = (
        df["failure"]
        .value_counts()
        .reset_index()
    )

    failure_counts.columns = [
        "Failure",
        "Count"
    ]

    fig = px.bar(
        failure_counts,
        x="Failure",
        y="Count",
        title="Failure Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------
    # Device Wise Failures
    # -------------------------

    st.subheader("📌 Device Wise Failures")

    device_failure = (
        df.groupby("device")["failure"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        device_failure,
        x="device",
        y="failure",
        title="Failures by Device"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------
    # Sensor Trend Analysis
    # -------------------------

    st.subheader("📌 Sensor Trend Analysis")

    metric_columns = [
        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric8",
        "metric9"
    ]

    selected_metric = st.selectbox(
        "Select Sensor",
        metric_columns
    )

    fig = px.box(
        df,
        x="failure",
        y=selected_metric,
        color="failure",
        title=f"{selected_metric} vs Failure"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------
    # Feature Importance
    # -------------------------

    st.subheader("📌 Feature Importance")

    X = df[metric_columns]
    y = df["failure"]

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "Feature": metric_columns,
        "Importance": rf.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Sensor Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    # -------------------------
    # Root Cause Summary
    # -------------------------

    st.subheader("📌 Root Cause Insights")

    top_feature = (
        importance_df.iloc[0]["Feature"]
    )

    st.info(
        f"Most influential sensor contributing to failure: {top_feature}"
    )

else:
    st.info(
        "Please upload a CSV file."
    )
