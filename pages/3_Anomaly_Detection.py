import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Anomaly Detection",
    layout="wide"
)

st.title("⚠️ Anomaly Detection")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ----------------------------
    # Date Conversion
    # ----------------------------

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # ----------------------------
    # Sidebar Controls
    # ----------------------------

    st.sidebar.header("Settings")

    selected_device = st.sidebar.selectbox(
        "Select Device",
        sorted(df["device"].unique())
    )

    threshold = st.sidebar.slider(
        "Z-Score Threshold",
        min_value=2.0,
        max_value=4.0,
        value=3.0,
        step=0.1
    )

    selected_metric = st.sidebar.selectbox(
        "Select Sensor",
        [
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
    )

    # ----------------------------
    # Device Filter
    # ----------------------------

    filtered_df = df[
        df["device"] == selected_device
    ].copy()

    # ----------------------------
    # Rolling Statistics
    # ----------------------------

    filtered_df["Rolling_Mean"] = (
        filtered_df[selected_metric]
        .rolling(window=10)
        .mean()
    )

    filtered_df["Rolling_STD"] = (
        filtered_df[selected_metric]
        .rolling(window=10)
        .std()
    )

    # ----------------------------
    # Z Score
    # ----------------------------

    mean_value = (
        filtered_df[selected_metric]
        .mean()
    )

    std_value = (
        filtered_df[selected_metric]
        .std()
    )

    filtered_df["Z_Score"] = (
        (
            filtered_df[selected_metric]
            - mean_value
        )
        / std_value
    )

    # ----------------------------
    # Anomaly Flag
    # ----------------------------

    filtered_df["Anomaly_Alert"] = np.where(
        np.abs(
            filtered_df["Z_Score"]
        ) > threshold,
        1,
        0
    )

    # ----------------------------
    # KPIs
    # ----------------------------

    st.subheader("📊 Monitoring Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Records",
            len(filtered_df)
        )

    with col2:
        st.metric(
            "Current Value",
            round(
                filtered_df[selected_metric]
                .iloc[-1],
                2
            )
        )

    with col3:
        st.metric(
            "Active Alerts",
            int(
                filtered_df[
                    "Anomaly_Alert"
                ].sum()
            )
        )

    # ----------------------------
    # Sensor Drift Chart
    # ----------------------------

    st.subheader("📈 Sensor Drift Analysis")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered_df["date"],
            y=filtered_df[selected_metric],
            mode="lines",
            name="Sensor Value"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=filtered_df["date"],
            y=filtered_df["Rolling_Mean"],
            mode="lines",
            name="Rolling Mean"
        )
    )

    failure_limit = (
        mean_value +
        threshold * std_value
    )

    fig.add_hline(
        y=failure_limit,
        line_color="red",
        annotation_text="Failure Ceiling"
    )

    fig.update_layout(
        title=f"{selected_metric} Drift Analysis",
        xaxis_title="Date",
        yaxis_title=selected_metric
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------
    # Anomaly Table
    # ----------------------------

    st.subheader("🚨 Detected Anomalies")

    anomaly_df = filtered_df[
        filtered_df[
            "Anomaly_Alert"
        ] == 1
    ]

    st.dataframe(
        anomaly_df,
        use_container_width=True
    )

    # ----------------------------
    # Download Report
    # ----------------------------

    csv = anomaly_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Anomaly Report",
        data=csv,
        file_name="anomaly_report.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Please upload a CSV file."
    )
