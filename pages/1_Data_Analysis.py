import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Analysis", layout="wide")

st.title("📊 Data Analysis")

uploaded_file = st.file_uploader(
    "Upload Predictive Maintenance Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Devices",
            df["device"].nunique()
        )

    st.subheader("Missing Values")

    missing_df = pd.DataFrame(
        df.isnull().sum(),
        columns=["Missing Values"]
    )

    st.dataframe(missing_df)

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe()
    )

    st.subheader("Failure Distribution")

    failure_count = (
        df["failure"]
        .value_counts()
        .reset_index()
    )

    failure_count.columns = [
        "Failure",
        "Count"
    ]

    fig = px.pie(
        failure_count,
        values="Count",
        names="Failure",
        title="Failure Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Device Distribution")

    device_count = (
        df["device"]
        .value_counts()
        .reset_index()
    )

    device_count.columns = [
        "Device",
        "Count"
    ]

    fig = px.bar(
        device_count,
        x="Device",
        y="Count",
        title="Records per Device"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Sensor Distribution")

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

    fig = px.histogram(
        df,
        x=selected_metric,
        nbins=30,
        title=f"{selected_metric} Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.info("Please upload a CSV file.")
