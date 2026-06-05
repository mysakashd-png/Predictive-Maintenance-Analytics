import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

st.set_page_config(
    page_title="Model Training",
    layout="wide"
)

st.title("🤖 Model Training & Comparison")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully")

    feature_cols = [
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

    X = df[feature_cols]
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    def evaluate_model(model, name):

        y_pred = model.predict(X_test)

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred
        )

        return [
            name,
            mse,
            rmse,
            r2
        ]

    results = []

    # ------------------------------------------------
    # Linear Regression
    # ------------------------------------------------

    with st.spinner(
        "Training Linear Regression..."
    ):

        lr = LinearRegression()

        lr.fit(
            X_train,
            y_train
        )

        results.append(
            evaluate_model(
                lr,
                "Linear Regression"
            )
        )

    # ------------------------------------------------
    # KNN Hyperparameter Tuning
    # ------------------------------------------------

    with st.spinner(
        "Training KNN..."
    ):

        knn_params = {
            "n_neighbors":[3,5,7,9],
            "weights":[
                "uniform",
                "distance"
            ]
        }

        knn_grid = GridSearchCV(
            KNeighborsRegressor(),
            knn_params,
            cv=3,
            scoring="neg_mean_squared_error"
        )

        knn_grid.fit(
            X_train,
            y_train
        )

        best_knn = (
            knn_grid.best_estimator_
        )

        results.append(
            evaluate_model(
                best_knn,
                "KNN"
            )
        )

    # ------------------------------------------------
    # Decision Tree
    # ------------------------------------------------

    with st.spinner(
        "Training Decision Tree..."
    ):

        dt_params = {
            "max_depth":[
                3,
                5,
                10,
                None
            ]
        }

        dt_grid = GridSearchCV(
            DecisionTreeRegressor(
                random_state=42
            ),
            dt_params,
            cv=3
        )

        dt_grid.fit(
            X_train,
            y_train
        )

        best_dt = (
            dt_grid.best_estimator_
        )

        results.append(
            evaluate_model(
                best_dt,
                "Decision Tree"
            )
        )

    # ------------------------------------------------
    # Random Forest
    # ------------------------------------------------

    with st.spinner(
        "Training Random Forest..."
    ):

        rf_params = {
            "n_estimators":[
                100,
                200
            ],
            "max_depth":[
                5,
                10,
                None
            ]
        }

        rf_grid = GridSearchCV(
            RandomForestRegressor(
                random_state=42
            ),
            rf_params,
            cv=3,
            n_jobs=-1
        )

        rf_grid.fit(
            X_train,
            y_train
        )

        best_rf = (
            rf_grid.best_estimator_
        )

        results.append(
            evaluate_model(
                best_rf,
                "Random Forest"
            )
        )

    # ------------------------------------------------
    # Gradient Boosting
    # ------------------------------------------------

    with st.spinner(
        "Training Gradient Boosting..."
    ):

        gb_params = {
            "n_estimators":[
                100,
                200
            ],
            "learning_rate":[
                0.01,
                0.1
            ]
        }

        gb_grid = GridSearchCV(
            GradientBoostingRegressor(
                random_state=42
            ),
            gb_params,
            cv=3
        )

        gb_grid.fit(
            X_train,
            y_train
        )

        best_gb = (
            gb_grid.best_estimator_
        )

        results.append(
            evaluate_model(
                best_gb,
                "Gradient Boosting"
            )
        )

    # ------------------------------------------------
    # Results Table
    # ------------------------------------------------

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "MSE",
            "RMSE",
            "R2"
        ]
    )

    results_df = results_df.sort_values(
        by="RMSE"
    )

    st.subheader(
        "📊 Model Comparison"
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # ------------------------------------------------
    # RMSE Chart
    # ------------------------------------------------

    st.subheader(
        "📈 RMSE Comparison"
    )

    fig = px.bar(
        results_df,
        x="Model",
        y="RMSE",
        title="RMSE Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------
    # R2 Chart
    # ------------------------------------------------

    st.subheader(
        "📈 R² Comparison"
    )

    fig = px.bar(
        results_df,
        x="Model",
        y="R2",
        title="R² Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------
    # Save Best Model
    # ------------------------------------------------

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    model_dict = {
        "Linear Regression": lr,
        "KNN": best_knn,
        "Decision Tree": best_dt,
        "Random Forest": best_rf,
        "Gradient Boosting": best_gb
    }

    best_model = model_dict[
        best_model_name
    ]

    if st.button(
        "💾 Save Best Model"
    ):

        joblib.dump(
            best_model,
            "models/best_model.pkl"
        )

        st.success(
            f"{best_model_name} saved successfully."
        )

else:

    st.info(
        "Please upload a CSV file."
    )
