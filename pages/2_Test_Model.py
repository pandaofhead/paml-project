import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Setup
st.set_page_config(page_title="Sleep Health and Lifestyle Dataset", layout="wide")
st.title("Test Trained Models")

# Evaluate function
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Collect available models
available_models = []
if 'L1_model' in st.session_state:
    available_models.append('L1 (Lasso)')
if 'L2_model' in st.session_state:
    available_models.append('L2 (Ridge)')

if available_models:
    st.sidebar.title("🔍 Model Selection")
    selected_models = st.sidebar.multiselect("Select trained model(s) to evaluate", available_models, default=available_models)

    if not selected_models:
        st.info("Please select one or more models from the sidebar to evaluate.")

    if len(selected_models) == 2:
        col_main1, col_main2 = st.columns(2)
    else:
        col_main1 = col_main2 = None

    for idx, model_key in enumerate(selected_models):
        model_session_key = 'L1_model' if 'L1' in model_key else 'L2_model'
        model_data = st.session_state[model_session_key]
        model = model_data['model']
        name = model_data['name']
        X_train, y_train = model_data['X_train'], model_data['y_train']
        X_val, y_val = model_data['X_val'], model_data['y_val']
        X_test, y_test = model_data['X_test'], model_data['y_test']
        feature_names = model_data['feature_names']

        container = col_main1 if idx == 0 and col_main1 else col_main2 if idx == 1 and col_main2 else st

        container.markdown(f"### 📈 {name} Evaluation")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_metrics = evaluate_model(y_train, y_train_pred)
        val_metrics = evaluate_model(y_val, y_val_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)

        container.subheader("Model Performance")
        c1, c2, c3 = container.columns(3)
        with c1:
            container.markdown("**Training Set**")
            container.metric("MAE", f"{train_metrics[0]:.4f}")
            container.metric("RMSE", f"{train_metrics[1]:.4f}")
            container.metric("R²", f"{train_metrics[2]:.4f}")
        with c2:
            container.markdown("**Validation Set**")
            container.metric("MAE", f"{val_metrics[0]:.4f}")
            container.metric("RMSE", f"{val_metrics[1]:.4f}")
            container.metric("R²", f"{val_metrics[2]:.4f}")
        with c3:
            container.markdown("**Test Set**")
            container.metric("MAE", f"{test_metrics[0]:.4f}")
            container.metric("RMSE", f"{test_metrics[1]:.4f}")
            container.metric("R²", f"{test_metrics[2]:.4f}")

        # Actual vs Predicted Plot
        container.subheader("Actual vs Predicted (Test Set)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test, y_test_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Actual vs Predicted: {name}")
        container.pyplot(fig)

        # Coefficients
        container.subheader("Model Coefficients")
        coefficients = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": model.weights
        }).sort_values("Coefficient", key=abs, ascending=False)
        container.dataframe(coefficients)

        # Feature Importance
        container.subheader("Feature Importance")
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(
            data=coefficients,
            x="Coefficient",
            y="Feature",
            ax=ax_imp,
            orient="h",
            hue="Feature",
            legend=False
        )
        ax_imp.set_title(f"Feature Importance: {name} Regularization")
        container.pyplot(fig_imp)

elif not available_models:
    st.warning("‼️ Trained models not found. Please go to **Train Model** and complete training first")
