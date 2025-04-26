import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class L1LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Predict
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Add L1 regularization
            dw += (self.lambda_param / n_samples) * np.sign(self.weights)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class L2LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (
                2 * self.lambda_param / n_samples
            ) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def main():
    st.title("Train and Evaluate")
    
    if 'preprocessed_data' not in st.session_state:
        st.warning("‼️ Preprocessed data not found. Please go to Explore & Preprocess page first.")
        st.stop()

    df = st.session_state['preprocessed_data']

    st.header("Dataset Overview:")
    st.dataframe(df.head())

    target_column = st.selectbox("Select target variable", df.columns, index=df.columns.get_loc("Stress Level") if "Stress Level" in df.columns else 0)
    st.session_state['target_column'] = target_column

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    st.sidebar.header("Model Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
    n_iterations = st.sidebar.slider("Number of Iterations", 100, 2000, 1000)
    lambda_param = st.sidebar.slider("Regularization Strength", 0.0, 1.0, 0.1)

    if st.button("Train L1 and L2 Models"):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop="first", sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)
        X_test_transformed = preprocessor.transform(X_test)

        numeric_feature_names = numeric_features
        categorical_feature_names = []
        for feature in categorical_features:
            categories = X[feature].dropna().unique()[1:]
            categorical_feature_names.extend([f"{feature}_{cat}" for cat in categories])

        all_feature_names = numeric_feature_names + categorical_feature_names

        l1_model = L1LinearRegression(learning_rate, n_iterations, lambda_param)
        l2_model = L2LinearRegression(learning_rate, n_iterations, lambda_param)

        l1_model.fit(X_train_transformed, y_train)
        l2_model.fit(X_train_transformed, y_train)

        st.session_state['l1_model'] = l1_model
        st.session_state['l2_model'] = l2_model
        st.session_state['feature_names'] = all_feature_names
        st.session_state['scaler_means'] = dict(zip(numeric_features, preprocessor.named_transformers_['num'].mean_))
        st.session_state['scaler_stds'] = dict(zip(numeric_features, preprocessor.named_transformers_['num'].scale_))

        st.success("Both L1 and L2 models trained successfully!")

        col1, col2 = st.columns(2)

        for model, model_name, container in zip([l1_model, l2_model], ["L1 (Lasso)", "L2 (Ridge)"], [col1, col2]):
            container.header(f"Feature Importance - {model_name}")
            feature_importance = pd.DataFrame({"Feature": all_feature_names, "Coefficient": np.abs(model.weights)})
            feature_importance = feature_importance.sort_values("Coefficient", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=feature_importance, x="Coefficient", y="Feature", ax=ax)
            ax.set_title(f"Feature Importance ({model_name})")
            container.pyplot(fig)

            y_train_pred = model.predict(X_train_transformed)
            y_val_pred = model.predict(X_val_transformed)
            y_test_pred = model.predict(X_test_transformed)

            train_metrics = evaluate_model(y_train, y_train_pred)
            val_metrics = evaluate_model(y_val, y_val_pred)
            test_metrics = evaluate_model(y_test, y_test_pred)

            container.header(f"Model Performance - {model_name}")
            c1, c2, c3 = container.columns(3)

            with c1:
                container.subheader("Training Set")
                container.metric("MAE", f"{train_metrics[0]:.4f}")
                container.metric("RMSE", f"{train_metrics[1]:.4f}")
                container.metric("R²", f"{train_metrics[2]:.4f}")

            with c2:
                container.subheader("Validation Set")
                container.metric("MAE", f"{val_metrics[0]:.4f}")
                container.metric("RMSE", f"{val_metrics[1]:.4f}")
                container.metric("R²", f"{val_metrics[2]:.4f}")

            with c3:
                container.subheader("Test Set")
                container.metric("MAE", f"{test_metrics[0]:.4f}")
                container.metric("RMSE", f"{test_metrics[1]:.4f}")
                container.metric("R²", f"{test_metrics[2]:.4f}")

            container.header(f"Actual vs Predicted Values - {model_name}")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_test, y_test_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            container.pyplot(fig)

            container.header(f"Model Coefficients - {model_name}")
            coefficients = pd.DataFrame({"Feature": all_feature_names, "Coefficient": model.weights})
            coefficients = coefficients.sort_values("Coefficient", ascending=False)
            container.dataframe(coefficients)

if __name__ == "__main__":
    main()
