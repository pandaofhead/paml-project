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


def load_and_preprocess_data():
    # Load the real sleep disorder dataset
    df = pd.read_excel("Preprocessed_Sleep_Health_Dataset.xlsx")

    # Display basic information about the dataset
    st.header("Dataset Overview:")
    st.write(f"Number of samples: {len(df)}")
    st.write("Available columns:", df.columns.tolist())

    # Let user select target variable
    target_column = st.selectbox("Select target variable (Sleep Duration)", df.columns)

    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_features = [col for col in numeric_features if col != target_column]
    categorical_features = df.select_dtypes(include=["object"]).columns

    st.write("\nNumeric Features:", list(numeric_features))
    st.write("Categorical Features:", list(categorical_features))

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y, numeric_features, categorical_features


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    st.title("Sleep Disorder Prediction using Machine Learning")
    st.write(
        "This application uses L1 and L2 Regularization Linear Regression to predict sleep disorders based on health and lifestyle data."
    )

    # Sidebar for model parameters
    st.sidebar.header("Model Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
    n_iterations = st.sidebar.slider("Number of Iterations", 100, 2000, 1000)
    lambda_param = st.sidebar.slider("L1 Regularization Parameter", 0.0, 1.0, 0.1)
    model_type = st.sidebar.selectbox(
        "Regularization Type", ["L1 (Lasso)", "L2 (Ridge)"]
    )

    # Load and preprocess data
    X, y, numeric_features, categorical_features = load_and_preprocess_data()

    # Create preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Transform the data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names after transformation
    numeric_feature_names = numeric_features
    categorical_feature_names = []
    for i, feature in enumerate(categorical_features):
        unique_values = X[feature].unique()
        for value in unique_values[1:]:  # Skip first value due to drop='first'
            categorical_feature_names.append(f"{feature}_{value}")

    all_feature_names = list(numeric_feature_names) + categorical_feature_names

    # Train model
    # model = L1LinearRegression(
    #     learning_rate=learning_rate,
    #     n_iterations=n_iterations,
    #     lambda_param=lambda_param,
    # )
    if model_type == "L1 (Lasso)":
        model = L1LinearRegression(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            lambda_param=lambda_param,
        )
    else:
        model = L2LinearRegression(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            lambda_param=lambda_param,
        )
    model.fit(X_train_transformed, y_train)

    # Display feature importance
    st.header("Feature Importance")
    feature_importance = pd.DataFrame(
        {"Feature": all_feature_names, "Coefficient": np.abs(model.weights)}
    )
    feature_importance = feature_importance.sort_values("Coefficient", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="Coefficient", y="Feature", ax=ax)
    ax.set_title(f"Feature Importance ({model_type} Regularization)")

    st.pyplot(fig)

    # Make predictions
    y_train_pred = model.predict(X_train_transformed)
    y_val_pred = model.predict(X_val_transformed)
    y_test_pred = model.predict(X_test_transformed)

    # Evaluate model
    train_metrics = evaluate_model(y_train, y_train_pred)
    val_metrics = evaluate_model(y_val, y_val_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)

    # Display results
    st.header("Model Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Training Set")
        st.write(f"MAE: {train_metrics[0]:.4f}")
        st.write(f"RMSE: {train_metrics[1]:.4f}")
        st.write(f"R²: {train_metrics[2]:.4f}")

    with col2:
        st.subheader("Validation Set")
        st.write(f"MAE: {val_metrics[0]:.4f}")
        st.write(f"RMSE: {val_metrics[1]:.4f}")
        st.write(f"R²: {val_metrics[2]:.4f}")

    with col3:
        st.subheader("Test Set")
        st.write(f"MAE: {test_metrics[0]:.4f}")
        st.write(f"RMSE: {test_metrics[1]:.4f}")
        st.write(f"R²: {test_metrics[2]:.4f}")

    # Plot actual vs predicted
    st.header("Actual vs Predicted Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_test_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    st.pyplot(fig)

    # Display model coefficients
    st.header("Model Coefficients")
    coefficients = pd.DataFrame(
        {"Feature": all_feature_names, "Coefficient": model.weights}
    )
    coefficients = coefficients.sort_values("Coefficient", ascending=False)
    st.dataframe(coefficients)


if __name__ == "__main__":
    main()
