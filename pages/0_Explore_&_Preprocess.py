import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from helper_functions import fetch_dataset

# Page Setup
st.set_page_config(page_title="Sleep Health and Lifestyle", layout="wide")
st.title("Explore and Preprocess Dataset")

# Load Dataset
df_raw = fetch_dataset()

if df_raw is not None:
    st.sidebar.title("Explore & Preprocess")
    section = st.sidebar.radio("Select Section", ["Explore Dataset", "Preprocessing"])

    if section == "Explore Dataset":
        st.markdown("## Dataset Overview")

        st.markdown("### Dataset Preview")
        st.dataframe(df_raw.head())

        st.markdown("### Summary Statistics")
        st.dataframe(df_raw.describe())

        st.markdown("## Visualization")
        st.info("Note: All visualizations below are based on the original raw dataset (not preprocessed).")

        visual_option = st.radio("Select a Visualization", [
            "Histogram: Sleep Duration",
            "Histogram: Quality of Sleep",
            "Boxplot: Sleep Quality vs. Gender",
            "Boxplot: Sleep Quality vs. Occupation",
            "Scatterplot: Physical Activity vs. Sleep Quality",
            "Correlation Heatmap"
        ])

        if visual_option == "Histogram: Sleep Duration":
            fig, ax = plt.subplots()
            sns.histplot(df_raw['Sleep Duration'], bins=20, kde=True, ax=ax)
            ax.set_title("Distribution of Sleep Duration")
            ax.set_xlabel("Sleep Duration (normalized)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        elif visual_option == "Histogram: Quality of Sleep":
            fig, ax = plt.subplots()
            sns.histplot(df_raw['Quality of Sleep'], bins=10, kde=True, ax=ax)
            ax.set_title("Distribution of Sleep Quality Score")
            ax.set_xlabel("Quality of Sleep")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        elif visual_option == "Boxplot: Sleep Quality vs. Gender":
            fig, ax = plt.subplots()
            sns.boxplot(x='Gender', y='Quality of Sleep', data=df_raw, hue='Gender', ax=ax)
            ax.set_title("Sleep Quality versus Gender")
            ax.set_xlabel("Gender")
            ax.set_ylabel("Quality of Sleep")
            st.pyplot(fig)

        elif visual_option == "Boxplot: Sleep Quality vs. Occupation":
            fig, ax = plt.subplots()
            sns.boxplot(x='Occupation', y='Quality of Sleep', data=df_raw, hue='Occupation', ax=ax)
            ax.set_title("Sleep Quality versus Occupation")
            ax.set_xlabel("Occupation")
            ax.set_ylabel("Quality of Sleep")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        elif visual_option == "Scatterplot: Physical Activity vs. Sleep Quality":
            fig, ax = plt.subplots()
            sns.scatterplot(x='Physical Activity Level', y='Quality of Sleep', data=df_raw, ax=ax)
            ax.set_title("Physical Activity versus Sleep Quality")
            ax.set_xlabel("Physical Activity Level (normalized)")
            ax.set_ylabel("Quality of Sleep")
            st.pyplot(fig)

        elif visual_option == "Correlation Heatmap":
            fig, ax = plt.subplots()
            sns.heatmap(df_raw.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            ax.set_title("Correlation Heatmap of Numerical Features")
            st.pyplot(fig)

    elif section == "Preprocessing":
        st.markdown("## Data Preprocessing")
        df = df_raw.copy()

        st.subheader("Split Blood Pressure into Systolic and Diastolic")
        if 'Blood Pressure' in df.columns:
            try:
                df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
                st.success("Split 'Blood Pressure' into 'Systolic_BP' and 'Diastolic_BP'.")
            except Exception as e:
                st.warning(f"Could not split 'Blood Pressure': {e}")
        else:
            st.warning("'Blood Pressure' column not found in dataset.")

        st.subheader("Drop Unused Columns")
        df.drop(columns=['Person ID', 'Blood Pressure'], inplace=True, errors='ignore')
        st.success("Dropped 'Person ID' and original 'Blood Pressure' columns.")

        st.subheader("One-Hot Encoding for Categorical Column")
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
        categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
        st.write(f"Encoding: {categorical_cols}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        st.success("Categorical features encoded using one-hot encoding.")

        st.subheader("Handle Missing Values")
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        st.success("Numerical columns filled with median, categorical columns filled with mode.")

        st.subheader("Min-Max Scaling for Numerical Features")
        numerical_cols = ['Age', 'Sleep Duration', 'Physical Activity Level',
                          'Stress Level', 'Heart Rate', 'Daily Steps',
                          'Systolic_BP', 'Diastolic_BP']
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        st.success("Applied Min-Max Scaling to numerical features.")

        st.subheader("Cap outliers in numerical columns using IQR")
        def cap_outliers_iqr(df, cols):
            for col in cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)
            return df
        df = cap_outliers_iqr(df, numerical_cols)
        st.success("Outliers capped using IQR method.")

        st.subheader("Apply PCA to numerical columns (after scaling)")
        pca_cols = df.select_dtypes(include=['float64', 'int64']).columns
        pca = PCA(n_components=0.95)
        principal_components = pca.fit_transform(df[pca_cols])
        pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
        st.success("Applied PCA to retain 95% variance.")

        st.subheader("Feature Engineering")
        df['Sleep Efficiency'] = df['Quality of Sleep'] / (df['Sleep Duration'] + 1e-5)
        df['Stress_Activity_Ratio'] = df['Stress Level'] / (df['Physical Activity Level'] + 1e-5)
        st.success("Created 'Sleep Efficiency' and 'Stress_Activity_Ratio'.")

        st.subheader("Final Preprocessed Dataset")
        st.dataframe(df.head())

        st.session_state['preprocessed_data'] = df
        st.info("Preprocessing complete.")

else:
    st.warning("‼️ Please download the dataset from [Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) and upload it to proceed.")
