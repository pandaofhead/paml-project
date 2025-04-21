import streamlit as st
from PIL import Image

# Page Setup
st.set_page_config(page_title="Sleep Health and Lifestyle", layout="wide")
st.title("Welcome to Sleep Health and Lifestyle Insights")
st.markdown("### Project Overview")

# Two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Sleep plays a crucial role in overall health, yet many people suffer from poor sleep quality due to lifestyle and physiological factors. 
    This project aims to address the lack of personalized, data-driven insights into sleep health.

    Our goal is to develop an interactive system that predicts sleep quality using health and lifestyle data. 
    We implement custom machine learning models — particularly L1 and L2 regularized linear regression — to estimate sleep quality scores based on user input.

    The web application includes:
    - Interactive data exploration and preprocessing
    - Model training with hyperparameter tuning
    - Model evaluation using MAE, RMSE, and R²
    - A personalized sleep quality prediction app
    """)

with col2:
    image = Image.open("assets/kate-stone-matheson-uy5t-CJuIK4-unsplash.jpg")
    st.image(image, use_container_width=True, caption="© Kate Stone Matheson on Unsplash")

st.markdown("### Dataset Introduction")
st.markdown("""
    The **Sleep Health and Lifestyle Dataset** from Kaggle contains 400 entries and 13 features,
    covering sleep duration, sleep quality, physical activity, stress levels, BMI categories, blood pressure,
    heart rate, daily steps, and the presence of sleep disorders such as insomnia and sleep apnea.

    This dataset, created by **Laksika Tharmalingam** for educational use, allows us to build a 
    comprehensive view of the relationship between lifestyle factors and sleep health.

    🔗 [View Dataset on Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
    """)

st.markdown("""
    ---
    #### 👇 Ready to begin?
    Click the button below to explore the dataset.
    """)

if st.button("Explore Dataset"):
    st.switch_page("pages/0_Explore_&_Preprocess.py")
