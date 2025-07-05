import streamlit as st
import joblib
import numpy as np
import pandas as pd

# load model and expected features
model = joblib.load("heart_disease_model.pkl")
features = joblib.load("model_features.pkl")

st.title("Heart Disease Predictor")
st.markdown("Enter patient information to predict heart disease risk.")

# inupt fields — for training features
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
physical_health = st.slider("Physical Unhealthy Days (last 30 days)", 0, 30)
mental_health = st.slider("Mentally Unhealthy Days (last 30 days)", 0, 30)
sleep_time = st.slider("Average Sleep Time (hours)", 0, 24)

# binary features
smoking = st.selectbox("Do you smoke", ['No', 'Yes'])
alcohol = st.selectbox("Do you consume alcohol?", ['No', 'Yes'])
stroke = st.selectbox("History of Stroke?", ['No', 'Yes'])
diff_walking = st.selectbox("Difficulty Walking?", ['No', 'Yes'])
sex = st.selectbox("Sex", ['Female', 'Male'])
physical_activity = st.selectbox("Physically Active", ['No', 'Yes'])
asthma = st.selectbox("Asthma?", ['No', 'Yes'])
kidney_disease = st.selectbox("Kidney Disease?", ['No', 'Yes'])
skin_cancer = st.selectbox("Skin Cancer?", ['No', 'Yes'])

# multi=category inputs
diabetic = st.selectbox("Diabetic?", ['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)'])
gen_health = st.selectbox("General Health", ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
age_category = st.selectbox("Age Category", [
    '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
    '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
])
race = st.selectbox("Race", ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other'])

if st.button("Predic Risk"):
    # manual encoding — must match training
    input_dict = {
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'SleepTime': sleep_time,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'AlcoholDrinking': 1 if alcohol == 'Yes' else 0,
        "Stroke": 1 if stroke == 'Yes' else 0,
        'DiffWalking': 1 if diff_walking == 'Yes' else 0,
        'Sex': 1 if sex == 'Male' else 0,
        'PhysicalActivity': 1 if physical_activity == 'Yes' else 0,
        'Asthma': 1 if asthma == 'Yes' else 0,
        'KidneyDisease': 1 if kidney_disease == 'Yes' else 0,
        'SkinCancer': 1 if skin_cancer == 'Yes' else 0
    }
    
    # map input variables for categorical features
    user_inputs = {
        "Diabetic": diabetic,
        "GenHealth": gen_health,
        "AgeCategory": age_category,
        "Race": race
    }
    
    # one-hot encode categorical inputs
    for cat_prefix in ["Diabetic", "GenHealth", "AgeCategory", "Race"]:
        for col in features:
            if col.startswith(cat_prefix + "_"):
                val = col.split("_", 1)[1]
                user_val = user_inputs[cat_prefix]
                input_dict[col] = 1 if user_val == val else 0
                
    # fill missing one-hot columns with 0
    for col in features:
        if col not in input_dict:
            input_dict[col] = 0
    
    # convert to dataframe in correct order
    input_df = pd.DataFrame([input_dict])[features]
    
    # predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("Prediction:")
    st.write("🩺 **At Risk**" if prediction == 1 else "✅ **Not at Risk**")
    st.write(f"Confidence Score: `{probability * 100: .2f}%`")