#import os
#print("Working directory:", os.getcwd())

#os.chdir(r"C:\Users\ADMIN\Desktop\INX EMOLOYEE PERFORMANCE PREDICTION")

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Streamlit App
st.title("INX Employee Performance Prediction App")
st.write("This app predicts employee performance based on key features using Gradient Boosting Classifier.")

# Load the trained model and scaler
model=joblib.load("pretrained_model.pkl")
scaler=joblib.load("scaler.pkl")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        # Load your saved model and scaler
        model = joblib.load('gradient_boosting_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'gradient_boosting_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

# Load the model and scaler
model, scaler = load_model()

# User Inputs Section
st.header("Enter Employee Details")

# Top 5 features as identified from your dataset
experience_years = st.slider(
    "Experience Years in Current Role", 
    min_value=0, 
    max_value=30, 
    value=2,
    help="Number of years the employee has been in their current role"
)

work_life_balance = st.slider(
    "Work Life Balance Rating", 
    min_value=1, 
    max_value=4, 
    value=3,
    help="1: Low, 2: Medium, 3: High, 4: Very High"
)

environment_satisfaction = st.slider(
    "Environment Satisfaction Rating", 
    min_value=1, 
    max_value=4, 
    value=3,
    help="1: Low, 2: Medium, 3: High, 4: Very High"
)

last_salary_hike = st.slider(
    "Last Salary Hike Percentage", 
    min_value=0, 
    max_value=25, 
    value=11,
    help="Percentage of last salary hike received"
)

years_since_promotion = st.slider(
    "Years Since Last Promotion", 
    min_value=0, 
    max_value=15, 
    value=2,
    help="Number of years since the employee's last promotion"
)

# Performance categories mapping (adjust based on your actual target variable)
performance_categories = {
    0: "Low Performer",
    1: "Medium Performer", 
    2: "High Performer"
}

# Make Prediction
if st.button("Predict Performance Level"):
    if model is not None and scaler is not None:
        try:
            # Prepare input data in the same order as training
            input_data = np.array([[
                experience_years,
                work_life_balance, 
                environment_satisfaction,
                last_salary_hike,
                years_since_promotion
            ]])
            
            # Scale the input data using the same scaler from training
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
            
            # Get the predicted class and its probability
            predicted_class = prediction[0]
            predicted_probability = probabilities[0][predicted_class]
            
            # Display results
            st.subheader("Prediction Results")
            
            # Performance level with color coding
            if predicted_class == 0:
                st.error(f"**Predicted Performance: {performance_categories[predicted_class]}**")
            elif predicted_class == 1:
                st.warning(f"**Predicted Performance: {performance_categories[predicted_class]}**")
            else:
                st.success(f"**Predicted Performance: {performance_categories[predicted_class]}**")
            
            st.write(f"**Confidence: {predicted_probability:.2%}**")
            
            # Show all probabilities
            st.write("**Detailed Probabilities:**")
            for i, prob in enumerate(probabilities[0]):
                st.write(f"- {performance_categories[i]}: {prob:.2%}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model not loaded. Please check if the model files exist.")

# Add some information about the model
st.sidebar.header("About This Model")
st.sidebar.write("""
This model uses Gradient Boosting Classifier trained on INX Employee Performance dataset.

**Top 5 Features Used:**
- Experience Years in Current Role
- Work Life Balance Rating  
- Environment Satisfaction Rating
- Last Salary Hike Percentage
- Years Since Last Promotion
""")

# Instructions for deployment
st.sidebar.header("Deployment Instructions")
st.sidebar.write("""
1. Save your trained model as 'gradient_boosting_model.pkl'
2. Save your scaler as 'scaler.pkl'
3. Place both files in the same directory as this app
4. Run: `streamlit run app.py`
""")
st.title("Gradient Boosting Classifier App")
st.write("This app makes predictions using a Gradient Boosting Classifier model.")

# Load the model and scaler
model, scaler = load_model()

# User Inputs
st.header("Enter Feature Values")

feature1 = st.number_input("Feature 1", value=0.0, step=0.1)
feature2 = st.number_input("Feature 2", value=0.0, step=0.1)
feature3 = st.number_input("Feature 3", value=0.0, step=0.1)
feature4 = st.number_input("Feature 4", value=0.0, step=0.1)

# Make Prediction
if st.button("Predict Class"):
    # Prepare input data
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    
    # Scale the input data (same scaling used during training)
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    # Display results
    st.subheader(f"Predicted Class: {prediction[0]}")
    st.write(f"Class Probabilities: {probabilities[0]}")

