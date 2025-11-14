import streamlit as st
import joblib
import numpy as np
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Streamlit App
st.title("INX Employee Performance Prediction App")
st.write("This app predicts employee performance rating (1-4) based on key features using Gradient Boosting Classifier.")

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
    help="1: Poor, 2: Fair, 3: Good, 4: Excellent"
)

environment_satisfaction = st.slider(
    "Environment Satisfaction Rating", 
    min_value=1, 
    max_value=4, 
    value=3,
    help="1: Poor, 2: Fair, 3: Good, 4: Excellent"
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

# Performance rating mapping (1-4 scale)
performance_ratings = {
    1: {"label": "Low", "description": "Needs improvement in key areas"},
    2: {"label": "Good", "description": "Meets expectations consistently"},
    3: {"label": "Excellent", "description": "Exceeds expectations regularly"},
    4: {"label": "Outstanding", "description": "Consistently exceptional performer"}
}

# Make Prediction
if st.button("Predict Performance Rating"):
    if model is not None and scaler is not None:
        try:
            # Prepare input data
            input_data = np.array([[
                experience_years,
                work_life_balance, 
                environment_satisfaction,
                last_salary_hike,
                years_since_promotion
            ]])
            
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction - get absolute performance rating (1-4)
            predicted_rating = model.predict(input_scaled)[0]
            
            # Ensure the prediction is within 1-4 range
            predicted_rating = max(1, min(4, predicted_rating))
            
            # Get performance details
            performance_info = performance_ratings.get(predicted_rating, 
                {"label": f"Rating {predicted_rating}", "description": "Performance rating"})
            
            # Display results
            st.subheader("Prediction Results")
            
            # Color-coded display based on rating
            if predicted_rating == 1:
                st.error(f"**Predicted Performance Rating: {predicted_rating} - {performance_info['label']}**")
                st.info(f"**Description:** {performance_info['description']}")
            elif predicted_rating == 2:
                st.warning(f"**Predicted Performance Rating: {predicted_rating} - {performance_info['label']}**")
                st.info(f"**Description:** {performance_info['description']}")
            elif predicted_rating == 3:
                st.success(f"**Predicted Performance Rating: {predicted_rating} - {performance_info['label']}**")
                st.info(f"**Description:** {performance_info['description']}")
            else:  # rating 4
                st.success(f"**Predicted Performance Rating: {predicted_rating} - {performance_info['label']}** 🎉")
                st.info(f"**Description:** {performance_info['description']}")
            
            # Display key factors influencing the prediction
            st.subheader("Key Performance Factors")
            factors = [
                f"• Experience in Current Role: {experience_years} years",
                f"• Work-Life Balance: {work_life_balance}/4",
                f"• Environment Satisfaction: {environment_satisfaction}/4", 
                f"• Last Salary Hike: {last_salary_hike}%",
                f"• Years Since Promotion: {years_since_promotion} years"
            ]
            
            for factor in factors:
                st.write(factor)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            # Add debugging info
            if model is not None:
                st.write(f"Model type: {type(model)}")
                st.write(f"Model features expected: {getattr(model, 'n_features_in_', 'Unknown')}")
    else:
        st.error("Model not loaded. Please check if the model files exist.")

# Add deployment information
st.sidebar.header("Deployment Info")
st.sidebar.info("""
**Performance Rating Scale:**
- **1 (Low):** Needs improvement
- **2 (Good):** Meets expectations  
- **3 (Excellent):** Exceeds expectations
- **4 (Outstanding):** Exceptional performer

**Model:** Gradient Boosting Classifier
**Input Features:** 5 key employee metrics
**Output:** Absolute performance rating (1-4)
""")