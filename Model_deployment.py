import streamlit as st
import joblib
import numpy as np
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Streamlit App
st.title("INX Employee Performance Prediction App")
st.write("This app predicts employee performance based on key features using Gradient Boosting Classifier.")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        # Load your saved model and scaler
        model = joblib.load('gradient_boosting_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'pretrained_model.pkl' and 'scaler.pkl' are in the same directory.")
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

# Performance categories mapping
performance_categories = {
    0: "Low Performer",
    1: "Medium Performer", 
    2: "High Performer"
}

# Make Prediction
if st.button("Predict Performance Level"):
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
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
            
            # Get the predicted class and its probability
            predicted_class_index = prediction[0]
            
            # ✅ SAFE ACCESS: Check if index exists
            if predicted_class_index < len(probabilities[0]):
                predicted_probability = probabilities[0][predicted_class_index]
            else:
                predicted_probability = 0.0
                st.warning("Warning: Class index out of bounds, using default probability")
            
            # Display results
            st.subheader("Prediction Results")
            
            # ✅ SAFE PERFORMANCE MAPPING
            performance_labels = {
                0: "Low Performer",
                1: "Medium Performer", 
                2: "High Performer",
                3: "Exceptional Performer"  # Add if your model has 4 classes
            }
            
            # Use the actual predicted class from the model
            predicted_class_label = performance_labels.get(
                predicted_class_index, 
                f"Class {predicted_class_index}"
            )
            
            # Color coding
            if predicted_class_index == 0:
                st.error(f"**Predicted Performance: {predicted_class_label}**")
            elif predicted_class_index == 1:
                st.warning(f"**Predicted Performance: {predicted_class_label}**")
            else:
                st.success(f"**Predicted Performance: {predicted_class_label}**")
            
            st.write(f"**Confidence: {predicted_probability:.2%}**")
            
            # ✅ SAFE PROBABILITY DISPLAY
            st.write("**Detailed Probabilities:**")
            for i, prob in enumerate(probabilities[0]):
                label = performance_labels.get(i, f"Class {i}")
                st.write(f"- {label}: {prob:.2%}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            # Add debugging info
            if model is not None:
                st.write(f"Model classes: {model.classes_}")
                st.write(f"Number of classes: {len(model.classes_)}")
    else:
        st.error("Model not loaded. Please check if the model files exist.")