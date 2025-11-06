import streamlit as st
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

st.title("INX Employee Performance Prediction App")
st.write("This app predicts employee performance based on key features using Gradient Boosting Classifier.")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('gradient_boosting_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'pretrained_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_model()

# Performance categories
performance_categories = {
    0: "Low Performer",
    1: "Medium Performer", 
    2: "High Performer"
}

if model is not None and scaler is not None:
    # Get the expected number of features from the scaler
    expected_features = scaler.n_features_in_
    st.sidebar.write(f"Model expects {expected_features} features")
    
    st.header("Enter Employee Details")
    
    # Create input for all expected features with default values
    input_data = []
    feature_names = []
    
    # Top 5 important features (customize these based on your actual feature names)
    important_features = [
        ('experience_years', 'Experience Years in Current Role', 0, 30, 2),
        ('work_life_balance', 'Work Life Balance Rating', 1, 4, 3),
        ('environment_satisfaction', 'Environment Satisfaction Rating', 1, 4, 3),
        ('last_salary_hike', 'Last Salary Hike Percentage', 0, 25, 11),
        ('years_since_promotion', 'Years Since Last Promotion', 0, 15, 2)
    ]
    
    # Collect the 5 important features
    for feature_id, label, min_val, max_val, default_val in important_features:
        value = st.slider(label, min_value=min_val, max_value=max_val, value=default_val)
        input_data.append(value)
        feature_names.append(feature_id)
    
    # For the remaining features, use default values (0 or mean values)
    remaining_features = expected_features - len(important_features)
    
    if remaining_features > 0:
        st.subheader("Additional Employee Information")
        st.info(f"Please provide {remaining_features} additional features or use default values.")
        
        # You can either:
        # Option A: Collect all remaining features (if you know their names)
        # Option B: Use default values for remaining features
        
        # For now, we'll use default values (0) for remaining features
        default_values = [0] * remaining_features
        input_data.extend(default_values)
        
        st.warning(f"Using default values for {remaining_features} additional features.")

    # Make Prediction
    if st.button("Predict Performance Level"):
        try:
            # Prepare input data
            input_array = np.array([input_data])
            
            # Scale the input data
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
            
            # Get the predicted class and its probability
            predicted_class = prediction[0]
            predicted_probability = probabilities[0][predicted_class]
            
            # Display results
            st.subheader("Prediction Results")
            
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
            st.info("This error usually occurs when the number of features doesn't match the trained model.")

else:
    st.error("Model not loaded properly.")

st.sidebar.header("About This Model")
st.sidebar.write("""
This model uses Gradient Boosting Classifier trained on INX Employee Performance dataset.
""")