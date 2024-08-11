import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders
with open('label_encoder_model.pkl', 'rb') as file:
    label_encoder_model = pickle.load(file)

with open('label_encoder_fuel.pkl', 'rb') as file:
    label_encoder_fuel = pickle.load(file)

# Load the model metrics
with open('model_metrics.pkl', 'rb') as file:
    model_metrics = pickle.load(file)

# Streamlit app title
st.title("ShareRide Cost Suggestion")


# User inputs
st.header("Enter the ride details")

car_model = st.selectbox("Select Car Model", label_encoder_model.classes_)
fuel_type = st.selectbox("Select Fuel Type", label_encoder_fuel.classes_)
kilometers = st.number_input("Enter Distance (in Kilometers)", min_value=0.0, step=0.1)

# Prediction
if st.button("Predict Cost"):
    if kilometers <= 0:
        st.error("Distance must be greater than 0 kilometers.")
    else:
        # Encode the inputs
        car_model_encoded = label_encoder_model.transform([car_model])[0]
        fuel_type_encoded = label_encoder_fuel.transform([fuel_type])[0]
        
        # Prepare the input for prediction
        X_input = np.array([[car_model_encoded, fuel_type_encoded, kilometers]])
        
        # Make prediction
        predicted_cost = model.predict(X_input)[0]
        
        # Display the prediction
        st.success(f"Predicted Cost for the ride: Rs. {predicted_cost:.2f}")

# Display model performance metrics
st.header("Model Performance Metrics")
st.write(f"Mean Squared Error: {model_metrics['Mean Squared Error']:.2f}")
st.write(f"R^2 Score: {model_metrics['R^2 Score']:.2f}")

