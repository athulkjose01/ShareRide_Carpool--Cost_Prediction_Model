import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and performance metrics
@st.cache_resource
def load_model():
    with open('carpool_model.pkl', 'rb') as file:
        model = pickle.load(file)
    # Load metrics
    with open('model_metrics.pkl', 'rb') as file:
        metrics = pickle.load(file)
    return model, metrics

# Make predictions
def predict_cost(model, kilometers, car_model):
    input_data = pd.DataFrame({'kilometers': [kilometers], 'car_model': [car_model]})
    prediction = model.predict(input_data)
    return max(prediction[0], 20)  # Ensure a minimum prediction of 20 rupees

# Streamlit app
def main():
    st.title("ShareRide Cost Predictor")
    
    # Load the model and metrics
    model, metrics = load_model()
    

    # User input
    st.header("Predict Cost")
    kilometers = st.number_input("Enter the distance in kilometers:", min_value=0.1, value=10.0, step=0.1)
    car_models = ['maruti', 'kia', 'skoda', 'mahindra', 'tata', 'honda', 'hyundai', 'toyota', 'nissan', 'isuzu', 'benz', 'audi', 'porsche', 'bmw', 'Other']
    car_model = st.selectbox("Select the car model:", car_models)
    
    if car_model == 'Other':
        car_model = st.text_input("Enter the car model:")
    
    if st.button("Predict Cost"):
        if car_model:
            cost = predict_cost(model, kilometers, car_model)
            st.success(f"The predicted cost for the ride is ₹{cost:.2f}")
        else:
            st.warning("Please enter a car model.")

   
    # Display model performance metrics
    st.header("Model Performance")
    st.write(f"Mean Squared Error: {metrics['mse']:.2f}")
    st.write(f"R-squared Score: {metrics['r2']:.4f}")
    st.write(f"Mean Absolute Error: {metrics['mae']:.2f}")
    st.write(f"Accuracy: {metrics['accuracy_percentage']:.2f}%")
    st.write(f"Relative MAE: {metrics['relative_mae_percentage']:.2f}%")
    

if __name__ == "__main__":
    main()
