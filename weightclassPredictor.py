import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model, setup, pull 

pipeline = load_model("best_model")

st.title('Get your Weight category based on the below criteria')
age = st.slider(label='please enter your age', min_value= 1, max_value=111, value = 25)
gender = st.selectbox('select your gender', ['male', 'female'])
height = st.slider(label='enter your height in cms', min_value= 50, max_value= 250, value = 160)
weight = st.slider(label='enter your weight in kgs', min_value= 10, max_value= 200, value= 60)
ID = 1
BMI = weight/(height/100)**2

attribute = [ID, age, gender, height, weight, BMI]

if st.button('predict'):
    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'ID': [50],
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'BMI': [BMI]
    })
    
    # Predict using the model and user data
    prediction = predict_model(pipeline, data=user_data)
    
    # Display the prediction

    st.write(prediction)