import streamlit as st 
import pandas as pd 
from ydata_profiling import ProfileReport
import os 
from streamlit_pandas_profiling import st_profile_report
import pycaret
from pycaret.regression import setup, compare_models, pull, save_model
from pycaret.classification import setup, compare_models, pull, save_model, evaluate_model


with st.sidebar:
    st.title('AutoML')
    choice = st.radio('Navigation', ["Upload file", "Profiling", "ML models","Download Model"])
    st.info(''' This application allows you to automate the Machine Learning
            Model pipeline using streamlit, pandas profiling ''')

if os.path.exists('sourcefile.csv'):
    df = pd.read_csv('sourcefile.csv', index_col=None)
    
if choice == "Upload file":
    st.title("Upload your data for modeling")
    file = st.file_uploader('Upload your file here')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcefile.csv", index=None) #for profiling and modeling 
        st.dataframe(df)
        

if choice =="Profiling":
    st.title('Exploratory Data Analysis') 
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)
    
if choice == "ML models":
    st.title('ML Models comparision')
    data_type = st.selectbox('please select what type of learning you want to perform (classification or regression)', ['classification', 'regression'])
    target = st.selectbox('select your target column', df.columns)
    if st.button('predict'):
        if data_type == "regression":
            setup(df, target=target)
            setup_df = pull()
            st.info('This is the ML experiment settings')
            st.dataframe(setup_df)
            best_model = pycaret.regression.compare_models()
            compare_df = pull()
            st.info('This is the Model')
            st.dataframe(compare_df)
            best_model
            save_model(best_model, "best_model")
            
        elif data_type == "classification":
            setup(df, target=target)
            setup_df = pull()
            st.info('This is the ML experiment settings')
            st.dataframe(setup_df)
            best_model = pycaret.classification.compare_models()
            compare_df = pull()
            st.info('This is the Model comparision')
            st.dataframe(compare_df)
            best_model
            save_model(best_model, "best_model")
    
if choice == "Download Model":
    st.title('Download the Best model')
    with open("best_model.pkl", 'rb') as f:
            st.download_button('Download the model', f, 'trained_model')