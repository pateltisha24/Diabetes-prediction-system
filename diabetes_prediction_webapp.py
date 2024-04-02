# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 07:45:19 2024

@author: patel
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

loaded_model=pickle.load(open("C:/Projects/diabetes-prediction-system/diabetes_prediction_system",'rb'))

def predict(input_data):
    numpy_input_data=np.asarray(input_data)
    #reshape for predict one data
    input_data_reshaped=numpy_input_data.reshape(1,-1)
    scaler=StandardScaler()
    std_data=scaler.fit_transform(input_data_reshaped)
    prediction=loaded_model.predict(std_data)[0]
    print(prediction)
    
    if prediction == 0:
        print('Not diabetic')
        return 'the person is not diabetic'
    elif prediction == 1:
        print('diabetic')
        return 'the person is diabetic'
    
def main():
    
    st.title('Diabetes Prediction System')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure Level')
    SkinThickness=(st.text_input('Skin Thickness Level'))
    Insulin=(st.text_input('Insulin Level'))
    BMI=(st.text_input('BMI'))
    DiabetesPedigreeFunction=(st.text_input('Diabetes Pedigree Function value'))
    Age=(st.text_input('Age'))
       
    diagnosis = " "
    if st.button("Diabetes Test Result"):
        diagnosis=predict([Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,  Age])

    st.success(diagnosis)
        
if __name__ == '__main__':
    main()