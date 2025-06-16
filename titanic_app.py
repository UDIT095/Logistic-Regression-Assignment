import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('titanic_model.pkl')

# Streamlit app
st.title('Titanic Survival Prediction')

# User inputs
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('SibSp', min_value=0, max_value=10, value=0)
parch = st.number_input('Parch', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, value=20.0)
cabin = st.selectbox('Cabin', ['C', 'E', 'G', 'D', 'A', 'B', 'F', 'T', 'Unknown'])
embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])

# Preprocess user inputs
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Cabin': [cabin],
    'Embarked': [embarked],
    'FamilySize': [sibsp + parch + 1]
})

# Encode categorical features
input_data['Sex'] = input_data['Sex'].map({'male': 1, 'female': 0})
input_data['Embarked'] = input_data['Embarked'].map({'S': 2, 'C': 0, 'Q': 1})
input_data['Cabin'] = input_data['Cabin'].map({'C': 2, 'E': 4, 'G': 6, 'D': 3, 'A': 0, 'B': 1, 'F': 5, 'T': 7, 'Unknown': 8})

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction[0] == 1:
        st.write(f'Predicted Survival: Yes (Probability: {probability:.2f})')
    else:
        st.write(f'Predicted Survival: No (Probability: {1 - probability:.2f})')