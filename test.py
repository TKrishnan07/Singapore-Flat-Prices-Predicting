import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
from PIL import Image

# Load dataset
df = pd.read_csv('ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv')

# Data Preprocessing
df['month'] = pd.to_datetime(df['month'])
df['year'] = df['month'].dt.year

# Convert relevant columns to categorical data
df['town'] = df['town'].astype('category')
df['flat_type'] = df['flat_type'].astype('category')
df['flat_model'] = df['flat_model'].astype('category')

# Encode categorical data to numeric
df['town_code'] = df['town'].cat.codes
df['flat_type_code'] = df['flat_type'].cat.codes
df['flat_model_code'] = df['flat_model'].cat.codes

# Function to extract the midpoint from storey_range
def extract_midpoint(storey_range):
    # Split the string by 'TO' to get the range
    parts = storey_range.split(' TO ')
    
    # If the storey range contains two parts, calculate the midpoint
    if len(parts) == 2:
        low, high = int(parts[0]), int(parts[1])
        return (low + high) / 2
    else:
        # If the range does not have 'TO', return the value as integer
        return int(parts[0])

# Apply the function to the storey_range column
df['storey_range_midpoint'] = df['storey_range'].apply(extract_midpoint)

# Feature selection with the new 'storey_range_midpoint' column
features = ['town_code', 'flat_type_code', 'floor_area_sqm', 'storey_range_midpoint', 'lease_commence_date', 'year']
X = df[features]
y = df['resale_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit application for prediction

def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://img.veenaworld.com/wp-content/uploads/2022/04/julien-de-salaberry-viwdmfrbXfI-unsplash.jpg");
            background-size: cover;
            background-position: left bottom;
            background-repeat: no-repeat;
            background-color: #cccccc;
             
        }
        .title {
            color: white;
            font-weight: bold;
            font-size: 2.5em;
        }
        .header {
            color: white;   
            font-weight: bold;
            font-size: 1em; 
        }
        .result {
            color: yellow;
            font-weight: bold;
            font-size: 1.2em;
            margin-top: 10px;
        }
        .label {
            color: white;
            font-weight: bold;
            font-size: 1.1em;
            margin-top: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()
# The rest of your Streamlit app
st.markdown('<div class="title">Singapore Resale Flat Price Prediction</div>', unsafe_allow_html=True)

# Input from user
st.markdown('<div class="header">Select Town:</div>', unsafe_allow_html=True)
town = st.selectbox("", df['town'].unique())

st.markdown('<div class="header">Select Flat Type:</div>', unsafe_allow_html=True)
flat_type = st.selectbox("", df['flat_type'].unique())

st.markdown('<div class="header">Select Flat Model:</div>', unsafe_allow_html=True)
flat_model = st.selectbox("", df['flat_model'].unique())

st.markdown('<div class="header">Enter Floor Area (sqm):</div>', unsafe_allow_html=True)
floor_area_sqm = st.number_input("", min_value=20, max_value=200)

st.markdown('<div class="header">Select Storey Range:</div>', unsafe_allow_html=True)
storey_range = st.selectbox("", df['storey_range'].unique())

st.markdown('<div class="header">Enter Lease Commence Year:</div>', unsafe_allow_html=True)
lease_commence_date = st.number_input("", min_value=1960, max_value=2023)


# Extract the midpoint from storey_range selected by the user
def extract_midpoint(storey_range):
    parts = storey_range.split(' TO ')
    if len(parts) == 2:
        low, high = int(parts[0]), int(parts[1])
        return (low + high) / 2
    else:
        return int(parts[0])

storey_range_midpoint = extract_midpoint(storey_range)

# Convert user inputs to match model input
town_code = df['town'].cat.categories.tolist().index(town)
flat_type_code = df['flat_type'].cat.categories.tolist().index(flat_type)
flat_model_code = df['flat_model'].cat.categories.tolist().index(flat_model)

# Predict button
if st.button("Predict Resale Price"):
    input_data = pd.DataFrame({
        'town_code': [town_code],
        'flat_type_code': [flat_type_code],
        'floor_area_sqm': [floor_area_sqm],
        'storey_range_midpoint': [storey_range_midpoint],
        'lease_commence_date': [lease_commence_date],
        'year': [2023]  # Assuming current year
    })
    
    # Predict resale price using the trained model
    predicted_price = model.predict(input_data)
   
    
   # Display predicted price
    st.markdown(f'<div class="result">Predicted Resale Price: ${predicted_price[0]:,.2f}</div>', unsafe_allow_html=True)

# Display first few rows of the dataset
st.markdown('<div class="label">Sample Data:</div>', unsafe_allow_html=True)
st.dataframe(df.head())  # Using `st.dataframe` to keep the table style consistent

# Model evaluation
y_pred = model.predict(X_test)
st.markdown('<div class="label">Model Evaluation:</div>', unsafe_allow_html=True)
st.markdown(f'<div class="result">Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="result">Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="result">R2 Score: {r2_score(y_test, y_pred):.2f}</div>', unsafe_allow_html=True)
