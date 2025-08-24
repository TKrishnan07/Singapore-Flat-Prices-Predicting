import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page title and layout

st.set_page_config(page_title="Singapore Resale Flat Price Prediction", layout="wide")
st.title("Singapore Resale Flat Price Prediction")

st.sidebar.image("pngtree-singapore.jpg", width=200)

# Sidebar with title and menu
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Data", "EDA", "Prediction", "Model Performance"])

@st.cache_data
def load_data():
    df = pd.read_csv('ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv')
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    df['year'] = df['month'].dt.year
    df['town'] = df['town'].astype('category')
    df['flat_type'] = df['flat_type'].astype('category')
    df['flat_model'] = df['flat_model'].astype('category')
    df['town_code'] = df['town'].cat.codes
    df['flat_type_code'] = df['flat_type'].cat.codes
    df['flat_model_code'] = df['flat_model'].cat.codes

    def extract_midpoint(storey_range):
        parts = storey_range.split(' TO ')
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) / 2
        else:
            return int(parts[0])
    df['storey_range_midpoint'] = df['storey_range'].apply(extract_midpoint)
    return df

@st.cache_resource
def load_model_and_maps():
    with open('flat_price_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('category_maps.pkl', 'rb') as f:
        category_maps = pickle.load(f)
    return model, category_maps

df = load_data()
model, category_maps = load_model_and_maps()

features = ['town_code', 'flat_type_code', 'floor_area_sqm',
            'storey_range_midpoint', 'lease_commence_date', 'year']

if selection == "Data":
    st.header("Dataset Sample")
    st.dataframe(df.head(20))

elif selection == "EDA":
    st.header("Exploratory Data Analysis")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['resale_price'], bins=30, kde=True, ax=ax1)
    ax1.set_title('Distribution of Resale Prices')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.barplot(x='town', y='resale_price', data=df, estimator=np.mean, ax=ax2)
    ax2.set_title('Mean Resale Price by Town')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    st.pyplot(fig2)

    corr = df[['floor_area_sqm', 'lease_commence_date', 'year', 'resale_price']].corr()
    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='Blues', ax=ax3)
    ax3.set_title('Correlation Heatmap')
    st.pyplot(fig3)

elif selection == "Prediction":
    st.header("Predict Resale Price")

    towns = list(category_maps['town'].values())
    flat_types = list(category_maps['flat_type'].values())
    flat_models = list(category_maps['flat_model'].values())

    town = st.selectbox("Select Town:", towns)
    flat_type = st.selectbox("Select Flat Type:", flat_types)
    flat_model = st.selectbox("Select Flat Model:", flat_models)
    floor_area_sqm = st.number_input("Floor Area (sqm):", min_value=20, max_value=200, value=90)
    storey_range = st.text_input('Storey Range (e.g. "7 TO 9" or "4"):', value="7 TO 9")
    lease_commence_date = st.number_input("Lease Commencement Year:", min_value=1960, max_value=2025, value=2000)
    year = st.number_input("Transaction Year:", min_value=2017, max_value=2025, value=2025)

    if st.button("Predict"):
        try:
            town_code = towns.index(town)
            flat_type_code = flat_types.index(flat_type)
            flat_model_code = flat_models.index(flat_model)

            if 'TO' in storey_range.upper():
                parts = storey_range.upper().split(' TO ')
                storey_mid = (int(parts[0]) + int(parts[1])) / 2
            else:
                storey_mid = int(storey_range)

            input_df = pd.DataFrame([{
                'town_code': town_code,
                'flat_type_code': flat_type_code,
                'floor_area_sqm': floor_area_sqm,
                'storey_range_midpoint': storey_mid,
                'lease_commence_date': lease_commence_date,
                'year': year
            }])
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Resale Price: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

elif selection == "Model Performance":
    st.header("Model Evaluation Metrics")

    X = df[features]
    y = df['resale_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    y_pred = model.predict(X_test)

    st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
