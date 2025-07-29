import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load Dataset
data = pd.read_excel('product_sales_data.xlsx')

# Train Model
X = data[['Price', 'Stock', 'Discount']]
y = data['Sales']

model = LinearRegression()
model.fit(X, y)

# Custom CSS Styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 2.5rem;
            color: #333333;
            text-align: center;
            padding: 20px 0;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-size: 1rem;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .result {
            background-color: #e6f2ff;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            color: #004085;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="title">📈 Product Sales Predictor</div>', unsafe_allow_html=True)

price = st.number_input("Enter Product Price:", min_value=0)
stock = st.number_input("Enter Stock Quantity:", min_value=0)
discount = st.number_input("Enter Discount Percentage:", min_value=0, max_value=100)

if st.button("Predict Sales"):
    input_data = np.array([[price, stock, discount]])
    prediction = model.predict(input_data)
    st.markdown(f'<div class="result">Predicted Sales: <strong>{int(prediction[0])} units</strong></div>', unsafe_allow_html=True)
