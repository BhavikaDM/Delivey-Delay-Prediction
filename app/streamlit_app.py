# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="Delivery Delay Predictor", layout="wide")

# ---------------------- LOAD MODEL ----------------------
model = joblib.load("C:/Users/bhavi/Desktop/delivery-delay-prediction/models/train_model.pkl")  # Full pipeline with scaler

# ---------------------- HEADER ----------------------
st.title("🚚 Delivery Delay Prediction App")
st.markdown("""
Use this app to predict whether an e-commerce order will be **delivered on time** or **delayed** based on order and product details.
""")

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("📄 About This Project")
st.sidebar.markdown("""
This project uses machine learning to predict delivery delays.

**Data Source**: Olist Brazilian E-commerce  
**Target**: `Is Order Delayed?`  
**Type**: Classification  

**Models Compared**:
- Logistic Regression
- Random Forest 
- XGBoost
- SVM
""")

st.sidebar.markdown("---")
st.sidebar.caption("Created by Bhavika Mandavkar")

# ---------------------- INPUT FORM ----------------------
st.subheader("🔍 Enter Order Details")

col1, col2 = st.columns(2)

with col1:
    shipping_days = st.number_input("Shipping Days (Estimated - Purchase)", min_value=0, max_value=30, value=5)
    freight = st.number_input("Freight Value (BRL)", min_value=0.0, value=20.0)
    product_weight = st.number_input("Product Weight (grams)", min_value=0.0, value=1000.0)
    

with col2:
    product_volume = st.number_input("Product Volume (cm³)", min_value=0.0, value=1000.0)
    seller_score = st.slider("Seller Score (0-5)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    num_items = st.number_input("Number of Items in Order", min_value=1, max_value=10, value=1)


st.markdown("### 📍 Location & Category Details")

# Get options from training data or define manually
states = ['SP', 'RJ', 'MG', 'BA', 'PR', 'SC', 'RS', 'GO', 'PE', 'ES', 'PA', 'DF']
categories = ['bed_bath_table', 'health_beauty', 'sports_leisure', 'computers_accessories',
              'furniture_decor', 'watches_gifts', 'telephony', 'housewares', 'auto']

col3, col4 = st.columns(2)
with col3:
    customer_state = st.selectbox("Customer State", options=states, index=0)
    seller_state = st.selectbox("Seller State", options=states, index=0)
with col4:
    product_category_name = st.selectbox("Product Category", options=categories, index=0)
    estimated_days = st.number_input("Estimated Delivery Days", min_value=1, max_value=30, value=5)

# ---------------------- PREDICTION ----------------------
if st.button("🚀 Predict Delivery Status"):
    input_df = pd.DataFrame({
        'shipping_days': [shipping_days],
        'freight_value': [freight],
        'product_weight_g': [product_weight],
        'product_volume_cm3': [product_volume],
        'seller_score': [seller_score],
        'num_items': [num_items],
        'estimated_days': [estimated_days],
        'seller_state': [seller_state],
        'product_category_name': [product_category_name],
        'customer_state': [customer_state],
    })

    prediction = model.predict(input_df)[0]
    prediction_label = "🟢 On-Time Delivery" if prediction == 0 else "🔴 Delayed Delivery"

    st.success(f"**Prediction Result:** {prediction_label}")

# ---------------------- MODEL PERFORMANCE ----------------------
st.markdown("---")
st.subheader("📊 Model Performance Comparison (F1 Score)")

model_names = ["Logistic Regression", "Random Forest", "XGBoost", "SVM"]
f1_scores = [0.73, 0.88, 0.90, 0.76]  # Update with actual scores if different

fig, ax = plt.subplots()
ax.barh(model_names, f1_scores, color="teal")
ax.set_xlim(0, 1)
ax.set_xlabel("F1 Score")
ax.set_title("Model Performance")
st.pyplot(fig)

# ---------------------- EXPLAINER ----------------------
with st.expander("🧠 How It Works"):
    st.markdown("""
    - We trained classification models to predict whether an order will be delayed.
    - Features used include:
        - Shipping time difference (estimated - purchase)
        - Freight charges
        - Product dimensions (weight, volume)
        - Seller performance
        - Number of items in an order
    - Random Forest performed the best on evaluation metrics and is used here.
    """)

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption("📌 Note: This is a machine learning demo and not intended for production use.")
