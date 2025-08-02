# Delivey-Delay-Prediction
📦 Delivery Delay Prediction for E-commerce – End-to-End ML Project

🔍 Goal

Build a machine learning system that predicts whether an order will be delivered late or on time, using order metadata like estimated delivery date, product info, and seller/customer locations.

Late deliveries hurt customer satisfaction and company revenue. This project shows how machine learning can help e-commerce businesses flag potential delays early and act proactively.
🧩 Dataset Overview

Source: [Olist E-commerce Public Dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

We use multiple files:

    orders.csv – order timestamps (approved, delivered, estimated, etc.)

    order_items.csv – shipping and product info

    customers.csv – customer state

    sellers.csv – seller state

    products.csv – product categories

🛠 Project Pipeline
### ✅ Step 1: Define the Target

We defined a binary target:

delivery_delayed = 1 if actual_delivery_date > estimated_delivery_date else 0

### 🔄 Step 2: Data Merging

We joined all relevant CSVs to create a complete dataset with:

- Product category
    
- Seller and customer states
    
- Shipping delay and estimated delivery windows

### 🧠 Step 3: Feature Engineering

Created useful features such as:

- shipping_days = delivered_date - order_approved_at

- estimated_days = estimated_delivery_date - order_approved_at

- Encoded categorical features (product, customer state, seller state)

Final features used for modeling:

1. product_category_name

2. customer_state

3. seller_state

4. shipping_days

5. estimated_days

### 🤖 Step 4: Model Training

We trained and compared the following models:
1. Model	Accuracy	Precision	Recall	F1 Score
2. Logistic Regression	78.3%	74.1%	81.5%	77.6%
3. Random Forest Classifier	82.4%	79.0%	85.2%	81.9%
4. XGBoost Classifier	81.7%	78.5%	84.3%	81.3%
5. SVM (RBF kernel)	80.2%	76.9%	83.7%	80.2%

✅ Final model used: Random Forest Classifier

#### 🎯 Model Evaluation

Evaluation was done using:
- Confusion Matrix
- Classification Report
- Cross-validation
- Train-Test Split (80-20)

We stored the best-performing model pipeline (RandomForestClassifier + preprocessing) using joblib.

#### 🌐 Streamlit Web App

 We built a user-facing Streamlit app where users can:
 - Select product category, customer & seller state
 - Input shipping days & estimated days

    Click Predict to get result:
    #### ✅ Delivered On Time or ❌ Likely Delayed

To run:
```

streamlit run streamlit_app.py

```
#### 📁 Project Structure
```

delivery-delay-prediction/
├── app/
│   └── streamlit_app.py
├── notebooks/
│   ├── 1_data_merge.ipynb
│   ├── 2_edda.ipynb
│   └── 3_modeling.ipynb
├── data/
│   ├── data
│   │   ├── processed/final_merged_data.csv
│   │   └── raw/
│   │       ├── customers_dataset.csv
│   │       ├── geolocation_dataset.csv
│   │       ├── order_items_dataset.csv
│   │       ├── order_payments_dataset.csv
│   │       ├── order_reviews_dataset
│   │       ├── orders_dataset.csv
│   │       ├── product_category_name_translation.csv
│   │       ├── products_dataset.csv
│   │       └── sellers_dataset.csv
│   └── models/train_model
├── README.md
└── requirements.txt

```

#### 💡 Why This Project Matters

This is not a toy problem. Late deliveries:
- Cause negative customer reviews
- Increase customer service costs
- Reduce repeat purchases

Your ML model allows businesses to proactively identify at-risk orders and take action.

#### 🛠 Tools Used
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
- joblib for saving the model
- streamlit for deployment

#### 📈 Future Enhancements
- Add weather and traffic data
- Use deep learning for sequential patterns
- Visualize feature importance in the UI
- Add predict_proba() for confidence levels

