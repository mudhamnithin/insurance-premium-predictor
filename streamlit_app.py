import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("xgb_model.pkl")
input_columns = joblib.load("input_columns.pkl")
df = pd.read_excel("cleaned_premiums.xlsx")

st.title("🏥 Health Insurance Premium Predictor")

user_input = {}

# Take inputs
for col in input_columns:
    try:
        if df[col].dtype == object:
            user_input[col] = st.selectbox(f"Select {col}", sorted(df[col].dropna().unique().tolist()))
        elif pd.api.types.is_numeric_dtype(df[col]):
            user_input[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))
        else:
            user_input[col] = st.text_input(f"Enter {col}")
    except KeyError:
        st.warning(f"⚠️ Column missing: {col}")

# Predict
if st.button("Predict Premium"):
    try:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Premium: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
