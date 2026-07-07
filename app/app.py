import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

raw_df = joblib.load("models/raw_dataset.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.title("💻 Laptop Price Prediction")

st.markdown("""
### 🤖 Machine Learning Based Laptop Price Prediction

Enter the laptop specifications from the left sidebar and click **Predict Price** to estimate the laptop price using a trained Machine Learning model.

---
""")

model = joblib.load("models/laptop_price_prediction_model.pkl")
st.success("✅ Model Loaded Successfully!")

st.sidebar.header("💻 Laptop Specifications")

brand = st.sidebar.selectbox("Brand", sorted(raw_df["brand"].unique()))

processor = st.sidebar.selectbox("Processor", sorted(raw_df["processor"].unique()))

ram = st.sidebar.selectbox("RAM", sorted(raw_df["Ram"].unique()))

rom = st.sidebar.selectbox("Storage", sorted(raw_df["ROM"].unique()))

os = st.sidebar.selectbox("Operating System", sorted(raw_df["OS"].unique()))

gpu = st.sidebar.selectbox("GPU", sorted(raw_df["GPU"].unique()))

display_size = st.sidebar.selectbox(
    "Display Size",
    sorted(raw_df["display_size"].unique())
)

spec_rating = st.sidebar.slider(
    "Specification Rating",
    int(raw_df["spec_rating"].min()),
    int(raw_df["spec_rating"].max()),
    int(raw_df["spec_rating"].mean())
)

warranty = st.sidebar.selectbox(
    "Warranty (Years)",
    sorted(raw_df["warranty"].unique())
)
if st.sidebar.button("Predict Price"):

    input_data = {
        "spec_rating": spec_rating,
        "display_size": display_size,
        "resolution_width": 1920,
        "resolution_height": 1080,
        "warranty": warranty,
        "brand": brand,
        "processor": processor,
        "Ram": ram,
        "ROM": rom,
        "GPU": gpu,
        "OS": os
    }

    
    input_df = pd.DataFrame([input_data])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_df)

    st.success(f"💰 Estimated Laptop Price: ₹{prediction[0]:,.0f}")
    
    if prediction[0] < 40000:
        st.info("💵 Category: Budget Laptop")

    elif prediction[0] < 80000:
        st.info("⚡ Category: Mid-Range Laptop")

    else:
        st.info("🔥 Category: Premium Laptop")