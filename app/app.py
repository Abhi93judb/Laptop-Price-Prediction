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

st.sidebar.header("💻 Laptop Specifications")

st.sidebar.markdown("""
Select the laptop configuration below and click **Predict Laptop Price**.
""")

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
if st.sidebar.button("🔮 Predict Laptop Price", width="stretch"):

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
    
    with st.spinner("Predicting laptop price..."):
       prediction = model.predict(input_df)

    st.divider()

    st.subheader("💰 Prediction Result")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.metric(
            label="Estimated Laptop Price",
            value=f"₹ {prediction[0]:,.0f}"
        )

    with col2:
        if prediction[0] < 40000:
            st.success("💵 Budget")
        elif prediction[0] < 80000:
            st.warning("⚡ Mid-Range")
        else:
            st.error("🔥 Premium")

    st.divider()

    st.subheader("📋 Selected Laptop Configuration")

    config = pd.DataFrame({
        "Specification": [
            "Brand",
            "Processor",
            "RAM",
            "Storage",
            "Operating System",
            "GPU",
            "Display",
            "Specification Rating",
            "Warranty"
        ],
        "Selected Value": [
            brand,
            processor,
            f"{ram} GB",
            f"{rom} GB",
            os,
            gpu,
            display_size,
            spec_rating,
            f"{warranty} Year(s)"
        ]
    })
    
    config["Selected Value"] = config["Selected Value"].astype(str)

    st.dataframe(config, width="stretch", hide_index=True)

    st.success("✅ Prediction Completed Successfully!")