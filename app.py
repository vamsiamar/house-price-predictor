import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.joblib")

st.set_page_config(page_title="🏠 House Price Predictor - Indian Cities", layout="centered")

st.title("🏠 Indian Cities House Price Predictor")
st.markdown("🔍 Enter property details to estimate its market value.")

# Input form
with st.form("predict_form"):
    city = st.selectbox("Select City", ["Ahmedabad", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"])
    locality = st.text_input("Enter Locality (e.g., Bopal, Whitefield)")
    bhk = st.selectbox("Number of Bedrooms (BHK)", [1, 2, 3, 4, 5])
    size = st.number_input("Size (in sq ft)", min_value=100, max_value=10000, value=1000, step=50)
    is_furnished = st.selectbox("Is the property furnished?", ["Yes", "No"])
    property_type = st.selectbox("Property Type", ["Residential Apartment", "Independent House", "Villa", "Builder Floor"])
    is_apartment = st.selectbox("Is it an apartment?", ["Yes", "No"])
    is_ready = st.selectbox("Is it ready to move?", ["Yes", "No"])
    is_rera = st.selectbox("Is it RERA registered?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Convert to model input format
    input_df = pd.DataFrame({
        "City_name": [city],
        "Locality_Name": [locality],
        "No_of_BHK": [bhk],
        "Size": [size],
        "is_furnished": [is_furnished],
        "Property_type": [property_type],
        "is_Apartment": [True if is_apartment == "Yes" else False],
        "is_ready_to_move": [True if is_ready == "Yes" else False],
        "is_RERA_registered": [True if is_rera == "Yes" else False]
    })

    # Show input for debugging
    st.write("🔎 Input Provided:", input_df)

    try:
        # Predict price
        predicted_price = model.predict(input_df)[0]

        # Prevent negative predictions
        predicted_price = max(0, predicted_price)

        # Show result
        st.success(f"🏷️ Estimated Property Price: ₹{int(predicted_price):,}")
    except Exception as e:
        st.error(f"🚨 Prediction Failed: {e}")
