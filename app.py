import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.joblib")

# Page title
st.title("🏠 Indian Cities Housing Price Predictor")

st.markdown("This app predicts property prices based on features like location, size, BHK, and amenities.")

# --- Input Fields ---
city = st.selectbox("City", ['Mumbai', 'Pune', 'Hyderabad', 'Bangalore', 'Ahmedabad', 'Chennai', 'Delhi', 'Kolkata'])
locality = st.text_input("Locality Name", placeholder="Eg: Whitefield")
bhk = st.selectbox("Number of BHK", [1, 2, 3, 4, 5])
size = st.number_input("Size (in sq ft)", min_value=100, max_value=20000, step=10)

property_type = st.selectbox("Property Type", ['Apartment', 'Villa', 'Independent House'])
furnishing = st.selectbox("Furnishing", ['Furnished', 'Semi-Furnished', 'Unfurnished'])

is_apartment = st.checkbox("Is Apartment?", value=True)
is_ready = st.checkbox("Ready to Move?", value=True)
is_rera = st.checkbox("RERA Registered?", value=True)

# --- Predict Button ---
if st.button("Predict Price 💸"):
    input_df = pd.DataFrame([{
        "City_name": city,
        "Locality_Name": locality,
        "No_of_BHK": bhk,
        "Size": size,
        "Property_type": property_type,
        "is_furnished": furnishing,
        "is_Apartment": bool(is_apartment),
        "is_ready_to_move": bool(is_ready),
        "is_RERA_registered": bool(is_rera)
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🏷️ Estimated Price: ₹{int(prediction):,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
