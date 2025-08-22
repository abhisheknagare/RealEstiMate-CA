import streamlit as st
import pandas as pd
import pickle
import requests

# load the trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# load the list of features that the model was trained on
with open("features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# helper function to turn an address into latitude/longitude
def geocode_address(address):
    api_key = st.secrets["POSITIONSTACK_API_KEY"]
    base_url = "http://api.positionstack.com/v1/forward"
    params = {"access_key": api_key, "query": address, "limit": 1}
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            location = data["data"][0]
            return location["latitude"], location["longitude"]
        else:
            st.error("No results found for this address.")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Problem contacting geocoding API: {e}")
        return None, None

# streamlit app starts here
st.title("California House Price Predictor")

# basic property address
address = st.text_input("Property Address", "1600 Amphitheatre Parkway, Mountain View, CA")

# core numeric property inputs
living_area = st.number_input("Living Area (sq ft)", min_value=200, max_value=10000, value=1500)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
garage_spaces = st.number_input("Garage Spaces", min_value=0, max_value=5, value=2)
lot_size_sqft = st.number_input("Lot Size (sq ft)", min_value=500, max_value=100000, value=5000)
age = st.number_input("Property Age (years)", min_value=0, max_value=200, value=10)

# yes/no features
view_yn = st.checkbox("View")
pool_yn = st.checkbox("Private Pool")
attached_garage_yn = st.checkbox("Attached Garage")
fireplace_yn = st.checkbox("Fireplace")
new_construction_yn = st.checkbox("New Construction")

# stories (choose one option, we will one-hot encode later)
stories_choice = st.radio("Stories", ["One", "Two", "ThreeOrMore", "MultiSplit"])

# flooring type (choose one option, we will one-hot encode later)
flooring_choice = st.selectbox(
    "Flooring",
    ["Bamboo", "Brick", "Carpet", "Concrete", "Laminate", "SeeRemarks",
     "Stone", "Tile", "Unknown", "Vinyl", "Wood"]
)

# school district (choose one option, we will one-hot encode later)
districts = [col.replace("District__", "") for col in feature_columns if col.startswith("District__")]
district_choice = st.selectbox("School District", districts)

# when user clicks predict
if st.button("Predict Price"):
    lat, lon = geocode_address(address)
    if lat is None or lon is None:
        st.stop()

    # build a dictionary with base numeric and yes/no inputs
    input_dict = {
        "Latitude": [lat],
        "Longitude": [lon],
        "LivingArea": [living_area],
        "BathroomsTotalInteger": [bathrooms],
        "BedroomsTotal": [bedrooms],
        "GarageSpaces": [garage_spaces],
        "LotSizeSquareFeet": [lot_size_sqft],
        "Age": [age],
        "ViewYN": [int(view_yn)],
        "PoolPrivateYN": [int(pool_yn)],
        "AttachedGarageYN": [int(attached_garage_yn)],
        "FireplaceYN": [int(fireplace_yn)],
        "NewConstructionYN": [int(new_construction_yn)],
    }

    df_input = pd.DataFrame(input_dict)

    # one-hot encode the stories choice
    for option in ["MultiSplit", "One", "Two", "ThreeOrMore"]:
        df_input[option] = 1 if stories_choice == option else 0

    # one-hot encode the flooring choice
    for option in ["Flooring_Bamboo", "Flooring_Brick", "Flooring_Carpet", "Flooring_Concrete",
                   "Flooring_Laminate", "Flooring_SeeRemarks", "Flooring_Stone", "Flooring_Tile",
                   "Flooring_Unknown", "Flooring_Vinyl", "Flooring_Wood"]:
        df_input[option] = 1 if option == f"Flooring_{flooring_choice}" else 0

    # one-hot encode the district choice
    for dist in [c for c in feature_columns if c.startswith("District__")]:
        df_input[dist] = 1 if dist == f"District__{district_choice}" else 0

    # make sure every column in training is present, if missing fill with zero
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    # reorder columns to match training order
    df_input = df_input[feature_columns]

    # run prediction
    try:
        prediction = model.predict(df_input)[0]
        st.success(f"Estimated Price: ${prediction:,.0f}")
    except Exception as e:
        st.error(f"Error while predicting: {e}")
