import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Define the path to the saved model
MODEL_PATH = 'tourism_project/model_building/best_model.pkl'

# Load the best performing model
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title('Travel Package Purchase Prediction')
st.write('Enter customer details to predict if they will purchase the Wellness Tourism Package.')

# --- Define expected columns for one-hot encoding consistency ---
# This list MUST exactly match X.columns after get_dummies during training
# In a robust MLOps pipeline, these columns would be saved alongside the model.
X_columns = [
    'Age',
    'DurationOfPitch',
    'NumberOfFollowups',
    'NumberOfTrips',
    'MonthlyIncome',
    'TypeofContact_Self Enquiry',
    'CityTier_2',
    'CityTier_3',
    'Occupation_Free Lancer',
    'Occupation_Large Business',
    'Occupation_Salaried',
    'Occupation_Senior Manager',
    'Occupation_Small Business',
    'Occupation_VP',
    'Gender_Male',
    'NumberOfPersonVisited_2',
    'NumberOfPersonVisited_3',
    'NumberOfPersonVisited_4',
    'NumberOfPersonVisited_5',
    'ProductPitched_Deluxe',
    'ProductPitched_King',
    'ProductPitched_Standard',
    'ProductPitched_Super Deluxe',
    'PreferredPropertyStar_4.0',
    'PreferredPropertyStar_5.0',
    'MaritalStatus_Married',
    'MaritalStatus_Single',
    'MaritalStatus_Unmarried',
    'Passport_1',
    'PitchSatisfactionScore_2',
    'PitchSatisfactionScore_3',
    'PitchSatisfactionScore_4',
    'PitchSatisfactionScore_5',
    'OwnCar_1',
    'NumberOfChildrenVisited_1.0',
    'NumberOfChildrenVisited_2.0',
    'NumberOfChildrenVisited_3.0',
    'Designation_Executive',
    'Designation_Manager',
    'Designation_Senior Manager',
    'Designation_VP'
]

# Categorical options as used during preprocessing (hardcoded for standalone app)
# Ensure these lists contain all possible categories from training data for consistent one-hot encoding
categorical_options = {
    'TypeofContact': ['Company Invited', 'Self Enquiry'],
    'CityTier': [1, 2, 3],
    'Occupation': ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'],
    'Gender': ['Female', 'Male'],
    'NumberOfPersonVisited': [1, 2, 3, 4, 5],
    'ProductPitched': ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'],
    'PreferredPropertyStar': [3.0, 4.0, 5.0],
    'MaritalStatus': ['Married', 'Divorced', 'Single', 'Unmarried'],
    'Passport': [0, 1],
    'PitchSatisfactionScore': [1, 2, 3, 4, 5],
    'OwnCar': [0, 1],
    'NumberOfChildrenVisited': [0.0, 1.0, 2.0, 3.0],
    'Designation': ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']
}

# --- User Input Form ---
with st.form('prediction_form'):
    st.header('Customer Details')

    # Numerical Inputs
    age = st.number_input('Age', min_value=18, max_value=100, value=35)
    duration_of_pitch = st.number_input('Duration of Pitch (minutes)', min_value=5, max_value=150, value=15)
    number_of_followups = st.number_input('Number of Follow-ups', min_value=1, max_value=10, value=3)
    number_of_trips = st.number_input('Number of Trips Annually', min_value=1, max_value=50, value=3)
    monthly_income = st.number_input('Monthly Income (USD)', min_value=1000, max_value=100000, value=25000)

    # Categorical Inputs (using sorted options for consistent display)
    type_of_contact = st.selectbox('Type of Contact', options=sorted(categorical_options['TypeofContact']))
    city_tier = st.selectbox('City Tier', options=sorted(categorical_options['CityTier']))
    occupation = st.selectbox('Occupation', options=sorted(categorical_options['Occupation']))
    gender = st.selectbox('Gender', options=sorted(categorical_options['Gender']))
    number_of_person_visited = st.selectbox('Number of Persons Visiting', options=sorted(categorical_options['NumberOfPersonVisited']))
    product_pitched = st.selectbox('Product Pitched', options=sorted(categorical_options['ProductPitched']))
    preferred_property_star = st.selectbox('Preferred Property Star', options=sorted(categorical_options['PreferredPropertyStar']))
    marital_status = st.selectbox('Marital Status', options=sorted(categorical_options['MaritalStatus']))
    passport = st.selectbox('Passport', options=sorted(categorical_options['Passport']))
    pitch_satisfaction_score = st.selectbox('Pitch Satisfaction Score', options=sorted(categorical_options['PitchSatisfactionScore']))
    own_car = st.selectbox('Own Car', options=sorted(categorical_options['OwnCar']))
    number_of_children_visited = st.selectbox('Number of Children Visiting', options=sorted(categorical_options['NumberOfChildrenVisited']))
    designation = st.selectbox('Designation', options=sorted(categorical_options['Designation']))

    submitted = st.form_submit_button('Predict Purchase')

    if submitted:
        # Collect input data into a dictionary
        input_data = {
            'Age': age,
            'DurationOfPitch': duration_of_pitch,
            'NumberOfFollowups': number_of_followups,
            'NumberOfTrips': number_of_trips,
            'MonthlyIncome': monthly_income,
            'TypeofContact': type_of_contact,
            'CityTier': city_tier,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisited': number_of_person_visited,
            'ProductPitched': product_pitched,
            'PreferredPropertyStar': preferred_property_star,
            'MaritalStatus': marital_status,
            'Passport': passport,
            'PitchSatisfactionScore': pitch_satisfaction_score,
            'OwnCar': own_car,
            'NumberOfChildrenVisited': number_of_children_visited,
            'Designation': designation
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocessing: Ensure column types are correct before one-hot encoding
        # This is important for `get_dummies` to produce consistent columns
        for col, options in categorical_options.items():
            if col in input_df.columns:
                input_df[col] = pd.Categorical(input_df[col], categories=options)

        # Apply one-hot encoding, matching `drop_first=True`
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Reindex to ensure all columns from training are present and in the correct order
        # Fill missing columns (if a category was not present in input) with 0
        input_processed = input_encoded.reindex(columns=X_columns, fill_value=0)

        # Ensure all columns expected by the model are present and in order
        if len(input_processed.columns) != len(X_columns):
            st.error("Mismatch in feature columns after preprocessing. This is an internal error.")
        else:
            # Make prediction
            prediction = model.predict(input_processed)
            prediction_proba = model.predict_proba(input_processed)[:, 1] # Probability of positive class

            if prediction[0] == 1:
                st.success(f'Prediction: Customer is LIKELY to purchase the package! (Probability: {prediction_proba[0]:.2f})')
            else:
                st.info(f'Prediction: Customer is UNLIKELY to purchase the package. (Probability: {prediction_proba[0]:.2f})')
