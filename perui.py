import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
from src.encrypt import decrypt_data
from src.auth_helpers import get_access_token
from utils.chat_model import AIGatewayLangchainChatOpenAI
import streamlit as st

# Load configuration from JSON and environment variables
with open('secret/url.json', 'r') as file:
    url_data = json.load(file)

AI_GATEWAY_BASE_URL = url_data.get("ai_gateway")
issuer_url = url_data.get("issuer_url")

# Load the .env file
load_dotenv('secret/.env')

# Retrieve and decrypt the environment variables
loaded_fernet_key = os.getenv('FERNET_KEY').encode()
loaded_encrypted_client_id = os.getenv('ENCRYPTED_CLIENT_ID').encode()
loaded_encrypted_client_secret = os.getenv('ENCRYPTED_CLIENT_SECRET').encode()

client_id = decrypt_data(loaded_encrypted_client_id, loaded_fernet_key)
client_secret = decrypt_data(loaded_encrypted_client_secret, loaded_fernet_key)

# Get access token
access_token = get_access_token(client_id, client_secret, issuer_url)

# Initialize the AI model
model = AIGatewayLangchainChatOpenAI(
    access_token=access_token, 
    base_url=AI_GATEWAY_BASE_URL, 
    model="o1-2024-12-17", 
    deere_ai_gateway_registration_id="graphics-quality-check"
)

# Load dataset from CSV
df = pd.read_csv(r'C:\Users\W4FGXUV\Downloads\My\aw_fb_data.csv\aw_fb_data.csv')  # Replace with the path to your CSV file

# Preprocessing
X = df[['age', 'gender', 'height', 'weight', 'steps', 'activity']]
y = df['calories']

# One-hot encode categorical variables (activity)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['activity'])
    ],
    remainder='passthrough'  # Keep the rest of the columns
)

# Create a pipeline for the regression model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train the model
pipeline.fit(X, y)

# Function to generate health recommendations using the LLM
def get_health_recommendation(user_data):
    prompt = (
        f"Given the following user data:\n"
        f"Age: {user_data['age']}\n"
        f"Gender: {'Male' if user_data['gender'] == 1 else 'Female'}\n"
        f"Height: {user_data['height']} cm\n"
        f"Weight: {user_data['weight']} kg\n"
        f"Steps taken today: {user_data['steps']}\n"
        f"Activity type: {user_data['activity']}\n\n"
        f"Based on this information, what recommendations can you provide for maintaining or improving their health?"
    )
    
    response = model.invoke([{"role": "user", "content": prompt}])
    return response.content

# Streamlit UI
st.title("Health Prediction and Recommendation System")

# User input section
st.header("Enter Your Health Information")

age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", options=["Male", "Female"])
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
steps = st.number_input("Steps Taken Today", min_value=0, value=1000)
activity = st.selectbox("Activity Type", options=["Lying", "Sitting", "Self Pace walk", "Running 3 METs"])

# Button to generate recommendation
if st.button("Get Recommendation"):
    # Prepare user data for prediction
    user_data = {
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'height': height,
        'weight': weight,
        'steps': steps,
        'activity': activity
    }

    # Get health recommendation
    recommendation = get_health_recommendation(user_data)
    
    # Display the recommendation
    st.subheader("Health Recommendation:")
    st.write(recommendation)
