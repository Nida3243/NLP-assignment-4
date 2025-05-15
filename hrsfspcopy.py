import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
from src.encrypt import decrypt_data
from src.auth_helpers import get_access_token
from utils.chat_model import AIGatewayLangchainChatOpenAI
import streamlit as st
from io import StringIO

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
    model="gpt-4o-2024-05-13",
    deere_ai_gateway_registration_id="graphics-quality-check"
)

# Load dataset from CSV
df = pd.read_csv(r'C:\Users\W4FGXUV\Downloads\My\aw_fb_data.csv\aw_fb_data.csv')  # Replace with the path to your CSV file

# Preprocessing: Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'X1'])

# Define features and target variable
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

# Function to generate personalized health recommendations based on user data
def get_health_recommendations(user_data):
    age = user_data['age']
    
    if age <= 30:
        return [
            "🏃‍♂️ **Increase daily steps to 10,000.**",
            "🏋️‍♀️ **Incorporate strength training workouts twice a week.**",
            "💧 **Stay hydrated; drink at least 2.5 liters of water daily.**",
            "😴 **Aim for 7-9 hours of quality sleep.**",
            "🥗 **Focus on a balanced diet rich in whole grains, lean proteins, fruits, and vegetables.**"
        ]
    elif 31 <= age <= 60:
        return [
            "🚶‍♂️ **Gradually increase step count to at least 8,000 steps daily.**",
            "🚴‍♂️ **Include 150 minutes of brisk walking or cycling weekly.**",
            "💧 **Stay hydrated; drink at least 3 liters of water daily.**",
            "😴 **Aim for 7-8 hours of sleep; establish a calming bedtime routine.**",
            "🥗 **Focus on a diet with vegetables, whole grains, and healthy fats.**"
        ]
    else:  # age > 60
        return [
            "👟 **Aim for at least 5,000 steps a day.**",
            "🧘‍♀️ **Consider gentle exercises like tai chi or water aerobics.**",
            "💧 **Ensure adequate hydration; drink at least 2 liters of water daily.**",
            "😴 **Aim for 7-8 hours of sleep; maintain a consistent sleep schedule.**",
            "🥗 **Focus on a nutrient-dense diet with lean proteins, fruits, and vegetables.**"
        ]

# Streamlit UI
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f8ff;  /* Light background color */
    }
    h1 {
        color: #0072B5;
        text-align: center;
    }
    h2 {
        color: #005B8C;
    }
    .sidebar .sidebar-content {
        background-color: #E6F7FF;
        padding: 10px;
    }
    .metric {
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Personalized Health Prediction and Recommendation System")
st.markdown("""
    This application helps you predict your caloric burn based on your health metrics and physical activity. 
    Please select a user profile from the dataset or enter your information in the sidebar to receive personalized health interventions.
""")

# Sidebar for user input
st.sidebar.header("Select or Enter User Information")

# User selection from dataset
user_selection = st.sidebar.selectbox("Select User Profile", df.index)
if user_selection is not None:
    selected_user = df.loc[user_selection]
    age = selected_user['age']
    gender = 'Male' if selected_user['gender'] == 1 else 'Female'
    height = selected_user['height']
    weight = selected_user['weight']
    steps = selected_user['steps']
    hear_rate = selected_user['hear_rate']
    activity = selected_user['activity']
else:
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=250, value=70) 


    steps = st.sidebar.number_input("Steps Taken Today", min_value=0, value=1000)
    hear_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=30, value=70)
    activity = st.sidebar.selectbox("Activity Type", options=["Lying", "Sitting", "Self Pace walk", "Running 3 METs"])

# Prepare user data for prediction
user_data = {
    'age': age,
    'gender': 1 if gender == "Male" else 0,
    'height': height,
    'weight': weight,
    'steps': steps,
    'hear_rate': hear_rate,
    'activity': activity
}

# Button to generate recommendation
if st.sidebar.button("Get Recommendation"):
    # Get health recommendations
    recommendations = get_health_recommendations(user_data)
    
    # Display predicted calories burned
    predicted_calories = pipeline.predict(pd.DataFrame([user_data]))[0]
    st.markdown(f"### Predicted Calories Burned: **{predicted_calories:.2f} kcal**")
    
    # Display health recommendations
    st.subheader("Personalized Health Recommendations:")
    for item in recommendations:
        st.markdown(item)

    # Visualization of user metrics
    st.markdown("### User Health Metrics Visualization") 
    
    # Create a bar chart for health metrics
    metrics = {
        "Age": age,
        "Height (cm)": height,
        "Weight (kg)": weight,
        "Steps Taken": steps,
        "Heart Rate (bpm)": hear_rate
    }
    
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values(), color=['#0072B5', '#0095D9', '#00BFFF', '#4CAFEA', '#80D5FF'])
    ax.set_ylabel('Value')
    ax.set_title('User Health Metrics')
    
    # Show the plot in Streamlit
    st.pyplot(fig)

    # Optional: Add additional visualizations such as a pie chart for activity distribution
    st.markdown("### Activity Distribution")
    activity_data = df['activity'].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.pie(activity_data, labels=activity_data.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig2)

    # Optional: Provide a summary of user health metrics
    st.markdown("### Summary of Your Health Metrics")
    for metric, value in metrics.items():
        st.markdown(f"**{metric}:** {value}")

    # Button to generate detailed report
    if st.sidebar.button("Generate Detailed Report"):
        report = StringIO()
        report.write("### Detailed Health Report\n")
        report.write(f"**User Profile:**\n")
        report.write(f"- Age: {age}\n")
        report.write(f"- Gender: {gender}\n")
        report.write(f"- Height: {height} cm\n")
        report.write(f"- Weight: {weight} kg\n")
        report.write(f"- Steps Taken Today: {steps}\n")
        report.write(f"- Heart Rate: {hear_rate} bpm\n\n")
        report.write("### Recommendations:\n")
        for item in recommendations:
            report.write(f"- {item}\n")
        report.write("\n### Predicted Calories Burned:\n")
        report.write(f"**{predicted_calories:.2f} kcal**\n")

        # Create a downloadable link for the report
        st.download_button("Download Detailed Report", data=report.getvalue(), file_name="health_report.txt", mime="text/plain")

    # Additional suggestions based on health metrics
    if weight > 100:  # Example threshold for overweight
        st.markdown("### Weight Management Suggestions:")
        st.write("To support weight loss, focus on a diet that is lower in calories but rich in nutrients. Aim for a balanced intake of lean proteins, plenty of vegetables, and whole grains. Avoid sugary drinks and snacks, and consider meal prepping to help control portions.")

    if hear_rate > 100:  # Example threshold for elevated heart rate
        st.markdown("### Heart Rate Management Suggestions:")
        st.write("It's important to monitor your heart rate, especially during physical activity. Consider incorporating activities like yoga or swimming, which can help lower your heart rate over time. Always consult with a healthcare professional if you have concerns about your heart rate.")

    if age > 60:
        st.markdown("### Senior Health Recommendations:")
        st.write("As you age, focus on maintaining muscle mass through strength training and flexibility exercises. Ensure you are getting enough calcium and vitamin D in your diet to support bone health. Staying socially active is also crucial for overall well-being.")

    # Optional: Add links to resources or further reading
    st.markdown("### Additional Resources:")
    st.write("[American Heart Association](https://www.heart.org) - For heart health tips and guidelines.")
    st.write("[ChooseMyPlate.gov](https://www.choosemyplate.gov) - For dietary recommendations and meal planning.")
    st.write("[CDC Physical Activity Guidelines](https://www.cdc.gov/physicalactivity/guidelines/index.html) - For guidelines on physical activity.")
