import os
import json
import pandas as pd
import shap
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

# Function to generate personalized health recommendations using the LLM with few-shot prompting
def get_health_recommendation(user_data):
    # Few-shot examples to guide the LLM
    few_shot_examples = (
        "Example 1:\n"
        "User Data:\n"
        "Age: 30, Gender: Male, Height: 175 cm, Weight: 70 kg, Steps: 8000, Activity: Running 3 METs\n"
        "Recommendation: To enhance cardiovascular health, aim to increase your daily step count to 10,000 steps. "
        "Incorporate strength training exercises into your routine at least twice a week. Focus on a balanced diet rich in vegetables, lean protein, "
        "and whole grains, and stay hydrated by drinking plenty of water.\n\n"
        
        "Example 2:\n"
        "User Data:\n"
        "Age: 25, Gender: Female, Height: 160 cm, Weight: 60 kg, Steps: 5000, Activity: Sitting\n"
        "Recommendation: To improve overall well-being, try to integrate more physical activity into your daily routine. "
        "Aim for at least 150 minutes of moderate aerobic activity each week and include strength training exercises at least twice a week. "
        "Consider a diet that emphasizes fruits, vegetables, whole grains, and low-fat dairy products to maintain a healthy weight.\n\n"
        
        "Example 3:\n"
        "User Data:\n"
        "Age: 45, Gender: Male, Height: 180 cm, Weight: 90 kg, Steps: 2000, Activity: Lying\n"
        "Recommendation: It's crucial to increase your activity level. Start with light exercises, such as walking for 10-15 minutes daily, and gradually work your way up. "
        "Focus on a diet rich in fiber, lean meats, and healthy fats to support weight loss and overall health.\n\n"
    )

    # Prepare the prompt for the LLM based on user data
    prompt = (
        f"{few_shot_examples}"
        f"Based on the following user information:\n"
        f"Age: {user_data['age']} years\n"
        f"Gender: {'Male' if user_data['gender'] == 1 else 'Female'}\n"
        f"Height: {user_data['height']} cm\n"
        f"Weight: {user_data['weight']} kg\n"
        f"Steps taken today: {user_data['steps']}\n"
        f"Activity type: {user_data['activity']}\n\n"
        "What personalized health recommendations can you provide based on this information, including required activity, diet suggestions, and other health tips?"
    )
    
    # Call the LLM with the prompt
    response = model.invoke([{"role": "user", "content": prompt}])
    return response.content

# Streamlit UI
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f8ff;  /* Light background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Personalized Health Prediction and Recommendation System")
st.markdown("""
    This application helps you predict your caloric burn based on your health metrics and physical activity. 
    Please enter your information in the sidebar to receive personalized health interventions.
""")

# Sidebar for user input
st.sidebar.header("Enter Your Health Information")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
steps = st.sidebar.number_input("Steps Taken Today", min_value=0, value=1000)
hear_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=30, value=70)
device = st.sidebar.selectbox("Device", options=["apple watch", "fitbit"])
activity = st.sidebar.selectbox("Activity Type", options=["Lying", "Sitting", "Self Pace walk", "Running 3 METs"])

# Button to generate ```python
# Button to generate recommendation
if st.sidebar.button("Get Recommendation"):
    # Prepare user data for prediction
    user_data = {
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'height': height,
        'weight': weight,
        'steps': steps,
        'hear_rate': hear_rate,
        'device': device,
        'activity': activity
    }

    # Get health recommendation
    recommendation = get_health_recommendation(user_data)
    
    # Display the recommendation in the main area
    st.subheader("Personalized Health Recommendation:")
    st.write(recommendation)
    
    # Display predicted calories burned
    predicted_calories = pipeline.predict(pd.DataFrame([user_data]))[0]
    st.markdown(f"### Predicted Calories Burned: **{predicted_calories:.2f} kcal**")

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
    ax.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'purple', 'red'])
    ax.set_ylabel('Value')
    ax.set_title('User Health Metrics')
    
    # Show the plot in Streamlit
    st.pyplot(fig)

    # Optional: Add additional visualizations such as a pie chart for activity distribution
    st.markdown("### Activity Distribution")
    activity_data = df['activity'].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.pie(activity_data, labels=activity_data.index, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig2)

    # Optional: Provide a summary of user health metrics
    st.markdown("### Summary of Your Health Metrics")
    st.write(metrics)

    # Optional: Provide personalized activity suggestions based on user input
    if steps < 5000:
        st.markdown("### Suggestions for Increasing Activity:")
        st.write("Consider adding short walks into your daily routine, taking the stairs instead of the elevator, or engaging in light exercises like stretching or yoga to boost your step count.")
    elif steps >= 5000 and steps < 10000:
        st.markdown("### Suggestions for Maintaining Activity:")
        st.write("You're on the right track! To reach your daily fitness goals, aim to include more physical activities such as jogging, cycling, or group sports. Consistency is key to improving your overall health.")
    else:
        st.markdown("### Great Job!")
        st.write("You have achieved a solid level of physical activity. Keep it up! Consider exploring new activities, such as hiking, swimming, or fitness classes, to maintain motivation and diversity in your fitness routine.")

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

# Note: Ensure that the user has a way to provide feedback or ask questions about the recommendations.
