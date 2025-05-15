import os
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from src.encrypt import decrypt_data
from src.auth_helpers import get_access_token
from utils.chat_model import AIGatewayLangchainChatOpenAI

# Load configuration from JSON and environment
with open('secret/url.json', 'r') as file:
    url_data = json.load(file)
AI_GATEWAY_BASE_URL = url_data.get("ai_gateway")
issuer_url = url_data.get("issuer_url")

# Load the .env file from the specified directory
load_dotenv('secret/.env')  # Specify the full path to the .env file

# Retrieve and decrypt the environment variables
loaded_fernet_key = os.getenv('FERNET_KEY').encode()
loaded_encrypted_client_id = os.getenv('ENCRYPTED_CLIENT_ID').encode()
loaded_encrypted_client_secret = os.getenv('ENCRYPTED_CLIENT_SECRET').encode()

client_id = decrypt_data(loaded_encrypted_client_id, loaded_fernet_key)
client_secret = decrypt_data(loaded_encrypted_client_secret, loaded_fernet_key)

access_token = get_access_token(client_id, client_secret, issuer_url)

model = AIGatewayLangchainChatOpenAI(
    access_token=access_token, base_url=AI_GATEWAY_BASE_URL, model="o1-2024-12-17", deere_ai_gateway_registration_id="graphics-quality-check"
)

def invoke_model(data_frame, prompt):
    # Convert DataFrame to JSON string to send to the model
    data_json = data_frame.to_json(orient='records')
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "text",  # Change 'data' to 'text'
                "text": data_json,  # Include the JSON string in the message
            },
        ],
    )
    response = model.invoke([message])
    return response.content

def read_excel_file(file_path, sheet_name=None):
    try:
        # Read the specified sheet into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return None

def process_excel_with_prompt(file_path, sheet_name, requirements):
    df = read_excel_file(file_path, sheet_name)
    if df is None:
        return "Excel file reading failed."
    
    results = {}
    
    # Assuming that the only relevant column is "Anchor Text"
    anchor_text_column = "Anchor Text"  # Ensure this matches your sheet
    #requirements_column= "Requirement"

    # Constructing the prompt with the provided requirements
    prompt = f'''You are an expert in creating schematic legends and specializing in data mapping.

you will find a column named {requirements_input} that contains alphanumeric characters or values for which we need to find legend details in the code.

And in the sheet you will find The column {anchor_text_column} contains alphanumeric characters along with their full forms or legend values, which we will use for mapping.

Your task is to extract all entries from the {anchor_text_column} that match any of the requirements listed in the {requirements_input}. Please provide the entire content of any matching cells.

For example, if the requirement is "A046", you should search for "A046" in the {anchor_text_column}. If found, return all related entries, such as:

0A046—Drive Train Domain Control Unit (5-V Output) Circuit Test
A046—Drive Train Domain Control Unit (12-V Supply) Circuit Test
A046—Drive Train Domain Control Unit Electrical Schematic (12-V Supply)
A046—Drive Train Domain Control Unit Electrical Schematic (5-V Output)
A046—Drive Train Domain Control Unit
X870-8—Power Supply for Drive Train and Transmission Control Unit (A046, A062, A065)
These are the mappings for "A046" obtained from the {anchor_text_column} column.

Note: It is possible that not all requirements will have matching entries in the {anchor_text_column}.
Format everything as HTML that can be used in a website. 
Place the description in a <div> element.
'''

    # Invoke the model with the prompt and data from the DataFrame
    result = invoke_model(df, prompt)
    results[sheet_name] = result
    
    return results

# Example usage
excel_file_path = r"""C:\Users\W4FGXUV\Downloads\Legend_creation\extracted_anchors.xlsx"""  # Update with your actual file path
sheet_name = "Sheet1"  # Specify your sheet name here

# Input field for requirements
requirements_input = """A046
A047
A136
Alternator
B028-1
B028-2
Battery
Battery Cut-Off Relay
Cab Switch 
Circuit 
Diode PLBU
Drive Train Domain CAN Bus
Drive Train Domain Control Unit
F103
G001
G001-2
G004
Ground Point
Ground Point (Cab)
Ground Point (Roof)
Hydraulic and Options Domain CAN Bus"""

# Split the input into a list
requirements = [req.strip() for req in requirements_input.splitlines() if req.strip()]

# Process the Excel file with the provided requirements
results = process_excel_with_prompt(excel_file_path, sheet_name, requirements)
print(results)
